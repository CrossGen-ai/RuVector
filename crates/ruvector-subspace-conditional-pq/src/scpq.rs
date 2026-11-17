//! Subspace-Conditional Product Quantization.
//!
//! For m=0 we store one codebook of K centroids (classic PQ).
//! For m>0 we store K codebooks indexed by the previous-subspace code p,
//! each with K centroids trained from the points that encoded to p at m-1.
//!
//! Encoding is greedy left-to-right: pick arg-min centroid at m=0, then at
//! each m>0 search only the codebook keyed by the just-assigned code.
//! Decoding is symmetric. Query LUT has shape `[m-1, K, K]` after m=0.

use crate::{codebook::train_kmeans, Codebook, Quantizer};

pub struct ScPq {
    d: usize,
    m: usize,
    k: usize,
    sub_d: usize,
    /// `books[0]` holds a single codebook (k centroids).
    /// `books[m]` for m>=1 holds `k` codebooks (one per previous code).
    books: Vec<Vec<Codebook>>,
}

impl ScPq {
    pub fn train(data: &[f32], d: usize, m: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(d % m == 0, "D must be divisible by M");
        assert!(k <= 256, "K must fit in u8");
        let sub_d = d / m;
        let n = data.len() / d;

        let mut books: Vec<Vec<Codebook>> = Vec::with_capacity(m);

        // Subspace 0: ordinary PQ codebook on sub-vectors.
        let mut sub0 = vec![0f32; n * sub_d];
        for i in 0..n {
            sub0[i * sub_d..(i + 1) * sub_d]
                .copy_from_slice(&data[i * d..i * d + sub_d]);
        }
        let book0 = train_kmeans(&sub0, sub_d, k, iters, seed);
        books.push(vec![book0]);

        // Encode subspace-0 codes for every training point so m=1 can bucket.
        let mut prev_codes = vec![0u8; n];
        for i in 0..n {
            let x = &sub0[i * sub_d..(i + 1) * sub_d];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for j in 0..k {
                let c = books[0][0].centroid(j);
                let mut s = 0f32;
                for t in 0..sub_d {
                    let dx = x[t] - c[t];
                    s += dx * dx;
                }
                if s < best_d {
                    best_d = s;
                    best = j as u8;
                }
            }
            prev_codes[i] = best;
        }

        // Subspaces 1..M: K codebooks keyed by prev code.
        for mi in 1..m {
            let mut layer = Vec::with_capacity(k);
            // Bucket points by prev_code.
            let mut buckets: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
            for (i, &p) in prev_codes.iter().enumerate() {
                buckets[p as usize].push(i);
            }
            // For each prev code p, train a codebook on this subspace's samples.
            for p in 0..k {
                let bucket = &buckets[p];
                if bucket.len() >= k {
                    let mut sub = vec![0f32; bucket.len() * sub_d];
                    for (bi, &i) in bucket.iter().enumerate() {
                        let src = &data[i * d + mi * sub_d..i * d + (mi + 1) * sub_d];
                        sub[bi * sub_d..(bi + 1) * sub_d].copy_from_slice(src);
                    }
                    let seed_m = seed.wrapping_add((mi as u64) * 1_000 + p as u64);
                    layer.push(train_kmeans(&sub, sub_d, k, iters, seed_m));
                } else {
                    // Rare: fall back to the global pool so the codebook still has K
                    // centroids. Recorded as a failure mode in the research doc.
                    let mut sub = vec![0f32; n * sub_d];
                    for i in 0..n {
                        let src = &data[i * d + mi * sub_d..i * d + (mi + 1) * sub_d];
                        sub[i * sub_d..(i + 1) * sub_d].copy_from_slice(src);
                    }
                    let seed_m = seed.wrapping_add((mi as u64) * 1_000 + p as u64 + 7919);
                    layer.push(train_kmeans(&sub, sub_d, k, iters, seed_m));
                }
            }
            // Re-encode prev_codes for the next subspace using this layer.
            let mut next_prev = vec![0u8; n];
            for i in 0..n {
                let x = &data[i * d + mi * sub_d..i * d + (mi + 1) * sub_d];
                let book = &layer[prev_codes[i] as usize];
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for j in 0..k {
                    let c = book.centroid(j);
                    let mut s = 0f32;
                    for t in 0..sub_d {
                        let dx = x[t] - c[t];
                        s += dx * dx;
                    }
                    if s < best_d {
                        best_d = s;
                        best = j as u8;
                    }
                }
                next_prev[i] = best;
            }
            prev_codes = next_prev;
            books.push(layer);
        }

        Self {
            d,
            m,
            k,
            sub_d,
            books,
        }
    }
}

impl ScPq {
    /// Beam-search encoding: carries the top `beam` partial code sequences
    /// forward across subspaces, picking the final one with the smallest
    /// cumulative squared-L2. beam=1 is greedy; beam=4 is a solid default.
    pub fn encode_beam(&self, data: &[f32], beam: usize) -> Vec<u8> {
        let n = data.len() / self.d;
        let mut codes = vec![0u8; n * self.m];
        // Per-vector buffers reused across iterations to keep allocs low.
        let mut cand: Vec<(f32, Vec<u8>)> = Vec::new();
        let mut expand: Vec<(f32, Vec<u8>, u8)> = Vec::new();
        for i in 0..n {
            cand.clear();
            // Initialize from m=0: K candidates, each a one-byte prefix.
            let x0 = &data[i * self.d..i * self.d + self.sub_d];
            for j in 0..self.k {
                let c = self.books[0][0].centroid(j);
                let mut s = 0f32;
                for t in 0..self.sub_d {
                    let dx = x0[t] - c[t];
                    s += dx * dx;
                }
                cand.push((s, vec![j as u8]));
            }
            // Keep top beam.
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            cand.truncate(beam);

            for mi in 1..self.m {
                let x =
                    &data[i * self.d + mi * self.sub_d..i * self.d + (mi + 1) * self.sub_d];
                expand.clear();
                for (prev_cost, prev_codes) in cand.iter() {
                    let prev = *prev_codes.last().unwrap();
                    let book = &self.books[mi][prev as usize];
                    for j in 0..self.k {
                        let c = book.centroid(j);
                        let mut s = 0f32;
                        for t in 0..self.sub_d {
                            let dx = x[t] - c[t];
                            s += dx * dx;
                        }
                        expand.push((*prev_cost + s, prev_codes.clone(), j as u8));
                    }
                }
                expand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                expand.truncate(beam);
                cand.clear();
                for (cost, mut prev_codes, next) in expand.drain(..) {
                    prev_codes.push(next);
                    cand.push((cost, prev_codes));
                }
            }
            let best = cand
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            for mi in 0..self.m {
                codes[i * self.m + mi] = best.1[mi];
            }
        }
        codes
    }
}

impl Quantizer for ScPq {
    fn encode(&self, data: &[f32]) -> Vec<u8> {
        // Default encoding uses beam=4 — a sweet spot between greedy's local
        // bias and larger beams whose encode cost grows linearly in `beam`.
        self.encode_beam(data, 4)
    }

    fn build_lut(&self, query: &[f32]) -> Vec<f32> {
        // Layout:
        //   lut[0 * K*K + 0*K + j] = dist(q0, book0[j])     (prev ignored)
        //   lut[mi * K*K + p*K + j] = dist(q_mi, book_{mi,p}[j])   for mi>=1
        let mut lut = vec![0f32; self.m * self.k * self.k];
        // m=0
        {
            let q = &query[0..self.sub_d];
            for j in 0..self.k {
                let c = self.books[0][0].centroid(j);
                let mut s = 0f32;
                for t in 0..self.sub_d {
                    let dx = q[t] - c[t];
                    s += dx * dx;
                }
                // broadcast over "prev" axis so score() does one indexing path
                for p in 0..self.k {
                    lut[p * self.k + j] = s;
                }
            }
        }
        for mi in 1..self.m {
            let q = &query[mi * self.sub_d..(mi + 1) * self.sub_d];
            for p in 0..self.k {
                let book = &self.books[mi][p];
                for j in 0..self.k {
                    let c = book.centroid(j);
                    let mut s = 0f32;
                    for t in 0..self.sub_d {
                        let dx = q[t] - c[t];
                        s += dx * dx;
                    }
                    lut[mi * self.k * self.k + p * self.k + j] = s;
                }
            }
        }
        lut
    }

    fn score(&self, lut: &[f32], codes: &[u8]) -> Vec<f32> {
        let n = codes.len() / self.m;
        let mut out = vec![0f32; n];
        let kk = self.k * self.k;
        for i in 0..n {
            let mut s = 0f32;
            let mut prev = 0usize;
            for mi in 0..self.m {
                let cur = codes[i * self.m + mi] as usize;
                s += lut[mi * kk + prev * self.k + cur];
                prev = cur;
            }
            out[i] = s;
        }
        out
    }

    fn m(&self) -> usize {
        self.m
    }
    fn k(&self) -> usize {
        self.k
    }
    fn d(&self) -> usize {
        self.d
    }
}
