//! A compact IVF-PQ index.
//!
//! * IVF: k-means coarse centroids partition the dataset into posting lists.
//! * PQ: residual vectors (vector − coarse centroid) are split into `m`
//!   subvectors, each quantized by a per-subspace k-means codebook of size
//!   `ksub` (= 256 so a code fits in u8).
//!
//! The ADC distance for a query `q` against a code `c = (c_1 .. c_m)` is
//! `sum_j tab_j[c_j]` where `tab_j[k] = ||q_sub_j − centroid_jk||^2`.
//!
//! Posting lists are laid out in fixed-size *blocks*. Per-block, we store the
//! element-wise minimum PQ code (per subspace), which produces the smallest
//! possible ADC score any code in that block could achieve against *any*
//! future query — the "block-max" lower bound. (In BMW terms we are tracking
//! a *lower bound on distance*, equivalent to an *upper bound on similarity*.)

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub struct PqParams {
    pub m: usize,       // number of subspaces
    pub ksub: usize,    // centroids per subspace (<=256 so code is u8)
    pub kmeans_iter: usize,
}

impl Default for PqParams {
    fn default() -> Self { Self { m: 8, ksub: 256, kmeans_iter: 10 } }
}

pub struct IvfPqBuilder {
    pub nlist: usize,
    pub kmeans_iter: usize,
    pub block_size: usize,
    pub pq: PqParams,
    pub seed: u64,
}

impl Default for IvfPqBuilder {
    fn default() -> Self {
        Self { nlist: 64, kmeans_iter: 10, block_size: 64, pq: PqParams::default(), seed: 0xC0FFEE }
    }
}

pub struct IvfPq {
    pub dim: usize,
    pub nlist: usize,
    pub block_size: usize,

    // Coarse centroids: [nlist, dim]
    pub coarse: Vec<f32>,

    // PQ setup
    pub m: usize,
    pub ksub: usize,
    pub d_sub: usize,
    // Flat codebooks: pq_codebooks[j * ksub * d_sub .. (j+1) * ksub * d_sub]
    pub pq_codebooks: Vec<f32>,

    // Posting lists — one per coarse centroid.
    // Each posting list is a sequence of blocks. Each block holds up to
    // `block_size` codes (one byte per subspace) and the same number of vector
    // ids. `block_lb_codes[b]` holds the element-wise min code per subspace
    // used for the lower bound.
    pub lists: Vec<PostingList>,
}

pub struct PostingList {
    pub blocks: Vec<Block>,
}

pub struct Block {
    pub codes: Vec<u8>,   // length = block_size_actual * m
    pub ids: Vec<u32>,    // length = block_size_actual
    pub lb_codes: Vec<u8>, // length = m — element-wise min code per subspace
    /// Per-subspace bitmask of codes present in this block.
    /// Layout: `code_masks[j * 4 + w]` is 64-bit word `w` of subspace `j`'s
    /// 256-bit mask. Lets a scanner iterate only *unique* codes per subspace.
    pub code_masks: Vec<u64>,
}

impl IvfPqBuilder {
    pub fn train(&self, data: &[f32], dim: usize) -> IvfPq {
        assert!(data.len() % dim == 0);
        let n = data.len() / dim;
        assert!(n >= self.nlist, "need >= nlist points to train");
        assert!(dim % self.pq.m == 0, "dim must be divisible by m");

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        // --- coarse k-means ---
        let coarse = kmeans(data, n, dim, self.nlist, self.kmeans_iter, &mut rng);
        let assign = assign_nearest(data, n, dim, &coarse, self.nlist);

        // --- residuals ---
        let mut residuals = vec![0f32; n * dim];
        for i in 0..n {
            let c = assign[i] as usize;
            for d in 0..dim {
                residuals[i * dim + d] = data[i * dim + d] - coarse[c * dim + d];
            }
        }

        // --- per-subspace k-means on residuals ---
        let d_sub = dim / self.pq.m;
        let ksub = self.pq.ksub.min(256);
        let mut pq_codebooks = vec![0f32; self.pq.m * ksub * d_sub];
        let mut codes = vec![0u8; n * self.pq.m];
        let mut sub_buf = vec![0f32; n * d_sub];
        for j in 0..self.pq.m {
            // gather subvectors
            for i in 0..n {
                for s in 0..d_sub {
                    sub_buf[i * d_sub + s] = residuals[i * dim + j * d_sub + s];
                }
            }
            let cb = kmeans(&sub_buf, n, d_sub, ksub, self.pq.kmeans_iter, &mut rng);
            let a = assign_nearest(&sub_buf, n, d_sub, &cb, ksub);
            for i in 0..n {
                codes[i * self.pq.m + j] = a[i] as u8;
            }
            let dst = &mut pq_codebooks[j * ksub * d_sub..(j + 1) * ksub * d_sub];
            dst.copy_from_slice(&cb);
        }

        // --- posting lists ---
        let mut lists: Vec<Vec<(u32, Vec<u8>)>> = (0..self.nlist).map(|_| Vec::new()).collect();
        for i in 0..n {
            let c = assign[i] as usize;
            let code = codes[i * self.pq.m..(i + 1) * self.pq.m].to_vec();
            lists[c].push((i as u32, code));
        }

        let lists: Vec<PostingList> = lists.into_iter().map(|entries| {
            let mut blocks = Vec::new();
            for chunk in entries.chunks(self.block_size) {
                let bs = chunk.len();
                let mut block_codes = vec![0u8; bs * self.pq.m];
                let mut ids = Vec::with_capacity(bs);
                let mut lb = vec![u8::MAX; self.pq.m];
                let mut masks = vec![0u64; self.pq.m * 4];
                for (k, (id, code)) in chunk.iter().enumerate() {
                    ids.push(*id);
                    for j in 0..self.pq.m {
                        let c = code[j];
                        block_codes[k * self.pq.m + j] = c;
                        if c < lb[j] { lb[j] = c; }
                        let word = (c as usize) >> 6;
                        let bit = (c as usize) & 63;
                        masks[j * 4 + word] |= 1u64 << bit;
                    }
                }
                blocks.push(Block { codes: block_codes, ids, lb_codes: lb, code_masks: masks });
            }
            PostingList { blocks }
        }).collect();

        IvfPq {
            dim,
            nlist: self.nlist,
            block_size: self.block_size,
            coarse,
            m: self.pq.m,
            ksub,
            d_sub,
            pq_codebooks,
            lists,
        }
    }
}

impl IvfPq {
    /// Pick top `nprobe` coarse centroids for `query`.
    pub fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        debug_assert_eq!(query.len(), self.dim);
        let mut d: Vec<(usize, f32)> = (0..self.nlist).map(|c| {
            let mut s = 0f32;
            for i in 0..self.dim {
                let x = query[i] - self.coarse[c * self.dim + i];
                s += x * x;
            }
            (c, s)
        }).collect();
        d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        d.truncate(nprobe);
        d.into_iter().map(|(c, _)| c).collect()
    }

    /// Build the ADC lookup table for a given query and coarse centroid.
    /// Returns `tab[j * ksub + k] = ||q_sub_j − c_j - cb_j[k]||^2`.
    pub fn build_lut(&self, query: &[f32], c: usize) -> Vec<f32> {
        let mut tab = vec![0f32; self.m * self.ksub];
        let dim = self.dim;
        for j in 0..self.m {
            let off = j * self.d_sub;
            let cb = &self.pq_codebooks[j * self.ksub * self.d_sub..(j + 1) * self.ksub * self.d_sub];
            for k in 0..self.ksub {
                let mut s = 0f32;
                for s_i in 0..self.d_sub {
                    let q = query[off + s_i] - self.coarse[c * dim + off + s_i];
                    let diff = q - cb[k * self.d_sub + s_i];
                    s += diff * diff;
                }
                tab[j * self.ksub + k] = s;
            }
        }
        tab
    }
}

// ---------- helpers ----------
fn kmeans(data: &[f32], n: usize, dim: usize, k: usize, iters: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    // seed with reservoir sample
    let mut centroids = vec![0f32; k * dim];
    let mut picked: Vec<usize> = (0..n).collect();
    picked.shuffle(rng);
    for i in 0..k {
        let src = picked[i % n];
        centroids[i * dim..(i + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }
    for _ in 0..iters {
        let assign = assign_nearest(data, n, dim, &centroids, k);
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += data[i * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            } else {
                let src = picked[rng.gen_range(0..n)];
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
            }
        }
    }
    centroids
}

fn assign_nearest(data: &[f32], n: usize, dim: usize, centroids: &[f32], k: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    for i in 0..n {
        let x = &data[i * dim..(i + 1) * dim];
        let mut best = f32::INFINITY;
        let mut arg = 0u32;
        for c in 0..k {
            let cc = &centroids[c * dim..(c + 1) * dim];
            let mut s = 0f32;
            for d in 0..dim {
                let diff = x[d] - cc[d];
                s += diff * diff;
            }
            if s < best { best = s; arg = c as u32; }
        }
        out[i] = arg;
    }
    out
}
