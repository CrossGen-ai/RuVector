//! Classic Product Quantization (Jégou, Douze, Schmid 2011).
//!
//! Splits a `d`-vector into `m` contiguous subvectors of dimension `ds = d/m`,
//! trains a `k=256` codebook per subspace via k-means, and encodes each
//! subvector as the index of its nearest centroid (1 byte).

use crate::kmeans::kmeans;
use crate::Quantizer;
use serde::{Deserialize, Serialize};

const K: usize = 256;

#[derive(Clone, Serialize, Deserialize)]
pub struct Pq {
    pub d: usize,
    pub m: usize,
    pub ds: usize,
    /// `m * K * ds` row-major: centroids[sub][k] is at `(sub*K + k) * ds`.
    pub centroids: Vec<f32>,
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Pq {
    pub fn new(d: usize, m: usize) -> Self {
        assert!(d % m == 0, "d={} not divisible by m={}", d, m);
        Self {
            d,
            m,
            ds: d / m,
            centroids: Vec::new(),
            kmeans_iters: 20,
            seed: 0xC0FFEE,
        }
    }
}

impl Quantizer for Pq {
    fn fit(&mut self, data: &[f32], n: usize, d: usize) {
        assert_eq!(d, self.d);
        assert_eq!(data.len(), n * d);

        self.centroids = vec![0.0f32; self.m * K * self.ds];

        // Slice each subspace into a contiguous training buffer (n × ds).
        let mut sub = vec![0.0f32; n * self.ds];
        for s in 0..self.m {
            for i in 0..n {
                let src = &data[i * d + s * self.ds..i * d + (s + 1) * self.ds];
                sub[i * self.ds..(i + 1) * self.ds].copy_from_slice(src);
            }
            let cents = kmeans(&sub, n, self.ds, K, self.kmeans_iters, self.seed ^ (s as u64));
            self.centroids[s * K * self.ds..(s + 1) * K * self.ds].copy_from_slice(&cents);
        }
    }

    fn encode(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(x.len(), self.d);
        debug_assert_eq!(out.len(), self.m);
        for s in 0..self.m {
            let xs = &x[s * self.ds..(s + 1) * self.ds];
            let cb = &self.centroids[s * K * self.ds..(s + 1) * K * self.ds];
            let mut best = 0u8;
            let mut bd = f32::INFINITY;
            for c in 0..K {
                let cv = &cb[c * self.ds..(c + 1) * self.ds];
                let mut dd = 0.0f32;
                for j in 0..self.ds {
                    let e = xs[j] - cv[j];
                    dd += e * e;
                }
                if dd < bd {
                    bd = dd;
                    best = c as u8;
                }
            }
            out[s] = best;
        }
    }

    fn decode(&self, code: &[u8], out: &mut [f32]) {
        debug_assert_eq!(code.len(), self.m);
        debug_assert_eq!(out.len(), self.d);
        for s in 0..self.m {
            let c = code[s] as usize;
            let cv = &self.centroids[(s * K + c) * self.ds..(s * K + c + 1) * self.ds];
            out[s * self.ds..(s + 1) * self.ds].copy_from_slice(cv);
        }
    }

    fn adc(&self, query: &[f32], code: &[u8]) -> f32 {
        // Build per-subspace lookup table on the fly: 256 squared dists.
        // (For batch search you'd cache this once per query across many codes.)
        let mut sum = 0.0f32;
        for s in 0..self.m {
            let qs = &query[s * self.ds..(s + 1) * self.ds];
            let cb = &self.centroids[s * K * self.ds..(s + 1) * K * self.ds];
            let c = code[s] as usize;
            let cv = &cb[c * self.ds..(c + 1) * self.ds];
            for j in 0..self.ds {
                let e = qs[j] - cv[j];
                sum += e * e;
            }
        }
        sum
    }

    fn build_lut(&self, query: &[f32], lut: &mut [f32]) {
        debug_assert_eq!(query.len(), self.d);
        debug_assert_eq!(lut.len(), self.m * K);
        for s in 0..self.m {
            let qs = &query[s * self.ds..(s + 1) * self.ds];
            let cb = &self.centroids[s * K * self.ds..(s + 1) * K * self.ds];
            for c in 0..K {
                let cv = &cb[c * self.ds..(c + 1) * self.ds];
                let mut dd = 0.0f32;
                for j in 0..self.ds {
                    let e = qs[j] - cv[j];
                    dd += e * e;
                }
                lut[s * K + c] = dd;
            }
        }
    }
    fn m(&self) -> usize {
        self.m
    }
    fn d(&self) -> usize {
        self.d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn encode_decode_roundtrip_shapes() {
        let d = 16;
        let m = 4;
        let n = 600;
        let data = synth(n, d, 1);
        let mut pq = Pq::new(d, m);
        pq.fit(&data, n, d);
        let mut code = vec![0u8; m];
        let mut rec = vec![0.0f32; d];
        pq.encode(&data[..d], &mut code);
        pq.decode(&code, &mut rec);
        assert_eq!(code.len(), m);
        assert_eq!(rec.len(), d);
    }

    #[test]
    fn reconstruction_beats_random_codes() {
        // PQ-trained codes must outperform random 8-bit codes on MSE.
        let d = 32;
        let m = 8;
        let n = 1500;
        let data = synth(n, d, 2);
        let mut pq = Pq::new(d, m);
        pq.fit(&data, n, d);

        let mut code = vec![0u8; m];
        let mut rec = vec![0.0f32; d];
        let mut pq_err = 0.0f64;
        let mut rand_err = 0.0f64;
        let mut rng = StdRng::seed_from_u64(99);
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            pq.encode(x, &mut code);
            pq.decode(&code, &mut rec);
            for j in 0..d {
                pq_err += (x[j] - rec[j]) as f64 * (x[j] - rec[j]) as f64;
            }
            for s in 0..m {
                code[s] = rng.gen();
            }
            pq.decode(&code, &mut rec);
            for j in 0..d {
                rand_err += (x[j] - rec[j]) as f64 * (x[j] - rec[j]) as f64;
            }
        }
        assert!(pq_err < rand_err * 0.5, "pq={} random={}", pq_err, rand_err);
    }
}
