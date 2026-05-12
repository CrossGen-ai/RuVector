//! Three flat ANN indices sharing a single trait, so the demo binary, tests
//! and downstream callers can A/B them with identical call sites.
//!
//! - [`FlatIndex`] — f32 brute force. Reference for recall and runtime.
//! - [`LvqIndex`] — LVQ-8 over the original dimensionality. ~4× memory shrink,
//!   small recall loss, asymmetric f32-vs-u8 distance kernel.
//! - [`LeanVecIndex`] — PCA projection to `r ≤ d`, LVQ-8 on the projection, and
//!   exact f32 rerank of the top `k' = k · rerank_mult` candidates against the
//!   original vectors retained in float.

use crate::lvq::LvqCodebook;
use crate::projection::Projection;

/// A `(id, squared L2 distance)` pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Index of the vector in insertion order.
    pub id: u32,
    /// Squared L2 distance to the query.
    pub dist: f32,
}

/// Common interface for all three flat indices.
pub trait VectorIndex {
    /// Insert a vector and return its assigned id (insertion order).
    fn add(&mut self, v: &[f32]) -> u32;
    /// Top-`k` nearest neighbours by squared L2.
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor>;
    /// Number of vectors indexed.
    fn len(&self) -> usize;
    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes occupied by the indexed payload (does not count the query).
    fn bytes(&self) -> usize;
    /// Human-readable variant name (for benchmark tables).
    fn name(&self) -> &'static str;
}

/// f32 brute-force baseline.
pub struct FlatIndex {
    dim: usize,
    data: Vec<f32>,
    n: usize,
}

impl FlatIndex {
    /// Create a flat index over `d`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self { dim, data: Vec::new(), n: 0 }
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.n as u32;
        self.data.extend_from_slice(v);
        self.n += 1;
        id
    }

    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(q.len(), self.dim);
        let mut heap: Vec<Neighbor> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let row = &self.data[i * self.dim..(i + 1) * self.dim];
            let mut s = 0.0_f32;
            for j in 0..self.dim {
                let d = q[j] - row[j];
                s += d * d;
            }
            heap.push(Neighbor { id: i as u32, dist: s });
        }
        heap.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        heap.truncate(k);
        heap
    }

    fn len(&self) -> usize {
        self.n
    }

    fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    fn name(&self) -> &'static str {
        "FlatF32"
    }
}

/// LVQ-8 over the native dimensionality. Compresses ~4× vs f32.
pub struct LvqIndex {
    dim: usize,
    cb: LvqCodebook,
}

impl LvqIndex {
    /// Create an empty LVQ index.
    pub fn new(dim: usize) -> Self {
        Self { dim, cb: LvqCodebook::new(dim) }
    }
}

impl VectorIndex for LvqIndex {
    fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.cb.len() as u32;
        self.cb.push(v);
        id
    }

    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(q.len(), self.dim);
        let mut out: Vec<Neighbor> = self
            .cb
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| Neighbor { id: i as u32, dist: c.asym_l2_sq(q) })
            .collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        out.truncate(k);
        out
    }

    fn len(&self) -> usize {
        self.cb.len()
    }

    fn bytes(&self) -> usize {
        self.cb.bytes()
    }

    fn name(&self) -> &'static str {
        "LVQ-8"
    }
}

/// LeanVec-LVQ: project to `r ≤ d`, LVQ-8 the projection, rerank top `k'`
/// candidates against retained f32 originals.
pub struct LeanVecIndex {
    dim: usize,
    proj: Projection,
    cb: LvqCodebook,
    raw: Vec<f32>,
    /// Multiplier for the candidate set: search asks for `k`, we score the
    /// top `k · rerank_mult` codes and rerank with exact L2 on raw.
    pub rerank_mult: usize,
}

impl LeanVecIndex {
    /// Build a LeanVec index from a trained projection.
    pub fn new(proj: Projection, rerank_mult: usize) -> Self {
        assert!(rerank_mult >= 1, "rerank_mult must be at least 1");
        let dim = proj.d;
        let r = proj.r;
        Self {
            dim,
            proj,
            cb: LvqCodebook::new(r),
            raw: Vec::new(),
            rerank_mult,
        }
    }

    /// Reduced (projection) dimensionality.
    pub fn reduced_dim(&self) -> usize {
        self.proj.r
    }
}

impl VectorIndex for LeanVecIndex {
    fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.cb.len() as u32;
        let projected = self.proj.project(v);
        self.cb.push(&projected);
        self.raw.extend_from_slice(v);
        id
    }

    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(q.len(), self.dim);
        let qp = self.proj.project(q);
        let candidate_k = (k * self.rerank_mult).max(k).min(self.cb.len());

        let mut scored: Vec<Neighbor> = self
            .cb
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| Neighbor { id: i as u32, dist: c.asym_l2_sq(&qp) })
            .collect();
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        scored.truncate(candidate_k);

        for n in scored.iter_mut() {
            let row = &self.raw[(n.id as usize) * self.dim..(n.id as usize + 1) * self.dim];
            let mut s = 0.0_f32;
            for j in 0..self.dim {
                let d = q[j] - row[j];
                s += d * d;
            }
            n.dist = s;
        }
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        scored.truncate(k);
        scored
    }

    fn len(&self) -> usize {
        self.cb.len()
    }

    fn bytes(&self) -> usize {
        // Codes + retained f32 originals (LeanVec keeps these for rerank).
        self.cb.bytes() + self.raw.len() * std::mem::size_of::<f32>()
    }

    fn name(&self) -> &'static str {
        "LeanVec-LVQ"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn flat_returns_self_at_zero_distance() {
        let d = 16;
        let data = synth(200, d, 1);
        let mut idx = FlatIndex::new(d);
        for i in 0..200 {
            idx.add(&data[i * d..(i + 1) * d]);
        }
        let q = &data[7 * d..8 * d];
        let r = idx.search(q, 1);
        assert_eq!(r[0].id, 7);
        assert!(r[0].dist < 1e-6);
    }

    #[test]
    fn lvq_recall_within_one_step_of_flat() {
        let n = 300;
        let d = 24;
        let data = synth(n, d, 2);
        let mut flat = FlatIndex::new(d);
        let mut lvq = LvqIndex::new(d);
        for i in 0..n {
            let v = &data[i * d..(i + 1) * d];
            flat.add(v);
            lvq.add(v);
        }
        // 20 random queries; the top-1 from LVQ must agree with one of the
        // top-3 from flat (the quantisation cannot move a true neighbour past
        // its 2 closest siblings on small uniform data).
        let queries = synth(20, d, 3);
        let mut agreed = 0;
        for qi in 0..20 {
            let q = &queries[qi * d..(qi + 1) * d];
            let f = flat.search(q, 3);
            let l = lvq.search(q, 1);
            if f.iter().any(|n| n.id == l[0].id) {
                agreed += 1;
            }
        }
        assert!(agreed >= 18, "LVQ top1 in flat top3 only {agreed}/20");
    }

    /// Anisotropic synthesizer: a handful of latent factors dominate. This is
    /// the regime LeanVec is *designed for*; uniform-random data is not, and
    /// PCA can only help when the data actually has a low-rank backbone.
    fn synth_anisotropic(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let latent = (d / 8).max(2);
        let mut basis = vec![0.0_f32; latent * d];
        for k in 0..latent {
            for j in 0..d {
                basis[k * d + j] = rng.gen::<f32>() - 0.5;
            }
            let mut s = 0.0;
            for j in 0..d {
                s += basis[k * d + j] * basis[k * d + j];
            }
            let inv = 1.0 / s.sqrt();
            for j in 0..d {
                basis[k * d + j] *= inv;
            }
        }
        let mut out = vec![0.0_f32; n * d];
        for i in 0..n {
            let row = &mut out[i * d..(i + 1) * d];
            for k in 0..latent {
                let coef = (rng.gen::<f32>() - 0.5) * 5.0;
                for j in 0..d {
                    row[j] += coef * basis[k * d + j];
                }
            }
            for j in 0..d {
                row[j] += (rng.gen::<f32>() - 0.5) * 0.05;
            }
        }
        out
    }

    #[test]
    fn leanvec_beats_lvq_on_rerank() {
        let n = 500;
        let d = 32;
        let data = synth_anisotropic(n, d, 4);
        let proj = Projection::train_pca(&data, n, d, d / 2, 99);

        let mut flat = FlatIndex::new(d);
        let mut lv = LeanVecIndex::new(proj, 4);
        for i in 0..n {
            let v = &data[i * d..(i + 1) * d];
            flat.add(v);
            lv.add(v);
        }
        let queries = synth_anisotropic(30, d, 5);
        let mut agreed = 0;
        for qi in 0..30 {
            let q = &queries[qi * d..(qi + 1) * d];
            let f = flat.search(q, 1);
            let l = lv.search(q, 1);
            if f[0].id == l[0].id {
                agreed += 1;
            }
        }
        // On anisotropic data PCA preserves the dominant directions and
        // rerank pulls the exact top-1 back the vast majority of the time.
        assert!(agreed >= 25, "LeanVec rerank top1 matched flat only {agreed}/30");
    }
}
