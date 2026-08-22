//! Flat RVQ index: encode all vectors, LUT-scan on query.

use crate::{Quantizer, Rvq};

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub id: u32,
    pub score: f32,
}

/// A flat (brute-force scan) RVQ-compressed index. Storage is
/// `n · stages` bytes plus the codebook (`stages · k · dim · 4` bytes,
/// amortised across all vectors).
pub struct RvqIndex {
    quantizer: Rvq,
    /// Row-major `n × stages` codes.
    codes: Vec<u8>,
    /// Precomputed query-independent per-centroid squared norms LUT
    /// (shape `stages · k`), cached once at build.
    norm_lut: Vec<f32>,
    n: usize,
}

impl RvqIndex {
    pub fn build(quantizer: Rvq, data: &[f32]) -> Self {
        let d = quantizer.dim();
        assert!(data.len() % d == 0);
        let n = data.len() / d;
        let stages = quantizer.code_bytes();
        let mut codes = vec![0u8; n * stages];
        for i in 0..n {
            quantizer.encode(&data[i * d..(i + 1) * d],
                             &mut codes[i * stages..(i + 1) * stages]).unwrap();
        }
        let norm_lut = quantizer.l2_norm_lut();
        Self { quantizer, codes, norm_lut, n }
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn quantizer(&self) -> &Rvq { &self.quantizer }

    /// Search by cosine / inner-product (assumes `q` is L2-normalised if you
    /// want cosine — this method just returns `⟨q, x_i⟩` estimates).
    /// Returns top-k results sorted by descending score.
    pub fn search_ip(&self, q: &[f32], k: usize) -> Vec<SearchResult> {
        let lut = self.quantizer.ip_lut(q);
        let stages = self.quantizer.code_bytes();
        let mut top: Vec<SearchResult> = Vec::with_capacity(k + 1);

        for i in 0..self.n {
            let code = &self.codes[i * stages..(i + 1) * stages];
            let s = self.quantizer.estimate_ip(&lut, code);
            insert_top(&mut top, SearchResult { id: i as u32, score: s }, k, /*maximise=*/true);
        }
        top
    }

    /// Search by squared L2 distance. Returns top-k by *ascending* distance
    /// (smallest first).
    pub fn search_l2(&self, q: &[f32], k: usize) -> Vec<SearchResult> {
        let ip_lut = self.quantizer.ip_lut(q);
        let mut q_sq = 0f32;
        for &v in q { q_sq += v * v; }
        let stages = self.quantizer.code_bytes();
        let mut top: Vec<SearchResult> = Vec::with_capacity(k + 1);

        for i in 0..self.n {
            let code = &self.codes[i * stages..(i + 1) * stages];
            let s = self.quantizer.estimate_l2(q_sq, &ip_lut, &self.norm_lut, code);
            insert_top(&mut top, SearchResult { id: i as u32, score: s }, k, /*maximise=*/false);
        }
        top
    }

    /// Bytes held by the encoded vectors (excludes codebook overhead).
    pub fn code_bytes(&self) -> usize { self.codes.len() }

    /// Two-stage search: retrieve top-`rerank_pool` by RVQ-estimated L2,
    /// then re-score those candidates against the *full-precision* vectors
    /// supplied in `full` (row-major `n × d`) and return top-`k`.
    ///
    /// This is the standard production recipe (FAISS `RQ` + `IndexRefine`,
    /// ScaNN `AH+reordering`). RVQ handles the coarse filter cheaply;
    /// exact distance decides the final ranking.
    pub fn search_l2_rerank(&self, q: &[f32], k: usize, rerank_pool: usize, full: &[f32]) -> Vec<SearchResult> {
        let d = self.quantizer.dim();
        assert_eq!(full.len(), self.n * d);
        let pool = rerank_pool.max(k);
        let coarse = self.search_l2(q, pool);
        let mut rescored: Vec<SearchResult> = coarse.into_iter().map(|c| {
            let x = &full[c.id as usize * d..(c.id as usize + 1) * d];
            let mut s = 0f32;
            for j in 0..d { let e = x[j] - q[j]; s += e * e; }
            SearchResult { id: c.id, score: s }
        }).collect();
        rescored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        rescored.truncate(k);
        rescored
    }
}

fn insert_top(top: &mut Vec<SearchResult>, r: SearchResult, k: usize, maximise: bool) {
    // sorted insertion — small k, linear scan is fine
    let cmp = |a: f32, b: f32| if maximise { a > b } else { a < b };
    let pos = top.iter().position(|x| cmp(r.score, x.score)).unwrap_or(top.len());
    top.insert(pos, r);
    if top.len() > k { top.pop(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RvqConfig;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn brute_l2_topk(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<u32> {
        let n = data.len() / d;
        let mut scored: Vec<(u32, f32)> = (0..n).map(|i| {
            let mut s = 0f32;
            for j in 0..d { let e = data[i*d+j] - q[j]; s += e*e; }
            (i as u32, s)
        }).collect();
        scored.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(i,_)| i).collect()
    }

    #[test]
    fn recall_beats_random_on_gaussian() {
        let n = 2000; let d = 32; let k = 10;
        let data = synth(n, d, 11);
        let queries = synth(50, d, 22);

        let rvq = Rvq::train(&data, n, d, &RvqConfig{stages:8,k:64,kmeans_iters:12,seed:99}).unwrap();
        let idx = RvqIndex::build(rvq, &data);

        let mut hits = 0usize;
        for qi in 0..50 {
            let q = &queries[qi*d..(qi+1)*d];
            let truth: std::collections::HashSet<u32> = brute_l2_topk(&data, d, q, k).into_iter().collect();
            let got = idx.search_l2(q, k);
            for r in &got { if truth.contains(&r.id) { hits += 1; } }
        }
        let recall = hits as f32 / (50 * k) as f32;
        // Random baseline is k/n = 10/2000 = 0.005. RVQ (8x64) should easily
        // clear 0.5 recall@10 on i.i.d. Gaussians. The bench captures real
        // measured numbers; this test just proves the pipeline works.
        assert!(recall > 0.3, "recall@10 = {recall:.3} — RVQ pipeline is broken");
    }
}
