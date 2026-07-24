//! Two-stage IVF retrieval over MUVERA FDE vectors, with optional
//! exact-MaxSim rerank.
//!
//! Pipeline:
//!
//! 1. Build the FDE encoder and encode every document.
//! 2. Run a small k-means (Lloyd) on the doc FDEs to get `n_lists`
//!    centroids. Store each doc under its nearest centroid.
//! 3. At query time, encode the query FDE, pick the top `n_probe`
//!    centroids by inner product, scan the union of their posting
//!    lists, keep the top-`candidates` by FDE inner product.
//! 4. If `rerank > 0`, take the top-`rerank` candidates and rescore
//!    them with exact MaxSim over the original token bags. Return
//!    top-`k` from that reranked pool.
//!
//! This is the shape of a production MUVERA pipeline (paper §4): the
//! FDE handles the coarse filter cheaply on a single-vector index; the
//! expensive full multi-vector MaxSim is only run on a small
//! reranking pool.

use crate::chamfer::maxsim;
use crate::fde::{dot, FdeEncoder, FdeParams, FdeSide};
use crate::retriever::{Document, MultiVectorRetriever, RetrievalHit};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Config for [`MuveraIvf`].
#[derive(Debug, Clone, Copy)]
pub struct IvfParams {
    /// Number of k-means centroids.
    pub n_lists: usize,
    /// Number of centroids to probe at query time.
    pub n_probe: usize,
    /// Candidates surfaced from the IVF stage.
    pub candidates: usize,
    /// Rerank top-`rerank` candidates with exact MaxSim. Set to 0 to
    /// disable rerank (return raw FDE order).
    pub rerank: usize,
    /// K-means iterations.
    pub kmeans_iters: usize,
    /// K-means / init seed.
    pub seed: u64,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self { n_lists: 16, n_probe: 4, candidates: 64, rerank: 32, kmeans_iters: 10, seed: 1 }
    }
}

/// MUVERA + IVF + exact-rerank pipeline.
#[derive(Debug)]
pub struct MuveraIvf {
    encoder: FdeEncoder,
    ivf: IvfParams,
    d_token: usize,
    fde_dim: usize,
    docs: Vec<Document>,          // stored for rerank stage
    doc_fdes: Vec<f32>,           // [n_docs × fde_dim], parallel to `docs`
    centroids: Vec<f32>,          // [n_lists × fde_dim]
    posting: Vec<Vec<u32>>,       // per-centroid list of doc indices (into `docs`)
}

impl MuveraIvf {
    pub fn new(fde_params: FdeParams, ivf_params: IvfParams) -> Self {
        let encoder = FdeEncoder::new(fde_params);
        let fde_dim = encoder.fde_dim();
        Self {
            encoder,
            ivf: ivf_params,
            d_token: fde_params.d,
            fde_dim,
            docs: Vec::new(),
            doc_fdes: Vec::new(),
            centroids: Vec::new(),
            posting: Vec::new(),
        }
    }

    pub fn fde_dim(&self) -> usize { self.fde_dim }
    pub fn params(&self) -> IvfParams { self.ivf }

    fn slice_of(&self, doc_idx: usize) -> &[f32] {
        &self.doc_fdes[doc_idx * self.fde_dim..(doc_idx + 1) * self.fde_dim]
    }
}

impl MultiVectorRetriever for MuveraIvf {
    fn name(&self) -> &'static str { "muvera_fde_ivf" }

    fn build(&mut self, corpus: &[Document]) {
        self.docs = corpus.to_vec();
        self.doc_fdes.clear();
        self.doc_fdes.reserve(corpus.len() * self.fde_dim);
        for doc in corpus {
            let fde = self.encoder.encode(&doc.tokens, FdeSide::Document);
            self.doc_fdes.extend_from_slice(&fde);
        }

        // If the corpus is smaller than n_lists, clamp — we'll just get
        // singleton lists.
        let n_lists = self.ivf.n_lists.min(self.docs.len().max(1));
        let dim = self.fde_dim;
        let mut rng = ChaCha8Rng::seed_from_u64(self.ivf.seed);

        // Init: pick n_lists distinct docs as initial centroids.
        let mut idxs: Vec<usize> = (0..self.docs.len()).collect();
        for i in 0..n_lists.min(self.docs.len()) {
            let j = i + (rng.gen::<usize>() % (self.docs.len() - i));
            idxs.swap(i, j);
        }
        let mut centroids = vec![0.0f32; n_lists * dim];
        for (c, &src) in idxs.iter().take(n_lists).enumerate() {
            centroids[c * dim..(c + 1) * dim].copy_from_slice(self.slice_of(src));
        }

        // Lloyd's iterations under Euclidean distance on the FDE.
        let mut assign = vec![0u32; self.docs.len()];
        for _ in 0..self.ivf.kmeans_iters {
            // Assign.
            for (di, _) in self.docs.iter().enumerate() {
                let v = self.slice_of(di);
                let mut best_c = 0u32;
                let mut best_d = f32::INFINITY;
                for c in 0..n_lists {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let mut acc = 0.0f32;
                    for j in 0..dim {
                        let diff = v[j] - cent[j];
                        acc += diff * diff;
                    }
                    if acc < best_d {
                        best_d = acc;
                        best_c = c as u32;
                    }
                }
                assign[di] = best_c;
            }
            // Update: mean of assigned FDEs.
            let mut sums = vec![0.0f32; n_lists * dim];
            let mut counts = vec![0u32; n_lists];
            for (di, &c) in assign.iter().enumerate() {
                let v = self.slice_of(di);
                let dst = &mut sums[c as usize * dim..(c as usize + 1) * dim];
                for j in 0..dim {
                    dst[j] += v[j];
                }
                counts[c as usize] += 1;
            }
            for c in 0..n_lists {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    let dst = &mut sums[c * dim..(c + 1) * dim];
                    for x in dst.iter_mut() {
                        *x *= inv;
                    }
                    centroids[c * dim..(c + 1) * dim].copy_from_slice(dst);
                }
                // Empty centroids keep their previous value.
            }
        }

        // Build posting lists.
        self.centroids = centroids;
        self.posting = vec![Vec::new(); n_lists];
        for (di, &c) in assign.iter().enumerate() {
            self.posting[c as usize].push(di as u32);
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalHit> {
        if self.docs.is_empty() {
            return Vec::new();
        }
        let dim = self.fde_dim;
        let fq = self.encoder.encode(query, FdeSide::Query);

        // Rank centroids by inner product with fq (proxy for their
        // posting list's expected MaxSim).
        let mut cent_scores: Vec<(usize, f32)> = self
            .centroids
            .chunks_exact(dim)
            .enumerate()
            .map(|(c, cent)| (c, dot(&fq, cent)))
            .collect();
        cent_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let probes = cent_scores.iter().take(self.ivf.n_probe.max(1));

        // Scan probed posting lists.
        let mut candidates: Vec<RetrievalHit> = Vec::new();
        for &(c, _) in probes {
            for &di in &self.posting[c] {
                let s = dot(&fq, self.slice_of(di as usize));
                candidates.push(RetrievalHit { id: self.docs[di as usize].id, score: s });
            }
        }
        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.ivf.candidates.max(k));

        // Optional exact-MaxSim rerank.
        if self.ivf.rerank > 0 {
            let take = self.ivf.rerank.min(candidates.len());
            // Build id → doc lookup for the rerank pool. Small linear
            // scan is fine — `take` is small.
            let ids_of_docs: std::collections::HashMap<u32, usize> = self
                .docs
                .iter()
                .enumerate()
                .map(|(i, d)| (d.id, i))
                .collect();
            let mut reranked: Vec<RetrievalHit> = candidates
                .iter()
                .take(take)
                .map(|hit| {
                    let &di = ids_of_docs.get(&hit.id).expect("doc id");
                    let s = maxsim(query, &self.docs[di].tokens, self.d_token);
                    RetrievalHit { id: hit.id, score: s }
                })
                .collect();
            reranked.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            reranked.truncate(k);
            return reranked;
        }

        candidates.truncate(k);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::FlatMaxSim;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    fn corpus(n: usize, tok: usize, d: usize, seed: u64) -> Vec<Document> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|id| {
                let mut tokens = vec![0.0f32; tok * d];
                for x in tokens.iter_mut() {
                    let u1: f32 = rng.gen::<f32>().max(1e-9);
                    let u2: f32 = rng.gen::<f32>();
                    *x = (-2.0f32 * u1.ln()).sqrt()
                        * (2.0f32 * std::f32::consts::PI * u2).cos();
                }
                crate::chamfer::l2_normalize_set(&mut tokens, d);
                Document { id: id as u32, tokens, n_tokens: tok }
            })
            .collect()
    }

    #[test]
    fn ivf_with_rerank_matches_flat_on_self_query() {
        let d = 8;
        let c = corpus(50, 5, d, 33);
        let fde = FdeParams { d, k_sim: 4, reps: 8, seed: 7 };
        let ivf = IvfParams { n_lists: 8, n_probe: 4, candidates: 30, rerank: 10, kmeans_iters: 6, seed: 1 };
        let mut idx = MuveraIvf::new(fde, ivf);
        idx.build(&c);
        // Any doc used as query should recover itself after rerank
        // (rerank uses exact MaxSim, which is maximised by the doc's
        // own tokens for cosine-space vectors).
        for id in [3usize, 17, 41] {
            let hits = idx.search(&c[id].tokens, 5);
            assert!(
                hits.iter().any(|h| h.id as usize == id),
                "id {id} missing from ivf+rerank top-5: {hits:?}"
            );
        }
    }

    #[test]
    fn ivf_without_rerank_still_ranks_matches_first_most_of_the_time() {
        // With rerank off, the pipeline is pure FDE — approximation
        // error dominates. We only insist on "usually" (>= 30/50).
        let d = 8;
        let c = corpus(50, 5, d, 34);
        let fde = FdeParams { d, k_sim: 5, reps: 12, seed: 8 };
        let ivf = IvfParams { n_lists: 8, n_probe: 8, candidates: 20, rerank: 0, kmeans_iters: 6, seed: 2 };
        let mut idx = MuveraIvf::new(fde, ivf);
        idx.build(&c);
        let mut hits_ok = 0;
        for id in 0..c.len() {
            let h = idx.search(&c[id].tokens, 5);
            if h.iter().any(|x| x.id as usize == id) {
                hits_ok += 1;
            }
        }
        assert!(hits_ok >= 30, "only {hits_ok}/50 recovered under IVF-no-rerank");
    }

    #[test]
    fn ivf_agrees_with_flat_maxsim_top1_after_rerank() {
        let d = 8;
        let c = corpus(40, 5, d, 99);
        let fde = FdeParams { d, k_sim: 5, reps: 12, seed: 5 };
        let ivf = IvfParams { n_lists: 8, n_probe: 8, candidates: 40, rerank: 40, kmeans_iters: 6, seed: 4 };
        let mut idx = MuveraIvf::new(fde, ivf);
        idx.build(&c);
        let mut oracle = FlatMaxSim::new(d);
        oracle.build(&c);
        for id in [1usize, 12, 27] {
            let a = idx.search(&c[id].tokens, 1);
            let b = oracle.search(&c[id].tokens, 1);
            assert_eq!(a[0].id, b[0].id, "top1 mismatch at id {id}");
        }
    }
}
