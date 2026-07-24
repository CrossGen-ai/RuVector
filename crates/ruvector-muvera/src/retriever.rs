//! Swappable multi-vector retrievers.
//!
//! All variants implement [`MultiVectorRetriever`], which is the single
//! seam a caller uses to benchmark or A/B them under identical corpora.

use crate::chamfer::maxsim;
use crate::fde::{dot, FdeEncoder, FdeParams, FdeSide};

/// A document: an id + its multi-vector token bag stored as row-major
/// `[n × d]`.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: u32,
    pub tokens: Vec<f32>,
    pub n_tokens: usize,
}

/// A retrieval result: doc id + score. Higher is better.
#[derive(Debug, Clone, Copy)]
pub struct RetrievalHit {
    pub id: u32,
    pub score: f32,
}

/// Swappable retriever contract. `build` consumes a corpus; `search`
/// answers top-`k` given a multi-vector query.
pub trait MultiVectorRetriever {
    fn name(&self) -> &'static str;
    fn build(&mut self, corpus: &[Document]);
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalHit>;
}

// --------------------------------------------------------------------
// Variant 1 — FlatMaxSim (exact oracle)
// --------------------------------------------------------------------

/// Exact brute-force MaxSim over every document token bag. This is the
/// **recall ceiling**: any hit ordering it returns is by definition
/// ground truth for MaxSim.
#[derive(Debug, Default)]
pub struct FlatMaxSim {
    d: usize,
    docs: Vec<Document>,
}

impl FlatMaxSim {
    pub fn new(d: usize) -> Self {
        Self { d, docs: Vec::new() }
    }
}

impl MultiVectorRetriever for FlatMaxSim {
    fn name(&self) -> &'static str { "flat_maxsim" }
    fn build(&mut self, corpus: &[Document]) {
        self.docs = corpus.to_vec();
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalHit> {
        let mut hits: Vec<RetrievalHit> = self
            .docs
            .iter()
            .map(|d| RetrievalHit {
                id: d.id,
                score: maxsim(query, &d.tokens, self.d),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }
}

// --------------------------------------------------------------------
// Variant 2 — MuveraFlat (FDE + brute-force inner product)
// --------------------------------------------------------------------

/// MUVERA FDE + brute-force inner product over the FDE vectors.
///
/// This isolates the FDE approximation error: retrieval order is a
/// pure function of `⟨Φ_Q, Φ_D⟩`, no index approximation on top.
#[derive(Debug)]
pub struct MuveraFlat {
    encoder: FdeEncoder,
    doc_ids: Vec<u32>,
    doc_fdes: Vec<f32>, // [n_docs × fde_dim]
    fde_dim: usize,
}

impl MuveraFlat {
    pub fn new(params: FdeParams) -> Self {
        let encoder = FdeEncoder::new(params);
        let fde_dim = encoder.fde_dim();
        Self { encoder, doc_ids: Vec::new(), doc_fdes: Vec::new(), fde_dim }
    }
    pub fn fde_dim(&self) -> usize { self.fde_dim }
    pub fn encoder(&self) -> &FdeEncoder { &self.encoder }
}

impl MultiVectorRetriever for MuveraFlat {
    fn name(&self) -> &'static str { "muvera_fde_flat" }
    fn build(&mut self, corpus: &[Document]) {
        self.doc_ids.clear();
        self.doc_fdes.clear();
        self.doc_fdes.reserve(corpus.len() * self.fde_dim);
        for doc in corpus {
            let fde = self.encoder.encode(&doc.tokens, FdeSide::Document);
            debug_assert_eq!(fde.len(), self.fde_dim);
            self.doc_ids.push(doc.id);
            self.doc_fdes.extend_from_slice(&fde);
        }
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalHit> {
        let fq = self.encoder.encode(query, FdeSide::Query);
        let n = self.doc_ids.len();
        let mut hits: Vec<RetrievalHit> = (0..n)
            .map(|i| RetrievalHit {
                id: self.doc_ids[i],
                score: dot(&fq, &self.doc_fdes[i * self.fde_dim..(i + 1) * self.fde_dim]),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn make_corpus(n_docs: usize, n_tok: usize, d: usize, seed: u64) -> Vec<Document> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n_docs)
            .map(|id| {
                let mut tokens = vec![0.0f32; n_tok * d];
                for x in tokens.iter_mut() {
                    let u1: f32 = rng.gen::<f32>().max(1e-9);
                    let u2: f32 = rng.gen::<f32>();
                    *x = (-2.0f32 * u1.ln()).sqrt()
                        * (2.0f32 * std::f32::consts::PI * u2).cos();
                }
                crate::chamfer::l2_normalize_set(&mut tokens, d);
                Document { id: id as u32, tokens, n_tokens: n_tok }
            })
            .collect()
    }

    #[test]
    fn flat_maxsim_returns_query_itself_as_top1() {
        let d = 8;
        let corpus = make_corpus(20, 4, d, 11);
        let mut idx = FlatMaxSim::new(d);
        idx.build(&corpus);
        let query = corpus[7].tokens.clone();
        let hits = idx.search(&query, 3);
        assert_eq!(hits[0].id, 7);
    }

    #[test]
    fn muvera_flat_beats_random_on_self_query() {
        // The document that IS the query should almost always be in the
        // top-3 under MuveraFlat when reps is generous.
        let d = 8;
        let corpus = make_corpus(64, 6, d, 21);
        let params = FdeParams { d, k_sim: 5, reps: 12, seed: 900 };
        let mut idx = MuveraFlat::new(params);
        idx.build(&corpus);
        let query = corpus[42].tokens.clone();
        let hits = idx.search(&query, 3);
        let found = hits.iter().any(|h| h.id == 42);
        assert!(found, "expected doc 42 in top-3, got {hits:?}");
    }
}
