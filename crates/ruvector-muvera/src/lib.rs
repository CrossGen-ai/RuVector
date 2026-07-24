//! # ruvector-muvera
//!
//! MUVERA — **Mu**lti-**Ve**ctor **R**etrieval via fixed-dimensional **A**pproximation.
//!
//! Implements the FDE (Fixed-Dimensional Encoding) construction from
//! Dhulipala, Hadian, Jayaram, Lee, Mirrokni, "MUVERA: Multi-Vector Retrieval
//! via Fixed Dimensional Encodings", NeurIPS 2024.
//!
//! Given a document represented as a bag of `n_D` token vectors and a query
//! represented as a bag of `n_Q` token vectors, both in `R^d`, define the
//! Chamfer / MaxSim similarity
//!
//! ```text
//! MaxSim(Q, D) = Σ_{q ∈ Q}  max_{v ∈ D}  ⟨q, v⟩
//! ```
//!
//! Late-interaction retrievers (ColBERT, ColPali, ColBERTv2) score with
//! MaxSim, which is far more expressive than single-vector cosine but
//! prevents the use of ordinary single-vector ANN indexes (HNSW, IVF-PQ,
//! DiskANN, etc.). MUVERA fixes this by constructing per-vector-set
//! encodings `Φ_Q(Q), Φ_D(D) ∈ R^{d · B · R}` such that
//!
//! ```text
//! ⟨Φ_Q(Q), Φ_D(D)⟩  ≈  MaxSim(Q, D)
//! ```
//!
//! **The FDE construction** (SimHash LSH partition, `R` independent
//! repetitions):
//!
//! 1. Draw `R` independent SimHash partitions. Each partition draws
//!    `k_sim` random hyperplanes `h_1..h_{k_sim} ∈ R^d`. A token `v`
//!    is assigned bucket `b(v) = Σ_j 2^{j-1} · 1{⟨h_j, v⟩ > 0}` in
//!    `{0, .., B-1}` where `B = 2^{k_sim}`.
//! 2. For a **query** token set `Q`, define, for each repetition `r`
//!    and each bucket `b`,
//!    ```text
//!    Φ_Q^{(r,b)} = Σ_{q ∈ Q, b_r(q) = b}  q
//!    ```
//!    i.e. the *sum* of query tokens that fell in bucket `b`.
//! 3. For a **document** token set `D`, define
//!    ```text
//!    Φ_D^{(r,b)} = (1/|D_{r,b}|) · Σ_{v ∈ D, b_r(v) = b}  v      if  |D_{r,b}| ≥ 1
//!                = (fill via nearest non-empty bucket, chosen by Hamming)   otherwise
//!    ```
//!    i.e. the *centroid* of document tokens in that bucket, with empty
//!    buckets filled by the nearest non-empty bucket in Hamming distance
//!    over the SimHash code (paper §3.2, "fill-empty" variant).
//! 4. Concatenate and normalize by `1/R`:
//!    ```text
//!    Φ(X) = (1/R) · concat_{r=1..R} concat_{b=0..B-1} Φ^{(r,b)}(X)
//!    ```
//!    with dimensionality `d · B · R`.
//!
//! Because each query token contributes to exactly one bucket per
//! repetition, and MUVERA's document construction guarantees that bucket
//! carries the mean of document tokens whose SimHash code equals (or is
//! nearest to) the query token's code, the inner product `⟨Φ_Q, Φ_D⟩`
//! is an unbiased estimator of MaxSim in the limit `R → ∞`. See paper
//! Theorem 3.1 for the concentration bound.
//!
//! **What ruvector-muvera does with it.** The FDE is a *single* vector
//! per document. That means downstream retrieval reduces to plain
//! single-vector max-inner-product search — you can bolt this onto any
//! HNSW / IVF-PQ / DiskANN implementation that already exists in the
//! RuVector fleet. This crate ships three swappable retrievers behind
//! the [`MultiVectorRetriever`] trait:
//!
//! * [`FlatMaxSim`] — the exact oracle. Brute-force MaxSim, O(|Q|·|D|·d)
//!   per candidate document. This is the recall ceiling.
//! * [`MuveraFlat`] — MUVERA FDE + brute-force inner product over the
//!   FDE vectors. This isolates the *approximation error* of the FDE
//!   itself from any index approximation.
//! * [`MuveraIvf`] — MUVERA FDE + a tiny IVF (k-means partition of the
//!   FDE space) rerank pipeline: retrieve top-`n_probe·candidates_per`
//!   FDE candidates, then rerank the top-`rerank` with exact MaxSim.
//!   This shows the *end-to-end* two-stage pipeline that a production
//!   system would deploy.
//!
//! All three implement the same trait so a caller can A/B them under
//! identical corpora / queries — the pattern used elsewhere in ruvector
//! (see `crates/ruvector-maxsim`, `crates/ruvector-rabitq`).
//!
//! Deterministic: all randomness flows through a `u64` seed, so
//! benchmarks and tests are reproducible bit-for-bit.

#![deny(missing_debug_implementations)]

pub mod chamfer;
pub mod fde;
pub mod ivf;
pub mod retriever;

pub use chamfer::{maxsim, l2_normalize_inplace, l2_normalize_set};
pub use fde::{FdeParams, FdeEncoder};
pub use ivf::{IvfParams, MuveraIvf};
pub use retriever::{
    Document, MultiVectorRetriever, RetrievalHit, FlatMaxSim, MuveraFlat,
};
