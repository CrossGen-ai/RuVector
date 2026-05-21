---
adr: 194
title: "Anisotropic Product Quantization (ScaNN-style) — weighted-PQ codebooks for MIPS"
status: accepted
date: 2026-05-21
authors: [crossgen-ai, claude-nightly]
related: [ADR-193]
tags: [pq, quantization, mips, scann, anisotropic, vector-search, nightly-research]
---

# ADR-194 — Anisotropic Product Quantization

> **Provenance note.** The anisotropic loss is taken from Guo, Sun, Lindgren,
> Geng, Simcha, Chern, Kumar, "Accelerating Large-Scale Inference with
> Anisotropic Vector Quantization," ICML 2020 (arXiv:1908.10396) — the paper
> that introduces ScaNN. The OPQ baseline is from Ge, He, Ke, Sun, "Optimized
> Product Quantization," CVPR 2013 / TPAMI 2014. Plain PQ is Jégou, Douze,
> Schmid, "Product Quantization for Nearest Neighbor Search," TPAMI 2011.
> This crate is an honest *implementation* of those ideas — APQ in particular
> uses an iterative weighted-Lloyd approximation rather than ScaNN's exact
> closed-form per-subspace weights. Judge it by the reproducible recall
> numbers below, not by the citation.

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-21-anisotropic-pq` as
`crates/ruvector-anisotropic-pq`. Workspace build green; 4 integration tests
pass; demo binary `anisotropic-pq-demo` produces the recall numbers in this
ADR.

## Context

ruvector ships RaBitQ (1-bit rotation-based quantization), LVQ, LeanVec, and
several SOAR/RAIRS-style IVF variants — but no Product Quantization. PQ is
the lingua franca of production vector databases (FAISS IVF-PQ, Qdrant
IVF-PQ, Milvus, OpenSearch). For dense-retrieval embeddings (BERT,
SBERT, OpenAI ada-002, Cohere, Voyage) the *useful* notion of similarity is
maximum inner product (MIPS) on unit-normalized vectors, not symmetric L2.

Plain PQ trains its sub-codebooks by minimizing spherical L2 reconstruction
error: `E[||x − c||²]`. That objective treats every direction of residual
the same. But for MIPS the residual component **parallel to x** is exactly
what biases inner-product estimates, while the orthogonal component
contributes much less to ranking error. ScaNN (ICML 2020) showed that
re-weighting the loss to penalize parallel error 4–60× more than orthogonal
error lifts MIPS recall meaningfully at the same code budget.

We want a swappable Rust implementation that lets us:

* compare PQ, OPQ, and anisotropic-PQ on the same data with identical kernels;
* measure real compression and recall — no aspirational benchmarks;
* lay the groundwork for an IVF-PQ index that uses ScaNN-style codebooks.

## Decision

Add a new crate `ruvector-anisotropic-pq` with a single `Quantizer` trait
and three backends:

| Backend          | Training objective                                  | Code size |
|------------------|-----------------------------------------------------|-----------|
| `Pq`             | per-subspace Lloyd k-means, spherical L2            | `m` bytes |
| `Opq`            | learned orthogonal R + per-subspace L2 in rotated space | `m` bytes |
| `AnisotropicPq`  | weighted Lloyd k-means with per-point anisotropic weight | `m` bytes |

All three encode to `m` u8 codes per database vector (256 centroids per
subspace) and all share the same asymmetric-scoring kernel shape, so we can
A/B them with the same recall harness.

### What's actually different in APQ

1. Initialize with plain PQ.
2. For `refinement_sweeps` rounds:
   * decode current codes → residual `r_i = x_i − decode(encode(x_i))`;
   * compute `w_i = h_orth + (h_par − h_orth) * cos²(r_i, x_i)`, where
     `cos²(r,x) = (r·x)² / (||r||²·||x||²)` and `h_par/h_orth = eta`;
   * rerun weighted Lloyd in every subspace with those `w_i`.

This is **not** ScaNN's exact derivation (which decomposes the loss per
subspace using conditional norms). It is a faithful, simple
approximation — and we test it strictly against plain PQ in the same crate.

### What's NOT in scope

* SIMD AVX-512 / NEON kernels for the scan loop (planned: `ruvector-bench`
  integration).
* IVF coarse-quantizer wrapping (planned: combine with `ruvector-rairs`).
* GPU codebook training.
* Asymmetric `LUT4` packed-code layout (4-bit codes / SIMD shuffle).

## Reproducible numbers

Hardware: macOS 15.6 / darwin 24.6.0, Apple Silicon, single thread for the
demo. `cargo run --release -p ruvector-anisotropic-pq --bin
anisotropic-pq-demo`:

```
dataset:   n_train=4000, n_db=8000, n_queries=200, d=64
quantizer: m=8, k=256, code_bytes=8, eta=4
data:      unit-normalized anisotropic Gaussian, 16 clusters

train time:
  PQ      246 ms
  OPQ    1114 ms   (3 outer sweeps + 4 inner k-means passes each)
  APQ    1043 ms   (3 refinement sweeps)

encode 8 000 vectors:
  PQ      29 ms
  OPQ     41 ms   (extra cost: matmul d×d per vector)
  APQ     28 ms

recall@10 (IP ground truth, 200 queries):
  PQ      0.1875
  OPQ     0.1840
  APQ     0.1975   (+ 1.0 pp over PQ, + 1.4 pp over OPQ, same code size)

compression vs raw f32: 32×
```

APQ outperforms both baselines at identical code budget. OPQ's rotation
helps for true Gaussian product data; on the elongated-cluster mixture
used here OPQ slightly underperforms PQ because rotation can mix
anisotropy across subspaces — exactly the case APQ's per-point weighting
was designed to handle.

## Consequences

### Good

* First-class PQ family in ruvector, behind a uniform trait the rest of
  the codebase can program against.
* `AnisotropicPq` gives a measurable recall lift over PQ at the same
  bytes-per-vector — a tangible win for dense-retrieval workloads.
* Test suite asserts the lift, so future refactors won't silently
  regress it.

### Bad

* APQ's training is ~4× slower than PQ (multiple weighted-Lloyd passes).
  For now this is amortized over very large databases; we should add a
  "skip refinement when sweep doesn't improve recall on a holdout" early
  stop.
* OPQ Procrustes uses Jacobi eigendecomposition — fine up to a few
  hundred dims, but a real BLAS-backed SVD would be needed for d > 512.
* APQ weighting is the iterative approximation, not ScaNN's closed-form.
  Closing that gap is the obvious next-iteration follow-up.

### Neutral

* No new external dependencies (just `rand`, `rand_distr`, `thiserror`,
  and `rayon` on native). criterion is a dev-only addition.

## Alternatives considered

1. **Stay PQ-free, push everyone to RaBitQ.** Rejected: RaBitQ is 1-bit
   and great for cheap rerank, but full-precision PQ at 8–32 bytes/vec
   gives much higher recall at the same memory budget for top-k retrieval.
2. **Use a third-party PQ crate (faiss-rs, ndarray-pq).** Rejected: ties
   us to FFI for the wrong reason, breaks no_std-ish targets,
   and we want the codebooks to coexist with `ruvector-rabitq` rotation
   matrices and `ruvector-rairs` IVF lists in a uniform way.
3. **Implement only ScaNN's exact loss with closed-form weights.**
   Rejected for this iteration: the closed-form is per-subspace and
   requires careful norm bookkeeping; we wanted a measurable v0 first.
   Tracked as next-iter work.

## What to improve next (next-iter roadmap)

* **Closed-form anisotropic weights** (one more iteration).
* **SIMD `LUT4` 4-bit code scan** — `m/2` bytes per vector with `vpshufb`
  / `tbl` lookup; ~5–10× scan speedup.
* **IVF wrapper**: combine with `ruvector-rairs` so we get IVF-APQ with
  redundant assignment and SEIL residual scoring.
* **GPU training**: codebook k-means on Hailo / Metal for >100k training
  points.
* **Streaming insertion**: live re-quantization without full retrain
  (track centroid drift, retrain triggered by KL divergence threshold).
