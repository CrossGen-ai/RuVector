---
adr: 194
title: "RoarGraph — Projected Bipartite Graph Index for Cross-Modal / OOD ANN"
status: accepted
date: 2026-05-23
authors: [ruvnet, claude-nightly]
related: [ADR-143, ADR-178, ADR-193]
tags: [ann, vector-search, graph-index, cross-modal, ood, nightly-research]
---

# ADR-194 — RoarGraph: ruvector's first OOD-aware graph index

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-23-roargraph-cross-modal-ann` as the new crate
`crates/ruvector-roargraph`. `cargo build --release -p ruvector-roargraph`
and `cargo test --release -p ruvector-roargraph` are green. Benchmark
binary `ruvector-roargraph-bench` reports real numbers (see
`docs/research/nightly/2026-05-23-roargraph-cross-modal-ann/README.md`).

## Context

Every existing graph index in ruvector (HNSW in `ruvector-core`, DiskANN in
`ruvector-diskann`, hyperbolic-HNSW, MUVERA, ACORN-filtered HNSW, …) builds
its adjacency from **base–base distances**. This is an implicit assumption:
*the query distribution looks like the base distribution.*

That assumption silently breaks for:

| Workload                            | Why base–base graphs underperform                              |
|-------------------------------------|----------------------------------------------------------------|
| Cross-modal retrieval (CLIP-T → CLIP-I) | text and image embeddings occupy slightly different regions |
| Retrieval-augmented LLMs (query → passage) | question embeddings ≠ document embeddings              |
| Recommender systems (user → item)   | user and item vectors are co-trained but distinct manifolds   |
| Hybrid sparse/dense (SPLADE → dense)| different sparsity statistics                                 |

Published evidence: [1] reports up to **40-point recall@10 gaps** between
in-distribution and out-of-distribution queries on HNSW for real cross-modal
benchmarks (T2I-ANN, LAION-CLIP). Our own synthetic test in
`ruvector-roargraph-bench` reproduces an 8-point gap on a clean mean-shift.

ruvector ships into RAG and cross-modal production workloads. The lack of an
OOD-aware index is a real recall-quality gap, not an academic curiosity.

## Decision

Add `crates/ruvector-roargraph`, a new workspace member implementing the
**Neighborhood-Aware Projection (NAP)** graph from Chen et al. [1], with:

* A static-index build path:
  `RoarGraph::build(base: Vec<Vec<f32>>, train_queries: &[Vec<f32>], cfg)`.
* α-RobustPrune (DiskANN-style) on the projected adjacency.
* Reverse-link augmentation + reachability repair via beam search.
* Standard best-first beam search at query time:
  `search(q, k, ef) -> Vec<(u32, f32)>`.
* Two honest baselines in the same crate for apples-to-apples comparison:
  `KnnGraph` (in-base kNN + beam) and `RandomGraph` (degree-M random + beam).
* A `ruvector-roargraph-bench` binary producing the markdown numbers
  consumed by the research doc and the public gist.

The crate is dependency-light (`rand`, `rand_chacha`, `rayon`) and has no
unsafe code. Vectors are owned `Vec<f32>`; squared-L2 only in this nightly.

## Why now / why this design

* **Trait-based swap-in.** RoarGraph deliberately mirrors the `search(q, k,
  ef)` shape of `RoarGraph` / `KnnGraph` / `RandomGraph` so a future
  `ruvector_core::Index` trait can host all three with no API churn.
* **Honest baselines in the same crate.** Including the random-graph floor
  inside the bench prevents the common "compare RoarGraph to nothing and
  call it a 90 % win" trap; we report the absolute recall, not just the lift.
* **Synthetic OOD now, real OOD later.** Gaussian mean-shift isolates the
  OOD signal from confounders (anisotropy, label noise). Real-dataset
  benchmarks are explicitly deferred to a follow-up nightly so this ADR can
  land with provably-reproducible numbers today.

## Numbers (from `cargo run --release -p ruvector-roargraph --bin ruvector-roargraph-bench`)

n=4 000, d=64, train=2 000, 500 ID + 500 OOD queries, k=10, ef=64, degree=24.

| variant        | build (ms) | edges  | bytes (idx+vecs) | QPS    | rec@10 ID | rec@10 OOD |
|----------------|-----------:|-------:|-----------------:|-------:|----------:|-----------:|
| random-graph   |        2.1 |  96000 |          1.41 MB | 10,400 |     0.354 |      0.346 |
| knn-graph      |       95.1 |  79241 |          1.34 MB | 13,595 |     0.907 |      0.829 |
| **roargraph**  |    **858.7** | **85455** |       **1.37 MB** | **12,729** | **0.898** | **0.903** |

Acceptance criteria (all PASS):

1. `roar.recall_ood ≥ knn.recall_ood` — 0.903 vs 0.829.
2. `roar.recall_ood >  random.recall_ood` — 0.903 vs 0.346.
3. `roar.recall_ood ≥ 0.80`.

## Consequences

### Positive

* ruvector becomes (to our knowledge) the **first OSS vector engine with a
  built-in OOD-aware graph index in Rust**.
* Cross-modal RAG and recommender workloads gain a recall-quality lever that
  Milvus / Qdrant / Weaviate / Pinecone do not currently expose.
* The crate is a clean reference for future OOD-aware variants (τ-MNG,
  OOD-DiskANN, etc.).

### Negative / cost

1. **Build is slower.** ~9× the knn-graph build at `|Q|/|B| = 0.5` in the
   PoC because of brute-force training-query kNN. Acceptable for a static
   PoC; addressed by Roadmap item 2 (IVF/RaBitQ pre-filter).
2. **Memory parity, not memory win.** Index footprint is within ~2 % of the
   kNN graph at identical degree — no reduction, just better recall.
3. **No incremental insert/delete.** Static-only for this iteration.
4. **Training-set lifecycle.** Adopters must version the training queries
   alongside the encoder version; encoder drift silently degrades recall.

### Out of scope (this ADR)

* Incremental insert/delete → future ADR.
* Disk-resident layout → future ADR (compose with `ruvector-diskann`).
* Filtered RoarGraph (predicate masks) → future ADR (compose with
  `ruvector-acorn`).
* Quantised vectors → future ADR (compose with `ruvector-rabitq` /
  `ruvector-lvq`).
* WASM bindings → trivial wrapper, deferred until the static API stabilises.

## Alternatives considered

| Option                              | Why rejected for *this* ADR                                |
|-------------------------------------|-------------------------------------------------------------|
| Patch HNSW with OOD-sample reweighting (τ-MNG style) | Larger change to `ruvector-core`; would conflate two ADRs |
| Build only the NAP, skip RobustPrune | Recall holds but adjacency blows up to O(|Q|·nap_k²)        |
| Use sampled OOD queries to *retrain* an encoder | Out of scope; ruvector does not own the encoders |
| Re-export from existing C++ RoarGraph reference via FFI | Violates the Rust-only constraint of nightly research |
| In-base kNN graph (Vamana init) and call it a day | This is the *baseline*, not the improvement; bench shows it lags by 7.4 OOD recall points |

## References

1. Chen, M., Zhang, K., He, Z., Jin, Y., Mao, Q., Zhu, Y., Wang, B., & Chen, J.
   *RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal
   Approximate Nearest Neighbor Search.* PVLDB 17(11), 2024.
2. Jaiswal, S., et al. *OOD-DiskANN.* arXiv:2211.12850, 2024.
3. Subramanya, S. J., et al. *DiskANN.* NeurIPS 2019.
4. Malkov, Y. A., & Yashunin, D. A. *HNSW.* IEEE TPAMI 2018.
