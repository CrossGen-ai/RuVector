# ADR-359: LeanVec-Style Low-Rank Pre-Projection Shortlist

**Status:** Proposed (nightly research, 2026-11-11)
**Context branch:** `research/nightly/2026-11-11-leanvec-shortlist`
**Research doc:** `docs/research/nightly/2026-11-11-leanvec-shortlist/README.md`

## Context

Brute-force L2² over a flat float32 dataset is the correctness oracle inside ruvector, but it costs `n·d` multiplies per query. On 20k × 128 that is already 1.6 ms per query on Apple Silicon, well above the budget for mid-size RAG indexes that want sub-millisecond latency.

Prior nightlies in this tree explored a wide set of post-training compression and probing schemes (RaBitQ, SQ cascades, PFOR postings, FastScan LUT16, Hadamard SQ, LVQ, among others). None of them attack the input dimensionality itself. LeanVec (Intel Labs, 2024) shows that a learned linear projection into a rank-d' subspace — where d' ≈ d/4 to d/8 — preserves rank-order well enough to drive a shortlist, and that this combines cleanly with downstream quantization.

## Decision

Introduce a new nightly crate, `ruvector-leanvec-shortlist`, that implements a dependency-free Rust reference of the LeanVec cascade:

1. Train `P ∈ R^{d'×d}` deterministically via subspace power iteration (PCA) on a random subsample.
2. Materialize `X' = X P^T`.
3. At query time, shortlist in `R^{d'}`, rerank in `R^d`.

The crate exposes three interchangeable `Searcher` implementations (brute force, PCA-16, PCA-32) plus a random-Gaussian JL projection for ablation.

It lives as a standalone crate (its own `[workspace]` table) outside the root workspace. If the design moves out of nightly, ADR-360+ will cover the production split described in the research doc's "Production crate layout" section.

## Consequences

**Positive**

- Measured **7.68× speedup over brute force at 91.7% recall@10** and **4.82× at 96.6% recall@10** on the 20k × 128 synthetic clustered benchmark included with the crate.
- Pure `std` Rust, `#![forbid(unsafe_code)]`, no external deps, under 500 lines per file. Trivially auditable.
- Composes with every existing ruvector index: a LeanVec projection in front of IVF, HNSW, or a bitvector shortlist is strictly additive — the shortlisted IDs just get handed to the next stage.
- The random-JL ablation baked into the benchmark lets us quantify when learned projection matters vs when it does not — important for datasets that are nearly isotropic.

**Negative / risks**

- Training cost scales as `sample_n · d · d'`, unavoidable but tunable via `sample_n`.
- Pure PCA is a weak baseline for out-of-distribution queries. The production version will need the LeanVec-OOD objective.
- Projection matrix is a per-index artifact that must be persisted and versioned alongside the dataset.

**Neutral**

- Workspace is unmodified; root `cargo build --workspace` is unaffected.

## Alternatives considered

- **OPQ rotation only.** Covered by `gist-opq-rotation-2026`. Keeps dimensionality the same; targets quantization quality, not shortlist cost.
- **Random Gaussian JL projection.** Implemented here as an ablation. Matches PCA-16 on this cluster-structured dataset but is expected to lag on real OOD data.
- **Full OPQ + PQ cascade.** More complex, requires a product codebook, larger binary artifact.
- **Low-rank sketch via Hadamard transform.** Covered by `gist-hadamard-sq-2026`; shares spirit but has no learned component.

The LeanVec shortlist is the smallest, most composable intervention that targets input dimensionality directly, so it is the right nightly addition.
