# ADR-194: Matryoshka Adaptive Retrieval (MAR)

- **Status:** Proposed (PoC merged on `research/nightly/2026-05-30-matryoshka-adaptive-retrieval`)
- **Date:** 2026-05-30
- **Authors:** ruvector nightly research agent
- **Related:** ADR-193 (RaIRS-IVF), ADR on RaBitQ (1-bit quantization)

## Context

Modern embedding models — OpenAI `text-embedding-3-*`, Nomic Embed
v1.5, Snowflake Arctic-Embed-L 2.0, Mixedbread mxbai-embed-large-v1,
Cohere `embed-v3` — are increasingly trained with **Matryoshka
Representation Learning** (MRL). The defining property is that every
*prefix* of the vector is itself a usable, L2-meaningful embedding.

ruvector ships brute-force, HNSW, IVF, RaBitQ, LeanVec, OPQ, RoarGraph
and several graph index variants, but has no first-class story for
prefix-truncated retrieval. Users who already use an MRL model today
are paying the full-dimension cost on every search, even when they
could trade 1–5% recall for 5–25× QPS.

The HuggingFace "Matryoshka Embedding Models" blog popularised the
two-stage *adaptive retrieval* recipe: search a short prefix, then
re-rank the top candidates at the full dimension. No published Rust
implementation existed in the ruvector ecosystem.

## Decision

Add a new workspace member `crates/ruvector-matryoshka` exposing:

- a `Retriever` trait,
- `BruteForceFull` (baseline),
- `BruteForceLow` (prefix-only baseline),
- `MatryoshkaAdaptive { low_dim, rerank_factor }` (the new path),
- a `mar-demo` binary that emits real recall/QPS/latency numbers.

The crate is dependency-light (`rand`, `rand_distr`, `thiserror`,
optional `rayon`) and follows the workspace `[workspace.dependencies]`
versions. No new transitive dependencies enter the workspace.

The trait-based shape lets future work compose MAR with HNSW or RaBitQ
on the coarse pass without API churn.

## Consequences

### Positive

- Users with MRL embeddings get 4× QPS at 0.96 recall@10 (measured on
  20k × 768 corpus, Apple M4 Max) with +33% RAM. The high-speed
  operating point reaches 25× QPS at 0.51 recall@10 for cases where
  approximate recall is acceptable.
- Crate is small (~440 lines including tests), easy to review, easy to
  graduate to a production layout (see research doc).
- Trait + multiple impls give us a clean place to add SIMD, HNSW
  prefix, RaBitQ prefix, and filter integration without breaking
  callers.
- Real benchmark numbers are captured in `BENCHMARK.md` style at the
  research doc; the binary is the source of truth.

### Negative

- Doubles resident memory (prefix + full). For very large corpora this
  matters; mitigated by RaBitQ-on-prefix (future work, see research
  doc roadmap item 3).
- Recall is **model-dependent**. Embeddings not trained with MRL will
  not benefit; the PoC's synthetic distribution simulates a well-trained
  MRL model with a deliberately mild signal-decay exponent (α=0.02).
- One more crate to maintain. Justified by the SOTA gap and the small
  surface area.

### Neutral

- Adds one workspace member; build time impact on `cargo build
  --workspace` is negligible (<1s on M4 Max release build).

## Alternatives considered

1. **Add MAR as a method on `ruvector-core`.** Rejected: core has a
   wider blast radius for refactors and a stricter API stability bar.
2. **Implement MAR as a feature flag on `ruvector-leanvec`.** Rejected:
   LeanVec is a *learned* projection; MAR works on already-trained MRL
   embeddings. Conflating them would confuse the public API.
3. **Skip the dedicated crate and document the pattern in a README.**
   Rejected: the SOTA value is in a concrete `Retriever` trait that
   composes with the rest of ruvector; docs alone do not give us that.

## Acceptance criteria

The PoC binary asserts and prints on every run:

- best MAR config recall@10 ≥ 0.95 — **0.9556 ✓**
- best MAR config beats prefix-only recall — **0.9556 vs 0.1788 ✓**
- best MAR config beats brute-full QPS — **600.4 vs 148.5 ✓**

All 5 integration tests pass: `cargo test --release -p ruvector-matryoshka`.
