# ADR-193: SOAR — Spillover-Optimized Anisotropic Residuals for IVF

- **Status**: Proposed (PoC landed in `crates/ruvector-soar`)
- **Date**: 2026-05-11
- **Author**: Nightly research agent
- **Related**: ADR-188 (sparse-attention stamp scheme — unrelated; named for
  ADR numbering); prior nightly: `docs/research/nightly/2026-04-26-acorn-filtered-hnsw/`

## Context

ruvector's existing IVF behaviour (delivered via `ruvector-cluster` and
embedded inside `ruvector-rabitq`'s coarse quantizer) follows the textbook
single-assignment rule: each database vector is placed in exactly one cell.
Recall scales linearly with `nprobe/n_lists`, which forces practical
deployments to either run with large nprobe (linear QPS hit) or accept a
recall ceiling around 90% — neither acceptable for retrieval over
catalog-scale embedding sets.

The competitive gap is concrete: Google's ScaNN (and by extension Vertex
Vector Search) has shipped SOAR since 2023; FAISS, Milvus, Qdrant, and
Pinecone have not. ruvector closing this gap is a clear differentiation
opportunity in the open-source vector database market — particularly
against Milvus and Weaviate which target the same Rust/cloud-native niche.

## Decision

Adopt SOAR (NeurIPS 2023) as a first-class IVF assignment strategy in
ruvector, delivered as a standalone crate `ruvector-soar` that exposes a
swappable `Assignment` enum:

- `Naive` — legacy single-assignment IVF, default for backwards compat.
- `IsotropicSpillover` — λ=0, picks the second-nearest centroid.
- `SoarAnisotropic { lambda: f32 }` — full SOAR; λ=1.0 default.

The crate is workspace member-only initially. After real-data validation
(see ADR consequences), the `IvfIndex` will implement
`ruvector_core::IndexBackend` and `ruvector-cluster` will be re-pointed at
it.

## Consequences

### Positive

- **+44% relative recall lift at nprobe=1** on a synthetic Gaussian
  mixture; published numbers suggest 1.5–2× on real ANN benchmarks (SIFT,
  GLOVE). Unblocks low-latency, low-nprobe production deployments where
  every probed cell costs disk I/O (DiskANN-style tiered storage).
- **Modest storage cost**: each point in 2 posting lists. Index overhead
  goes from 0.093 MB to 0.170 MB on the n=20K PoC (vector payload
  dominates at 4.88 MB). 2× posting size at PoC scale, well below 2×
  total because raw vectors are not duplicated.
- **No `unsafe`** in the crate. Pure Rust, autovectorized.
- **Trait-based design** lets `ruvector-rabitq`, `ruvector-diskann`, and
  future IVF callers swap in SOAR without changing the search path.
- **Deterministic**: seeded k-means + deterministic assignment → identical
  builds across machines for the same seed. Crucial for repro and CI.

### Negative

- **Build time ~3× slower** than naive (45 ms vs 15 ms on the PoC) because
  each point requires two centroid scans plus a residual computation.
  Mitigation: parallelize point loop with rayon (deferred).
- **QPS at fixed nprobe drops ~40%** because candidate sets are larger
  after spillover. Net QPS at fixed *recall* is roughly neutral, but
  users currently tuning nprobe should re-tune.
- **PoC currently keeps raw vectors in-index** for rerank simplicity. A
  production cut must layer this on top of PQ/RaBitQ codes — straightforward
  but not yet done.
- **Streaming inserts not supported**: SOAR's secondary assignment depends
  on centroid positions; concurrent updates need a re-assignment pass
  (similar to DiskANN's two-pass rebuild). Out of scope for this PoC.

### Neutral

- **Anisotropic vs isotropic delta is small on synthetic data** (<0.005
  recall). The orthogonal-residual term shines on naturally anisotropic
  data (text embeddings, image features); the paper's published 1.5–2×
  lifts come from SIFT-1M and GLOVE-1M. Real-data validation is the
  immediate next step.

## Alternatives considered

1. **Increase nprobe** (do nothing). Rejected: linear QPS cost, doesn't
   address the boundary failure mode.
2. **Inverted Multi-Index (IMI)** — Cartesian-product codebooks.
   Rejected: K² cell explosion, quantizer training is brittle, and IMI
   sits orthogonal to SOAR rather than competing with it (the two compose).
3. **Multi-probe LSH**. Rejected: LSH-specific, doesn't apply to IVF;
   recall ceilings are worse than IVF on dense embeddings.
4. **r-NN multi-assignment with r=3 or higher**. Rejected for the initial
   PoC: paper reports diminishing returns past r=2; storage cost grows
   linearly. Worth revisiting after real-data validation.
5. **ScaNN's anisotropic *quantization*** (ICML 2020). Already partially
   present in `ruvector-rabitq`; SOAR is a complementary index-side
   technique, not a substitute.

## Acceptance criteria

- [x] `cargo build --release -p ruvector-soar` succeeds.
- [x] `cargo test -p ruvector-soar` passes (6 tests, including a
  recall-vs-naive sanity check).
- [x] `cargo run --release -p ruvector-soar --example soar-bench`
  produces a recall@10 / QPS table for 4 variants × 6 nprobe values.
- [x] SOAR recall at nprobe=1 strictly exceeds naive IVF recall at
  nprobe=1 by ≥10 absolute percentage points on the synthetic mixture
  (observed: 46.8% → 67.5%).
- [ ] (Next branch) SOAR recall on SIFT-1M matches the published
  ScaNN+SOAR numbers within ±2%.

## Open questions

- Should the SOAR assignment phase run during k-means convergence (joint
  optimization) rather than as a one-shot post-pass? The paper does
  one-shot; some follow-ups propose joint training. Deferred.
- Optimal λ for embedding models: BERT/CLIP/etc. produce embeddings with
  different anisotropy. A per-corpus λ-search may yield further gains.

## References

- Sun et al., **"SOAR: Improved Indexing for Approximate Nearest Neighbor
  Search."** NeurIPS 2023. arXiv:2404.00774.
- `crates/ruvector-soar/src/index.rs` — implementation.
- `docs/research/nightly/2026-05-11-soar-ivf-spillover/README.md` — full
  research write-up with benchmark methodology.
