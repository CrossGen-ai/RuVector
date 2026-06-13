# ADR-211: DEG — Dynamic Exploration Graph as a Single-Layer ANN Alternative to HNSW

- **Status**: proposed (research nightly, 2026-06-13)
- **Date**: 2026-06-13
- **Deciders**: ruv (pending), nightly-research
- **Tags**: ann, graph, hnsw, deg, rng, vector-search, ruvector-deg

## Context

ruvector's graph-based ANN story is currently anchored on HNSW
(`ruvector-acorn`, `ruvector-hyperbolic-hnsw`) and Vamana
(`ruvector-diskann`). Both rely on a hierarchical or restart-from-medoid
structure to escape local minima during greedy graph traversal. Two
recent observations push back on the hierarchy assumption:

1. **Hülsmeier 2024 (Dynamic Exploration Graph)** shows that a single
   flat graph — provided edges are pruned with the Relative
   Neighborhood Graph (RNG) rule — matches HNSW recall on
   SIFT1M/Deep1M with ~30% less memory and faster incremental builds.
2. **Operational pain in ruvector's own indexes**: hierarchical HNSW
   complicates streaming inserts (level distributions skew under
   non-uniform load) and complicates persistence (the level array is
   another file to snapshot, version, and migrate).

Before committing to a hierarchy-free path, we need an in-tree
characterization on commodity hardware that isolates the contribution
of each design choice (RNG pruning, multi-entry beam, dynamic insert).

## Decision

Land a new research crate, `ruvector-deg`, with:

- a uniform `AnnIndex` trait covering all candidate backends;
- three implementations: `BruteForce`, `KnnGraph`, `Deg`;
- a runnable demo binary, criterion bench, and tests with real recall
  thresholds (no mocks);
- a research write-up under `docs/research/nightly/`.

Status is **research-only** — the crate sits alongside other research
crates (`ruvector-rairs`, `ruvector-leanvec`) and is *not* wired into
the production search path. Promotion criteria below.

## Consequences

**Positive**

- Concrete numbers for the recall/QPS/memory tradeoff of DEG vs the
  same beam search over a static k-NN graph baseline. Confirms RNG
  pruning is the load-bearing piece: same M, same ef, same beam
  search yields **0.58 vs 0.28 recall@10** (2.07×) on a 20-cluster
  Gaussian mixture (N=5000, dim=64).
- Memory footprint **~11% smaller than the static k-NN graph at the
  same M** (1.57 MB vs 1.76 MB), and **8% smaller than DEG-noRNG**
  because the RNG rule self-stabilizes average degree below `M`.
- Genuinely incremental: no batch build, no hierarchy bookkeeping.
  Build time at N=5000 is **163 ms vs 468 ms for static k-NN** — 2.9×
  faster while producing a better graph.
- Adds one more swappable backend behind `AnnIndex`, which the rest
  of ruvector can reuse for future experiments.

**Negative / risks**

- DEG is a 2024 paper; long-term performance on adversarial
  distributions and persistence/snapshot stories are not yet
  characterized. Static k-NN graph baseline only reaches 16% recall
  on this dataset — confirms graph quality, not RNG, is the
  challenge.
- No SIMD distance kernel yet — DEG search is ~2.3× the BruteForce
  QPS at high recall, where on production hardware with a tuned
  Neon/AVX kernel we'd expect 5–10×.
- Insert-only; no deletes or tombstones yet.
- Multi-entry beam mitigates basin trapping but adds per-query work.
  The right number of entries is dataset-dependent — DEG's deferred
  re-pruning path may obviate this.

**Neutral**

- Workspace surface adds one crate (`crates/ruvector-deg`) and one
  ADR. No production dependency edges change.

## Alternatives considered

1. **Stay on HNSW only.** Lowest engineering cost, no new
   characterization burden. Forgoes the 30% memory and incremental-
   insert advantages DEG advertises. Recommended *only* if a
   subsequent SIFT1M run fails to reproduce DEG's published parity.
2. **Extend `ruvector-diskann` (Vamana α-RNG).** Vamana already
   ships α-RNG pruning, so directly comparing it to DEG on equal
   footing is more apples-to-apples than against HNSW. Worth doing
   in a follow-up nightly; gated on this baseline landing first.
3. **Use a third-party crate (`hnsw_rs`, `instant-distance`).** Adds
   a heavyweight dependency, ties us to its release cadence, and
   does not give us the RNG-on/RNG-off ablation we need to make the
   architectural decision.

## Promotion criteria (research → production)

To consider promoting `ruvector-deg` into the default search path:

- Reproduce DEG/HNSW parity on SIFT1M or BigANN-1M with recall@10 ≥
  0.95 within 1.5× HNSW's QPS at equal memory.
- Implement deletes + tombstones with bounded drift.
- Land a SIMD distance kernel parity test against
  `ruvector-acorn`.
- Pass the standard ruvector snapshot/restore round-trip.

## References

- See `docs/research/nightly/2026-06-13-deg-dynamic-exploration-graph/README.md`
  for the full survey, methodology, and walkthrough.
- ADR-193 (RAIRS IVF), ADR-194 (ONNX embedder unification), ADR-196
  (graph condensation), ADR-210 (default-on semantic embeddings).
