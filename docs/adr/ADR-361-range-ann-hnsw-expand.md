# ADR-361: Range (epsilon-radius) ANN with IVF-ball-prune and HNSW-expand

- **Status**: Proposed (nightly research — implementation shipped as
  `crates/ruvector-range-ann`, promotion to `crates/ruvector-range`
  pending)
- **Date**: 2026-11-24
- **Related**: [ADR-359](./ADR-359-e-means-shared-clustering.md), prior
  IVF/HNSW ADRs

## Context

Every ANN surface in ruvector today targets top-k retrieval: fixed `k`,
heap-bounded early termination, and a result set of exactly `k` points.
A growing fraction of real workloads instead need **range** search —
return every point within a radius `r` of the query — with no fixed
upper bound on the result-set size. Concrete triggers:

- Near-duplicate detection and clustering for ingestion pipelines.
- Compliance / policy checks ("no document closer than `r` to this
  embedding of a protected template").
- Geofence-style retrieval over latitude/longitude embeddings.
- RAG triggers that only fire when a semantic match crosses a
  pre-committed similarity threshold.
- Verification receipts (ADR-352 family) that need "every neighbor within
  an audit radius" rather than "the `k` closest".

Top-k code paths cannot answer these queries. Running a top-k search at
some large `k` and post-filtering by `r` is wasteful when the ball is
sparse, unsafe when the ball is dense (truncation), and wrong when the
caller needs the *complete* in-ball set. Range search is a different
retrieval mode and deserves a dedicated code path.

## Decision

Introduce a `RangeIndex` trait and three implementations:

1. `linear::LinearScan` — exact baseline, used as ground truth in tests.
2. `ivf::IvfRange` — exact IVF with a per-posting-list *circumscribed
   radius* annotation and a two-ball intersection prune.
3. `hnsw_expand::HnswExpand` — approximate best-first graph search
   bounded by an `alpha`-widened expand ball and a warmup phase.

All three return `(u32, f32)` where the `f32` is *squared* L2. The trait
is intentionally small; no streaming / cursor API in this ADR — callers
materialize the full in-ball set.

The prototype ships in `crates/ruvector-range-ann`. Promotion to the
stable `ruvector-range` crate is gated on:

- The anisotropic Mahalanobis bound, SIMD inner loops, SOAR spill, and
  incremental updates listed under "What to improve next" in the
  research doc.
- A test matrix covering uniform data, anisotropic clusters, and
  adversarial queries at the ball edge.

## Consequences

**Positive**

- Range search becomes a first-class retrieval mode without re-using the
  top-k path with large `k`.
- IVF-range-ball-prune gives **11.11x** speedup over linear scan on
  cluster-structured data *while remaining exact*. That is strictly
  better than the common "probe top-m centroids" heuristic on both
  axes: more aggressive pruning and no recall loss.
- The trait is backend-agnostic so downstream crates (RAG triggers,
  verification receipts) depend on `RangeIndex`, not on a specific
  implementation.

**Negative**

- Build time for IVF-range is dominated by k-means — ~24s for 50k
  points at 48 dims. We will need a streaming / mini-batch variant
  before we can retrofit this onto live indexes.
- The HNSW-expand backend is not competitive with IVF-range on
  cluster-structured data; it is retained because it is the only
  backend that works on unclustered data. The ADR commits to
  benchmarking both on a uniform-data corpus before stabilization.
- `max_expand` and `alpha` are two knobs with coupled effects. Exposing
  both to callers is error-prone; a `recall_target` wrapper that picks
  them is pre-committed future work.

**Operational**

- Added to workspace members alongside `crates/ruvector-entropy-ann`.
- `cargo build --release -p ruvector-range-ann` succeeds on stock
  macOS/Linux; no feature flags required.
- Tests are deterministic (seeded ChaCha20) and run in <1s in release.

## Alternatives considered

- **Reuse top-k with post-filter.** Rejected: truncation when the ball
  is dense, wasteful probing when it is sparse, and no guarantee of
  completeness.
- **Milvus-style search-iterator.** Expands top-k rings until no new
  hits appear within `r`. Works but has pathological failure at the
  ball edge; ball-prune on circumscribed radius is cleaner.
- **Cover-tree range queries.** Exact and well-understood but
  construction cost is prohibitive at ruvector's target scale.
- **DiskANN range (Pan et al., PVLDB 2023).** Closest relative of
  `hnsw-expand`. We did not adopt DiskANN directly because the
  on-disk I/O accounting is scope-mismatched for an in-memory core
  crate; the frontier logic is nonetheless borrowed.
