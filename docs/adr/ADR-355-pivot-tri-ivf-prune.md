# ADR-355: Pivot Triangle-Inequality Pruning for IVF Posting Scans

- Status: Proposed (nightly research, 2026-10-20)
- Related: ADR-353 (PQ-ADC early abandon), ADR-354 (ADSampling random projection)
- Crate: `crates/ruvector-pivot-tri-ivf/`

## Context

IVF (inverted-file) indexes dominate the posting-list scan phase of recall-
constrained vector search. During a scan over a probed cluster we compute an
exact L2 distance for every candidate even when the candidate is obviously
far from the query - the heap already knows a tighter bound but has no cheap
test to apply it.

Partial-L2 early abandon (ADR variants) addresses this with coordinate-wise
accumulation, but still touches every dimension of every candidate until the
abandon threshold triggers. ADSampling (ADR-354) uses randomized JL
projections. Both are query-driven. Neither exploits structure of the
cluster itself.

## Decision

Introduce a classical metric-space pruning primitive, applied inside each
IVF cluster:

1. **Build time** (per cluster): pick `m` pivots by farthest-point sampling
   over the cluster's own posting entries. Precompute `d(x, p_j)` for every
   vector `x` in the posting and every pivot `p_j`. Store these as a dense
   `[n_cluster][m]` float table.
2. **Query time** (per probed cluster): compute `d(q, p_j)` once per pivot
   (`m * nprobe` extra distances total). For each candidate `x`, derive the
   lower bound `LB(q, x) = max_j |d(q, p_j) - d(x, p_j)|` from the triangle
   inequality. If `LB^2 >= heap_top_sq`, skip the full L2.

The pruning is **exact**: the triangle inequality is lossless, so recall
remains identical to the baseline IVF scan.

## Consequences

**Positive**
- Zero recall impact. The same top-k ids are produced in the same order.
- Measurable speedup on cluster-structured data (46% candidates skipped,
  22% faster at m=4 on 64-cluster mixture; see research doc for numbers).
- Memory overhead is modest and predictable: `m * n * 4 B` plus pivot
  vectors themselves (`m * dim * 4 B` per cluster).
- Composes with any existing per-vector encoding (float, SQ, PQ, RaBitQ).
  The LB uses pre-stored scalar distances, not the vector representation.

**Negative**
- On iid high-dim data the LB is weak (concentration of measure makes all
  `d(x, p)` nearly equal). Pruning falls to ~5% and overhead can cost
  latency. The index MUST expose a toggle so operators can disable pivots
  on known worst-case distributions.
- Build time grows by `m * n_cluster` extra L2s per cluster. In practice
  this adds a one-off multi-second cost at index build on 20K/128-dim.
- The LB is tightest when pivots are far apart; farthest-point sampling
  within a cluster is good enough, but a cross-cluster pivot pool is
  future work.

## Alternatives considered

- **Pivot distances for IVF centroids only** — standard inverted multi-index
  trick. Prunes clusters, not candidates. Orthogonal to this proposal.
- **FINGER sign-bit LB** — tighter per-vector LB but requires a residual
  code and only applies to HNSW. We want an IVF-native primitive.
- **ADSampling** — random projection JL bound (ADR-354). Different
  mechanism; could stack on top of pivot pruning as a second-stage filter.

## Rollout

Nightly PoC crate only. Promotion to a feature-gated module inside
`ruvector-core` requires: (a) multi-cluster shared-pivot variant, (b)
int8-quantized pivot distance table, (c) SIMD pivot L2 kernel, (d)
benchmarks on SIFT1M / GIST1M with real embeddings.
