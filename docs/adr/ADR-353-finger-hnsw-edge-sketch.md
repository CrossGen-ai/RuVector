# ADR-353: FINGER-style HNSW edge-vector sketches for cheap distance approximation

**Status:** Proposed. 2026-10-24. PoC implemented and benchmarked at
`crates/ruvector-finger-hnsw` on branch `research/nightly/2026-10-24-finger-hnsw-edge-sketch`.

## Context

Best-first search on HNSW spends ≥ 80% of its wall-clock on exact L2
evaluations: one per neighbor visited, each O(d). As ruvector pushes toward
higher-dimensional embeddings (ADR-352 Python SDK targets 768- and 1024-dim
embedders) and larger graphs (edge-tenancy work in ADR-351), this cost
dominates search latency.

FINGER (ICML 2023, Yan et al.) shows that at each visited node `c`, the
distance to every graph neighbor can be rewritten as

```text
d(q, n_i)^2 = ||q - c||^2 + ||n_i - c||^2 - 2 * <q - c, n_i - c>
```

so the only neighbor-dependent term is a per-query–per-edge inner product.
If we precompute a low-rank Johnson-Lindenstrauss sketch of every edge
vector, we approximate that inner product in O(k) instead of O(d), and use
the approximation as a lower-bound filter to skip exact L2 on obvious
non-contenders.

ruvector has adjacent work: `ruvector-adsampling-random-projection` sketches
*data vectors* for early L2 abandonment; `ruvector-hnsw-neighbor-prefetch`
reduces per-eval cost via cache prefetch; `ruvector-pq-fastscan-lut16`
replaces exact L2 with codebook lookup. None of them target the same lever
(per-edge sketch of residuals). FINGER is complementary to all three.

To our knowledge no permissive Rust-native implementation of FINGER exists;
the published code is a Faiss patch.

## Decision

Adopt a FINGER-style edge-vector sketch as a swappable search backend
beside exact L2 inside ruvector's HNSW. The PoC lands as a standalone
crate `ruvector-finger-hnsw` to isolate the mechanism; production
integration is tracked under "Consequences" below.

Specifically:

1. Introduce a `Search` trait with `ExactSearch`, `FingerSearch`, and
   `FingerGuarded` implementations.
2. Store, per directed graph edge `(c, n_i)`:
   - `edge_proj[i]`: `(P * (n_i - c)) * (1/√k)` where `P` is a shared
     `k × d` Gaussian projection matrix.
   - `edge_sq_len[i]`: `||n_i - c||^2`.
3. At query time, at each visited node, project `q - c` into the same
   basis once and estimate each neighbor's squared distance in O(k).
4. Use the sketch strictly as a lower-bound filter (`FingerGuarded`) in
   production: fall through to exact L2 whenever the approximation
   suggests the neighbor might make the top-k. Pure sketch mode
   (`FingerSearch`) is kept as a research harness only.

Default tunables (chosen from measured recall-vs-savings curve):

- Sketch rank `k = 32` for `d` ∈ [128, 512].
- Guard slack = `0.15` (allow 15% headroom above the kth-best threshold
  before trusting the lower bound to filter).

## Consequences

### Positive

- 50–54% reduction in exact L2 evaluations per query at ≤ 0.5 pp recall
  loss (measured, see research doc).
- Fully training-free: no learned parameters, just one shared Gaussian
  projection matrix. Deterministic given a seed.
- Trait-based swap: a caller can pick exact, pure sketch, or guarded at
  runtime. Falling back to exact is one line.
- Mechanism is graph-structure-agnostic: works on flat kNN, HNSW, NSG,
  Vamana. No change to graph build or layering.
- Stacks additively with partial-L2 early-abandonment and PQ-ADC
  cascades: they prune different parts of the cost (fewer evaluations
  vs cheaper per-evaluation).

### Negative

- Memory overhead: `4 * (k*d + n*m*k + n*m)` bytes. For n=1M, m=32,
  k=16 this is ~2 GB. Non-trivial. Mitigation: tiered sketches on hot
  graph levels only; int8-quantized sketches as a future feature.
- Wall-clock parity requires a vectorized JL projection kernel. This PoC
  ships scalar f32; wall-clock crosses over exact only at d ≥ 1024. The
  algorithmic win (distance-call count) is immediate; the wall-clock win
  needs SIMD, which is a known follow-up, not a design limitation.
- Recall is sensitive to the `slack` tunable at small k and large d. The
  shipping default of 0.15 holds in our regime; downstream users with
  tighter recall budgets should probe their own slack/k.

### Rollout

- **Milestone A (now):** PoC in `ruvector-finger-hnsw`, standalone, flat
  kNN graph, scalar f32 kernel. Passes `cargo test`. Benchmark binary
  produces real numbers. This ADR.
- **Milestone B:** SIMD-vectorized JL projection behind `simd` feature,
  using `std::simd::f32x8` for portability. Expected 3–4× speedup on the
  sketch path.
- **Milestone C:** Integrate as a feature-flagged backend inside
  `ruvector-core`'s HNSW search path, with the exact backend as the
  default. Add runtime config to pick backend per index.
- **Milestone D:** Int8-quantized sketch storage; measure recall at
  halved sketch RAM.
- **Milestone E:** Per-node sketch bases (full FINGER method) — learned
  low-rank basis from node's edge-vector covariance instead of a shared
  global projection. Expected recall improvement at the same k.

## Alternatives considered

1. **ADSampling-style data-vector random projection** (SIGMOD 2023). Also
   JL-based but prunes exact L2 dimension-by-dimension on the *data*
   vector. Complementary, not substitute: ADSampling reduces per-eval
   cost; FINGER reduces number of evals. Could ship both.
2. **PQ-ADC + FastScan.** Replaces exact L2 entirely with a lookup table.
   Different compression regime: lossy per-vector codebook, no per-edge
   sketch. Already covered by `ruvector-pq-fastscan-lut16`.
3. **Pure cache prefetch** (`ruvector-hnsw-neighbor-prefetch`). Reduces
   per-eval wall-clock without touching the number of evals. Stacks
   additively with FINGER; neither replaces the other.
4. **Do nothing.** Exact HNSW is correct and currently fast enough on
   128-dim embeddings. Rejected because the Python SDK roadmap pushes
   to 768-dim+ embedders, where exact L2 is the hot path by a wide
   margin.

## Open questions

- Is a shared global projection matrix (this PoC) or per-node learned
  basis (full FINGER) the right default for ruvector's workloads? The
  per-node variant is more complex to build but should give measurably
  better recall at the same k. Deferred to Milestone E.
- Should sketches be stored only on HNSW's upper levels (where visits
  are rare but each pruning saves more exact work), or on level 0 too
  (where most visits land)? Deferred to Milestone C profiling.
