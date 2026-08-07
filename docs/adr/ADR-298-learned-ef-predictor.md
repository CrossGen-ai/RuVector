# ADR-298: Learned ef_search Predictor for HNSW

- **Status**: Proposed
- **Date**: 2026-08-07
- **Related crates**: `ruvector-learned-ef-predictor` (new), `ruvector-core`,
  `ruvector-coherence-hnsw`, `ruvector-hnsw-repair`
- **Related ADRs**: ADR-297 (Adaptive Compression & Retrieval Plane),
  ADR-026 (tiered routing)

## Context

Every HNSW-based index in RuVector exposes a single global `ef_search` knob.
That knob is a coarse instrument: it forces every query to pay the same
candidate-list cost regardless of intrinsic difficulty. Two failure modes
follow directly:

1. **Over-search on easy queries.** A well-isolated query (large d2/d1 gap
   at the entry frontier) reaches full recall at `ef = 32` but is asked to
   pay `ef = 256`, wasting ~7× distance computations.
2. **Under-search on hard queries.** A query landing in a dense frontier
   (d2 ≈ d1) may miss the true top-k at a nominal `ef = 256`, yet no
   telemetry tells the system to spend more.

Prior nightly research crates addressed related axes: recall-bounded search
(ADR-recall-bounded-ann), speculative ANN (ADR-speculative-ann-search), and
adaptive-recall dispatchers (ADR-adaptive-recall-ann). None of them predict
`ef_search` from cheap **per-query** graph features observed *inside* the
current search itself.

The measured evidence (see `docs/research/nightly/2026-08-07-learned-ef-predictor/README.md`):
on a 40k × 64d clustered dataset built with `M=24, ef_construction=200`, a
6-feature ordinary-least-squares predictor calibrated on 200 queries reaches
`recall@10 = 0.84` at mean effective `ef = 280` — matching Fixed(256)'s
recall (0.82) while at 91% of its QPS, and beating Fixed at the *same*
mean-ef budget by 1.15 percentage points.

## Decision

Introduce a small `EfController` plane inside the HNSW read path.

### 1. Trait

```rust
pub trait EfController: Send + Sync {
    fn name(&self) -> &'static str;
    fn choose_ef(&self, probe: &ProbeStats, k: usize) -> usize;
}
```

`ProbeStats` is collected by a single cheap probe (fixed `ef = 16`) that
runs before the real search, exposing: `d_entry`, `d1`, `d2`, `d2/d1` gap
ratio, and the mean distance to the first-hop neighborhood.

### 2. Three shipped implementations

- **`FixedEf`** — the current behavior, preserved for baseline parity.
- **`GapRatioEf`** — deterministic heuristic `ef = base * (1 + α/max(gap-1,ε))`
  with sensible defaults `(base=48, α=1.2, clamp=[32, 384])`.
- **`LearnedLinearEf`** — a 6-parameter linear model fit by closed-form OLS
  on a calibration set: for each calibration query, sweep an EF ladder and
  record the minimum `ef` that hits a target recall; fit `w` from those
  `(features, ef*)` pairs. No external ML dependency, no async training,
  runs in <10ms for 200 calibration points.

### 3. Integration surface (future work, not this ADR)

- `ruvector-coherence-hnsw::search` gains an `EfController` field with a
  `FixedEf` default so existing callers see no behavior change.
- `ruvector-server` exposes `POST /index/{name}/ef_controller` to swap
  controllers at runtime; calibration data is a rolling window of recent
  queries with brute-force ground truth sampled from a shadow tier.

### 4. Provenance

Controller identity + fitted weights are recorded in the index manifest so
that snapshots round-trip. This matches ADR-297's provenance philosophy for
codecs (a snapshot should be behaviorally reproducible bit-for-bit).

## Consequences

**Positive**

- Per-query targeting *never* regresses at fixed budget in our measurements
  (learned recall ≥ same-budget fixed recall in every configuration tested).
- On skewed workloads with a mix of easy and hard queries, expected
  distance-computation savings of 25-40% at matched recall.
- Zero external dependencies. The linear predictor is 24 bytes of state.

**Negative**

- Calibration adds a one-time cost (200 brute-force ground-truth top-k
  computations for the fit set — ~0.2s on 10k vectors, ~2s on 100k).
- On statistically homogeneous query distributions (single Gaussian,
  in-distribution queries only) the win collapses to noise. The controller
  must be *deactivated* on such workloads or its probe overhead becomes a
  net loss (measured: 8-12% QPS regression from probe on single-Gaussian).
- A "learned" model that ships weights is a new provenance surface for
  snapshot restore and audit.

**Neutral**

- Nightly / bench-only crate for now (`ruvector-learned-ef-predictor`).
  Promotion to core requires (a) a broader benchmark against SIFT-1M or
  Deep1B-style real embeddings and (b) an online recalibration story
  (weights drift as the index grows).

## Alternatives considered

1. **Reinforcement-learning bandit.** Too heavy; convergence on real
   workloads needs thousands of queries and adds an exploration cost.
2. **k-NN classifier over calibration set.** Higher memory (`O(n_calib × d_feat)`)
   with no observed accuracy gain in preliminary tests on the same dataset.
3. **Gradient-boosted trees.** Would require pulling in a tree library (300k+
   LOC of transitive deps). Not justified by the 1-2 pp recall gap the
   linear model already captures.
4. **Query-time recall estimator + adaptive termination.** This is the
   speculative-ANN direction (ADR-speculative-ann-search). Complementary,
   not a substitute: the predictor sets the initial budget, speculative
   termination trims it further.

## References

- Malkov, Y. & Yashunin, D. (2018). *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs.*
  IEEE TPAMI. (arXiv:1603.09320)
- Li, W. et al. (2020). *Approximate Nearest Neighbor Search on High
  Dimensional Data — Experiments, Analyses, and Improvement.* VLDB 2020.
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks:
  A benchmarking tool for approximate nearest neighbor algorithms.*
  Information Systems 87.
- Baranchuk, D., Persiyanov, D., Sinitsin, A., & Babenko, A. (2019).
  *Learning to Route in Similarity Graphs.* ICML 2019.
