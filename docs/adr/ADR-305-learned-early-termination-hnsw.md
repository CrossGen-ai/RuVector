# ADR-305: Learned Early Termination for HNSW-style Beam Search

## Status

Proposed. Experimental crate (`ruvector-learned-termination`); not wired into
the default query path of any production index.

## Context

Every graph-ANN implementation in the RuVector workspace (HNSW-style beam
search inside `ruvector-core`, `ruvector-diskann`, `ruvector-coherence-hnsw`,
`ruvector-hnsw-repair`, `ruvector-hnsw-rng-diverse`, and the sister PoCs
under `ruvector-adaptive-ann`, `ruvector-speculative-ann`, and
`ruvector-entropy-ann`) shares a single tuning knob: **`ef_search`**. That
knob is a fixed integer, set once at query time, that upper-bounds beam
width and — indirectly — search termination.

`ef_search` is a compromise. Easy queries (query near a cluster centroid,
top-k already found by step 5) do not benefit from expanding to `ef=80`;
they spend 60+ additional distance calls confirming what they already know.
Hard queries (boundary between clusters, poor entry point) are truncated
too early. Prior in-workspace attempts to make this adaptive:

- `ruvector-adaptive-ann` (ADR-207) — scalar distance-threshold rule.
  Works, requires per-dataset threshold sweep.
- `ruvector-entropy-ann` (ADR-303) — Shannon entropy of the candidate-heap
  distance distribution. Documented negative result: entropy saturates on
  real graph traversals.
- `ruvector-speculative-ann` (ADR-298) — parallel speculative expansion.
  Speedup on multi-core; ignores per-query difficulty.

None of these give a **per-query, per-step, learned** termination signal
with a footprint compatible with hot-loop inference.

External SOTA (Milvus AUTOINDEX, Qdrant `hnsw_ef=auto`, Pinecone
Serverless dynamic partition budget) is either batch-level or coarse-
grained. Ada-ef (arXiv:2512.06636) and EDEN (arXiv:2605.09745) address
adaptivity but require offline threshold calibration or introduce a
signal (entropy) that this workspace has already measured as unreliable.

## Decision

Add a new experimental crate `ruvector-learned-termination` implementing
a five-feature logistic-regression classifier that gates beam-search
termination once per expansion step. The model is:

- **Tiny**: 6 f32 weights (bias + 5 features) = 24 bytes.
- **Cheap**: one dot product + sigmoid per step, ~20 ns.
- **Backend-agnostic**: features come from the beam-search state, not
  the graph.
- **Trained offline**: SGD, 40 epochs, ~10 ms on 5 k samples collected
  from ~60 held-out queries. Retraining is cheap enough to run in the
  background from live traffic.

Runtime features:

1. `best_dist` — closest distance in the results heap.
2. `improve_rate` — moving-average `best_dist` decrease (window 4).
3. `gap_kth` — normalised gap between (k-1)-th and k-th results.
4. `steps_norm` — `steps_so_far / ef`.
5. `frontier_ratio` — fraction of the just-expanded node's neighbours
   that were unvisited.

Termination rule: stop when `P(top-k will still change) < tau`, gated
by a `min_steps` floor.

## Consequences

**Positive:**

- Measured `1.20×` beam-distance-call speedup at `-0.003` recall (tau=0.15)
  and `1.31×` at `-0.011` recall (tau=0.30) on the benchmark corpus. The
  Oracle upper bound on this dataset is only `1.09×` because standard HNSW
  pruning already handles trivial cases — the Learned rule exceeds Oracle
  because it terminates on *predicted* futility rather than *observed*
  stability, exposing a cost-vs-recall knob the Oracle cannot.
- 24-byte model footprint. No new runtime dependencies (pure Rust,
  `[dependencies]` empty).
- Backend-agnostic: adapters for `ruvector-core` HNSW,
  `ruvector-diskann`, and the coherence graph will each be < 50 LOC.
- Retraining is a background job, not an offline SLO.

**Negative:**

- Distribution shift requires retraining. Mitigation: 10 ms retrain
  makes this trivially schedulable.
- Wall-clock savings are dataset-scale-dependent. At n=2000 the beam is
  short enough that per-step logistic inference offsets some of the
  distance-call savings; at n≥10 000 dist calls dominate and savings
  translate linearly.
- Adds a per-query tuning knob (`tau`) on top of `ef`. Operator burden.
- The learned model can be adversarially probed. Worst case: fallback to
  Fixed-ef budget (soft failure, no correctness bug).

**Neutral / follow-ups:**

- Per-cluster predictors (ADR follow-up) — cheap because each is 24 bytes.
- Two- vs five-feature ablation — trained weights show `best_dist`
  contributes essentially nothing (weight 0.64 vs sigma-normalised
  peers at 9.7), so a three-feature model is likely equivalent.
- Wire into `ruvector-core` HNSW behind a `learned_termination` feature
  flag once the SIFT-1M / GIST-1M benchmark confirms the wall-clock
  story at scale.

## Alternatives considered

- **Neural gate** (2-layer MLP, ~256 params). Rejected: 100× the
  footprint, no observed accuracy delta over logistic on this feature
  set.
- **Entropy gate** (per `ruvector-entropy-ann`). Rejected: measured
  negative result — entropy saturates.
- **Bandit over discrete tau values** at inference time. Complementary,
  not competing; see "What to improve next" in the research README.
- **Server-side `ef` auto-tuning** (Milvus/Qdrant style). Batch-level
  granularity; misses the per-query bimodal cost structure.
- **Do nothing** — accept Fixed-ef as the industry standard. Rejected
  given the demonstrated 17 % beam-dist saving at negligible recall
  cost.

## Related

- ADR-207 — Adaptive-ef distance-threshold rule.
- ADR-298 — Speculative ANN expansion.
- ADR-303 — Entropy-adaptive ANN (negative result baseline).
- Research README: `docs/research/nightly/2026-08-16-learned-early-termination-hnsw/README.md`.
- Crate: `crates/ruvector-learned-termination/`.
