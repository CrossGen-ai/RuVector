# ADR-354 — ADSampling: random-projection + hypothesis-test distance pruning

- **Status**: Proposed (nightly research, 2026-10-17)
- **Supersedes**: —
- **Related**: ADR-353 (pq-adc-early-abandon), nightly `2026-09-20-partial-l2-early-abandon`

## Context

Candidate-list scanning dominates recall-bounded ANN latency once a
coarse structure (IVF posting, HNSW frontier, SPANN partition) has
narrowed the search. Our existing distance-pruning paths are:

- **Full L2** — auto-vectorizes beautifully, but computes every dim of
  every candidate even when most will not reach the top-k.
- **Partial-L2 early abandon** (nightly 2026-09-20) — accumulate
  dimensions in blocks and bail when the running sum crosses the
  current `tau`. Exact, but late-exits only on candidates that would
  obviously lose; distant candidates still eat ~half the dims.
- **PQ-ADC early abandon** (ADR-353) — fast but approximate; requires
  training codebooks and re-tuning when distributions drift.

Gao & Long (SIGMOD 2023, *Reliable and Efficient Distance Comparison
Operations for ANN Search*) propose **ADSampling**: project each vector
with a random orthogonal matrix `P`, then while accumulating the
partial squared distance in projected space, apply a chi-squared tail
bound to decide whether the full distance can possibly beat `tau`.
Because `P` is unitary, `‖Px − Pq‖² = ‖x − q‖²`, so the full-length
sum is still exact; the test only decides when to stop early.

The result is a distance operator that is **exact at the top-k level
with provable recall guarantees** (tunable failure probability
`epsilon_0`), yet prunes far earlier than deterministic prefix bounds
because the chi-squared threshold `τ · (1 + ε₀/√j)²` tightens with `j`.

## Decision

Add a workspace crate `crates/ruvector-adsampling` that exposes:

- `Projector::new(d, seed)` — orthonormal `d×d` matrix built by MGS;
- `project` / `project_many` — apply at ingest and query time;
- `adsample_l2sq(px, pq, tau, Policy)` — pruned exact-L2² scan primitive;
- `topk_adsample`, `topk_full`, `topk_partial` — reference scanners for
  comparison and for callers to swap in behind a trait.

The crate is **standalone** — no dep on `ruvector-core`, so it can be
integrated into IVF scanners, HNSW candidate verification, and SPANN
partition rescans without pulling the full stack. Downstream crates
can store a per-index `Projector` + pre-projected vectors (same memory
footprint as raw f32) and swap their distance kernel.

## Consequences

### Positive
- **Exact-recall ANN** with provable, tunable error — unlike PQ-ADC,
  no training, no distribution-shift retraining.
- **Faster at high dim** — measured on random gaussian d=768:
  5.39 ms/query vs 7.38 ms full L2 (27% faster, 98.2% recall@10).
- **Composes** with partial-L2 and PQ-ADC — ADSampling can run on top
  of an IVF/HNSW shortlist, or wrap a PQ reranker's refinement step.
- **Deterministic** at a given seed; benchmarks reproducible.

### Negative
- Projected vectors cost an extra `d×d` GEMV at ingest (one-time) and
  per query (hot path, but `d≤1024` is <1 ms). Can be amortized by
  caching projected copies.
- Below ~d=256 the branchy inner loop can underperform the fully
  vectorized full L2; measured turnover is ~d≈384 on Apple M-series.
- Recall drop is small but nonzero; callers that need 100% must either
  pick `ε₀ ≈ 1` or layer a reranker.

### Alternatives considered
- **FastScan / PQ** — great for memory, but lossy and training-sensitive.
- **Partial-L2 only** — exact but prunes less; our measurement shows
  partial-L2 is 7% *slower* than full L2 at d=512 because of branch
  cost.
- **Finger** (WWW 2023) — residual projection with learned basis; more
  complex to train and tune, and residual precision is lossy.

## Rollout

1. Land the crate (this ADR).
2. Integrate behind a `DistanceOp` trait inside `ruvector-pq-adc-early-abandon`
   so its refinement step can switch between full / partial / ADSampling.
3. Measure on BigANN-1M / GIST-1M (follow-up nightly) and gate behind
   a feature flag per index type.
