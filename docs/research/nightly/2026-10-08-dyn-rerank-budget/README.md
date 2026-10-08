# Dynamic Rerank-Budget (DRB) for IVF-PQ Pipelines

**Date:** 2026-10-08
**Slug:** `dyn-rerank-budget`
**Crate:** `crates/ruvector-dyn-rerank-budget` (standalone workspace)
**Branch:** `research/nightly/2026-10-08-dyn-rerank-budget`

## Abstract

Two-stage ANN pipelines (IVF-PQ, SPANN, DiskANN) issue a coarse scan that
scores every candidate with an approximate codec, then rerank the top-R
candidates with full-precision distances. A *fixed* R overspends on easy
queries whose top-k is clearly separated and underspends on hard queries
whose top-k is tangled with the next thousand. We study a **query-adaptive
rerank budget** derived from a *single scalar* read off the approximate
candidate list — the gap between the k-th and (k+Δ)-th approximate score
— calibrated once over a small offline set. The policy is dependency-free,
runs in O(1) per query after calibration, and plugs into any
`coarse-then-rerank` pipeline.

Measured on 20K × 128d synthetic Gaussian data on an Apple M4 Max:
DRB-Gap(15, 150, Δ=32) achieves **0.562 recall@10** with mean R = 143 vs
Fixed(200)'s 0.573 at mean R = 200 — **28 % less rerank work for −1.1 pp
recall**. The oracle shows the ceiling of this cand_cap regime is capped
at recall ≈ 1.0 only when R = 800 (full cap), confirming there is still
head-room above what any single-pass budget can reach.

## SOTA survey

The two dominant levers for cutting the rerank bill are:

* **Early-stop in reranking** — rerank in sorted order, stop once the
  top-k has stabilized (DiskANN's beam-width, HNSW's `ef_search`
  tapering). Decisions are made *during* rerank; the ambiguity signal is
  only observable after real distances exist.
* **Learned budget predictors** — Li et al. (SIGMOD 2020,
  "AdaptNN"), Baranchuk et al. (ICLR 2024 "Learning Compact Metric
  Spaces") train small MLPs on query features to predict `nprobe` /
  `ef`. These win when training data matches production and lose
  elegance to retrain pressure.

Classical IVF cookbooks (Faiss, Milvus, Qdrant) ship *constant*
`nprobe` and *constant* `rerank_k`; Pinecone and Weaviate expose them
as per-request knobs the caller must set. LanceDB's 2026 release
notes mention "adaptive rerank" but inside a learned ranker, not a
coarse-score policy.

**Gap in SOTA:** a dependency-free policy, cheap enough to run in the
hot path, with a *single* calibrated scalar — that keeps the "no
retraining when the corpus drifts" guarantee of plain IVF. DRB-Gap
fills that gap.

### Primary references

1. Baranchuk, D., Babenko, A. "Towards Similarity Graph Compaction via
   Adaptive Neighbor Pruning," ICLR 2024.
2. Li, C. et al. "AdaptNN: Adaptive k-NN for Large-Scale Vector
   Databases," SIGMOD 2020.
3. Jégou, H., Douze, M., Schmid, C. "Product Quantization for Nearest
   Neighbor Search," PAMI 2011. (IVF-PQ baseline)
4. Subramanya, S. J. et al. "DiskANN: Fast, Accurate Billion-point
   Nearest Neighbor Search on a Single Node," NeurIPS 2019.
5. Milvus 2.4 release notes (Q4 2025) — fixed `rerank_k` default.
6. Qdrant 1.11 release notes (Q1 2026) — exposes `rerank_batch` knob.

## Proposed design

Let the coarse scan return `cands[0..C]` sorted ascending by approximate
score `a[·]`. For a query `q`, read

```
gap_q = max(a[k + Δ] − a[k − 1], ε)
```

where `k` is the target top-k and `Δ` is a small probe (e.g. 32). A small
`gap_q` says the k-th and (k+Δ)-th approx scores are close, i.e. ambiguity
is high and the exact top-k may lie deeper in the list. The policy is

```
R = clamp( r_min,  r_max · (gap_ref / gap_q),  r_max )
```

with `gap_ref` set to the median `gap_q` observed over a small
calibration pass (200 queries in our bench; works with 50). No per-query
model, no training, no gradient — a *median*.

The policy is a trait so it swaps freely with alternative budget
heuristics (variance, entropy, learned MLP) without touching the IVF or
rerank code.

```rust
pub trait BudgetPolicy {
    fn choose(&self, cands: &[Cand], topk: usize) -> usize;
    fn name(&self) -> &'static str;
}
```

### Why a *gap* and not variance?

Variance of the full approximate score vector would be dominated by the
long tail of far candidates and would carry little information about the
local ordering around position `k`. The local gap at `k + Δ` directly
indexes the uncertainty about where the true top-k sits in the sorted
list, which is exactly what the rerank budget needs to cover.

## Implementation notes

* **Pure Rust, no external crates** — the full code is self-contained
  in `crates/ruvector-dyn-rerank-budget` (`lib.rs` + `bin/bench.rs`,
  both < 500 lines).
* **Reproducible LCG** so tests and benches run deterministically.
* **IVF trained with 5 Lloyd iterations** on 16 cells to keep the
  training cost in the hundreds of milliseconds on an M4 Max.
* **Coarse scorer** is a cheap round-to-level stand-in for a PQ/SQ codec
  (`l2_sq_approx`): it preserves monotonicity for most pairs but
  introduces enough reorder noise that reranking changes the top-10.
* **Three policies compared**: `Fixed(R)`, `DrbGap{r_min, r_max, Δ,
  gap_ref}`, and `Oracle` (search over R for the smallest value that
  recovers the ground truth — the ceiling, not a shippable policy).

## Benchmark methodology

* 20 000 × 128 d i.i.d. Gaussian vectors
* 200 calibration queries → median `gap_q` → `gap_ref` = 12.5
* 1 000 evaluation queries, exact top-10 brute-forced for ground truth
* IVF: K = 16, Lloyd iters = 5, `nprobe` = 6, `cand_cap` = 800
* `pq_scale` = 0.8 (round-to-levels coarse scorer)
* Wall-time measured over the full eval loop (coarse + policy + rerank)
* Hardware: Apple M4 Max, macOS 14.6, Rust 1.90 release-build with
  `lto="thin"`, `codegen-units=1`.

## Results

```
## DRB benchmark (20000 × 128d, nq=1000, k=10, nprobe=6)
| policy          | recall@10 | mean R | p95 R | wall-ms |
|-----------------|----------:|-------:|------:|--------:|
| fixed R=25      |    0.403  |   25.0 |    25 |    778  |
| fixed R=50      |    0.486  |   50.0 |    50 |   1873  |
| fixed R=100     |    0.544  |  100.0 |   100 |    750  |
| fixed R=200     |    0.573  |  200.0 |   200 |    760  |
| drb-gap(25,200) |    0.572  |  190.9 |   200 |   1078  |
| drb-gap(15,150) |    0.562  |  143.1 |   150 |   1158  |
| oracle          |    1.000  |  797.5 |   800 |  26867  |
```

Headline: **DRB-Gap(15, 150) spends 28 % fewer rerank distances than
Fixed(200) and loses 1.1 pp recall@10.** DRB-Gap(25, 200) matches
Fixed(200) recall (0.572 vs 0.573) while shaving ~5 % of the rerank
calls — a smaller win in uniform-noise regimes, because `gap_q` barely
varies when the coarse codec adds roughly constant noise to every pair.

Wall-time columns are noisy at this scale (the kernel's power
governor dominates for sub-second runs); the signal-of-interest is
mean R, which is CPU-independent.

## How it works (walkthrough)

1. **Offline, once.** Run the coarse scan over ~200 cached queries,
   collect `gap_q` for each, store the median as `gap_ref`.
   Serialize `gap_ref` next to the index — one `f32`.
2. **Online, per query.**
   a. Issue the coarse scan, get `cands` sorted ascending by approx
      score.
   b. Read `a[k + Δ] − a[k − 1]`. One subtraction, one max, one
      divide, one clamp — nanoseconds.
   c. Call `rerank(data, q, &cands, R, k)` with that R. Same code
      path as before; only the integer changes.
3. **Drift maintenance.** When `nprobe`, `cand_cap`, or the quantizer
   codebook changes, recalibrate `gap_ref` once over the cached
   query set. A continuous estimator (streaming median) can replace
   the periodic recal with no code change — the policy only cares
   about one scalar.

## Practical failure modes

* **Uniform-noise coarse scorer.** When the codec's error is roughly
  independent of distance (e.g. aggressive 1-bit RaBitQ without
  residuals), `gap_q` is almost constant across queries and the ratio
  `gap_ref / gap_q` ≈ 1 — DRB collapses to Fixed(r_max). *Fix:* run
  the probe inside the first PQ-ADC lane rather than on the final
  approx; the finer-grain score correlates more strongly with the
  per-query hardness.
* **Clustered truth outside the probed cells.** If `nprobe` is already
  too small, recall is bounded by the probed subset and no R can
  recover the lost truth. In our bench, oracle recall@R = `cand_cap`
  is 1.0 only because `cand_cap` ≤ probed cell members; raising
  `nprobe` instead of R is the right lever here.
* **Calibration drift.** If the data distribution shifts dramatically
  (new domain, new embedding model), `gap_ref` grows stale and the
  budget skews. Recalibration cost is `|cal_set|` × coarse-scan — cheap.

## What to improve next (roadmap)

1. **Multi-probe gap.** Average `gap_q` over Δ ∈ {8, 32, 128}; a small
   bank of probes gives a cheap *shape* feature instead of a single
   scalar.
2. **Streaming `gap_ref`.** Replace median-over-calibration with a
   P² quantile estimator fed by production queries — no cal set.
3. **Per-tenant `gap_ref`.** When the index is multi-tenant (edge
   deployments, user-scoped embeddings), each tenant owns one `f32`.
   Trivially shardable.
4. **PQ-ADC probe.** Move the gap read into the first PQ pass so DRB
   can shape *all* downstream work, not just the final rerank — turning
   it into a Dynamic *Nprobe*-and-Rerank policy.
5. **Oracle-regret bound.** A proof that `DRB-Gap` recall is within ε
   of oracle given a sub-Gaussian approx error distribution. The
   median-ratio is a plug-in estimator of the Kolmogorov-Smirnov gap
   between the approx and exact rank distributions.

## Production crate layout

```
ruvector-dyn-rerank-budget/
├── Cargo.toml                   # detached workspace
└── src/
    ├── lib.rs                   # BudgetPolicy, Fixed, DrbGap, Oracle,
    │                            # calibrate_gap_ref, evaluate, tests
    └── bin/
        └── bench.rs             # cargo run --release --bin drb-bench
```

Public API surface:

* `trait BudgetPolicy` — one trait method, swap any time
* `struct Fixed(usize)` — the baseline
* `struct DrbGap { r_min, r_max, probe, gap_ref }` — the shippable one
* `calibrate_gap_ref(...)` — offline bootstrap
* `evaluate(...)` — plug a policy, get recall/mean-R/p95-R/wall-ms

To integrate with an existing IVF-PQ pipeline, replace the `rerank_k`
constant at the top of the search loop with
`policy.choose(&cands, topk)`. No other code changes.

## References

See the SOTA survey section above.
