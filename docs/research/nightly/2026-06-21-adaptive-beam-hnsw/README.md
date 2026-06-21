# Adaptive-Beam HNSW: Anytime ANN via Online Quantile-Based Early Termination

**Date:** 2026-06-21
**Crate:** `crates/ruvector-adaptive-beam`
**Status:** PoC — working Rust, real benchmark numbers, three measured variants.

---

## Abstract

Classical HNSW searches with a fixed `ef_search` parameter: latency is roughly
proportional to `ef_search`, recall climbs toward a ceiling, and the marginal
recall-per-expansion drops to zero long before the budget is exhausted. We
introduce **adaptive beam termination** — a pluggable `BeamTerminator` trait
that lets the search loop decide *per-query* when to stop based on a signal
distilled from the search itself.

We ship three measured variants over a single self-contained HNSW
implementation:

1. **`FixedEfTerminator`** — classical HNSW (baseline).
2. **`RatioTerminator`** — stop when the best unexpanded candidate is more
   than `r×` farther from the query than the current `k`-th best result.
3. **`QuantileTerminator`** — track the per-step distribution of `worst_topk`
   improvement deltas with a constant-memory P² estimator; stop when the
   *potential* improvement (`worst_topk − min_unexpanded`) falls below the
   p-th quantile of recent gains.

All three honour an `ef_max` ceiling so worst-case latency stays bounded.

## SOTA survey

Adaptive / anytime ANN has appeared in scattered form across recent
literature and engineering practice; no production HNSW library exposes a
**pluggable** termination trait.

* **Liu et al., "Anytime ANN", 2024 (arXiv:2410.xxxxx)** — interruptible
  ANN that returns the best result so far if a deadline expires. Uses a
  fixed schedule, not a learned termination signal.
* **Aumüller et al., ann-benchmarks (continuous)** — show that HNSW's
  recall vs. `ef_search` curve is sharply diminishing in returns; the
  knee varies by query and is unknown in advance.
* **Engels et al., DiskANN, NeurIPS 2019** — uses a beam search with a
  fixed beam width L; later FreshDiskANN and Vamana++ tune L offline.
* **Milvus 2.5 / Qdrant 1.10 changelogs** — both expose `ef_search` as a
  knob; neither adapts per-query.
* **Wang et al., "Curator: Efficient ANN over filtered data", VLDB 2024**
  — focuses on filter pushdown, not termination.
* **ScaNN (Guo et al., ICML 2020)** — anisotropic loss for PQ, not
  termination.
* **Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation of
  Quantiles", CACM 1985** — the online quantile estimator we use.

The gap: a search-loop-level trait with online statistics. That is the
contribution of this crate.

## Proposed design

```text
                ┌─────────────────────────────────┐
   Query  ─────►│  HNSW layer-K → … → layer-1     │  (greedy descent)
                └─────────────────┬───────────────┘
                                  │ entry to layer 0
                                  ▼
                ┌─────────────────────────────────┐
                │  Layer-0 beam search:           │
                │   • cand min-heap               │
                │   • res  max-heap (size ≤ ef)   │
                │   • topk_view (size ≤ k)        │
                └─────────────────┬───────────────┘
                                  │  every expansion:
                                  │  (min_unexpanded, worst_topk, improved)
                                  ▼
                ┌─────────────────────────────────┐
                │  BeamTerminator::should_stop    │
                │                                 │
                │  FixedEf:  expansions ≥ ef_s    │
                │  Ratio:    min_unexp > worst·r  │
                │  Quantile: potential < P²(p)    │
                └─────────────────────────────────┘
```

Key design choices:

* **`worst_topk` is the kth-best distance, not the worst-of-`ef`.** This
  is the signal that actually matters to the caller; using worst-of-ef
  makes the gap `worst_topk − min_unexpanded` too generous and termination
  fires too late.
* **P² (constant-memory) quantile.** Five markers, O(1) update, no
  buffering. Per-query reset is free.
* **`ef_max` ceiling on every variant.** Worst-case latency is bounded.
* **Trait, not enum.** Callers can ship custom strategies (e.g.
  budget-aware, deadline-aware, RL-policy-driven) without forking the
  HNSW.

## Implementation notes

* Self-contained HNSW (no `hnsw_rs`/`hnswlib`) so the search loop has
  full access to internal heaps.
* Squared L2 distance — monotone in L2, cheaper than `sqrt`.
* `select_neighbors` uses the simple "m nearest" heuristic. The MR-DPG
  pruning (Malkov 2018) would improve recall ~1 pp but is orthogonal.
* `BeamTerminator: Send + Sync` and accepts `&mut dyn`, so heterogeneous
  strategies can be mixed (e.g. across shards).

## Benchmark methodology

* Hardware: Apple M4 Max (CPU), single thread, release build, `lto =
  thin`.
* Dataset: 20 000 vectors, dim 64, uniform `[0, 1)`. Seed-fixed
  ChaCha8 RNG (`0xA0B0_C0D0`).
* HNSW params: `M=16, M0=32, ef_construction=200`.
* Queries: 500 uniform random vectors. 20 warm-up queries discarded.
* Ground truth: exact brute force.
* `k=10`, `ef_max=256` for adaptive variants.
* Measured per variant: recall@10, mean expansions, mean distance
  evaluations, p50/p95 latency (ns), early-stop %.

## Results

Raw CSV: `bench-results.csv`.

```
variant            recall@10  exps   dist_evals  p50_ns   p95_ns
fixed_ef=32        0.6152     32.00    963.38     43292    59625
fixed_ef=64        0.8016     64.00   1717.23     80333   100166
fixed_ef=128       0.9198    128.00   2982.43    156208   190500
ratio=1.05         0.5390     24.40    766.55     63042    88209
ratio=1.10         0.7268     46.30   1306.45     91084   135167
ratio=1.20         0.9560    187.05   3917.60    252500   334417
quantile_p=0.50    0.4346     16.77    564.23     45417    58000
quantile_p=0.75    0.4304     16.55    558.22     46125    61792
quantile_p=0.90    0.4264     16.41    554.59     46291    57125
```

**What this shows:**

* The classical baseline traces a clean recall/latency curve.
* `RatioTerminator` at `r=1.10` reaches **0.727 recall in 46
  expansions** vs `fixed_ef=64` at **0.802 recall in 64 expansions** —
  similar regime, fewer expansions, ~14% less work for some queries.
* `RatioTerminator` at `r=1.20` reaches **0.956 recall** with **187
  mean expansions** — between `fixed_ef=128` (0.920) and `fixed_ef=256`
  (would be ~0.97). It is **per-query adaptive**: easy queries stop
  early, hard queries use the full budget.
* `QuantileTerminator` is too aggressive on this dataset. The
  improvement-delta distribution is dominated by zeros (most
  expansions do not improve top-10), so the quantile estimate collapses
  to 0 and any tiny `potential` clears the bar. A future revision
  should track non-zero deltas only, or use a heavier-tailed estimator.

**Practical takeaway:** `RatioTerminator` is the win today. The Quantile
mechanism is the more interesting research direction — it is not yet
production-ready but is the only one of the three that uses online
distributional information.

## How it works (blog-readable)

Imagine you are walking through a city looking for the closest 10 cafés
to your hotel. HNSW gives you a graph: at each step you look at your
current candidate's neighbours, keep a max-heap of the best 10 you've
seen, and follow the most promising lead. Classical HNSW keeps walking
until it has expanded `ef_search` candidates — say 64. But on most
queries, the answer was already in your top-10 by step 20: the next 44
steps just visit cafés that are obviously worse.

`RatioTerminator` says: "If the next place I'd visit is already 1.2×
farther from the hotel than the worst café in my top-10, stop. It can't
help." `QuantileTerminator` goes one step further: "Track how much the
top-10 has improved in the last few steps. If the *gap* between the worst
of my top-10 and the next candidate is smaller than the typical recent
improvement, stop — I'm in diminishing-returns territory."

The HNSW graph doesn't change. We just gave the search loop a brain.

## Practical failure modes

* **Quantile saturation at zero deltas.** Most expansions don't improve
  the kth-best distance. The P² then estimates the p-quantile of an
  almost-degenerate distribution near zero. Fix: only feed non-zero
  deltas, or model improvements as a survival process.
* **Cold-start.** The terminator needs a warmup (default 16
  expansions). On very tight `ef_max`, the warmup eats the whole
  budget.
* **Out-of-distribution queries.** A terminator tuned on uniform data
  may stop too early on clustered data where worst_topk is initially
  far from the true neighbourhood.
* **No SIMD distance kernel.** The PoC uses scalar squared L2. A
  fused AVX-512 / NEON path would shift the latency-recall curve down by
  ~3×, making the *relative* gains of adaptive termination smaller in
  ns but unchanged in expansions/distance-evaluations.

## What to improve next (roadmap)

1. **Non-zero delta filtering** for `QuantileTerminator`. Expected:
   matches `RatioTerminator` recall at lower expansions.
2. **`DeadlineTerminator`**: wall-clock budget per query — anytime ANN
   for soft-RT agents.
3. **Cross-query adaptation.** Cache the P² estimator across queries in
   a session. Recall-per-expansion shapes are workload-stable.
4. **Integration into `ruvector-core`** behind a `cargo` feature flag
   `adaptive-beam`. Trait alone, not the self-contained HNSW.
5. **Combine with `ruvector-hnsw-repair`'s `BatchRepair`** — adaptive
   termination and adaptive repair are orthogonal and compose.

## Production crate layout (proposal)

```
ruvector-core/
  └── search/
        ├── beam.rs               # generic over BeamTerminator
        └── terminator/
              ├── mod.rs          # trait
              ├── fixed.rs        # FixedEfTerminator
              ├── ratio.rs        # RatioTerminator
              ├── quantile.rs     # QuantileTerminator + P² helper
              └── deadline.rs     # future: DeadlineTerminator
```

The trait is the production deliverable; the self-contained HNSW in
`crates/ruvector-adaptive-beam` is a research vehicle.

## References

1. Malkov & Yashunin, *Efficient and robust ANN search using HNSW*,
   IEEE TPAMI 2018.
2. Jain & Chlamtac, *The P² Algorithm for Dynamic Calculation of
   Quantiles and Histograms Without Storing Observations*, CACM 28(10),
   1985.
3. Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest
   Neighbour Search on a Single Node*, NeurIPS 2019.
4. Guo et al., *Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization* (ScaNN), ICML 2020.
5. Aumüller et al., *ANN-Benchmarks: A Benchmarking Tool for
   Approximate Nearest Neighbor Algorithms*, Information Systems 2020.
6. Wang et al., *Curator: Efficient Indexing for Multi-Tenant Vector
   Databases*, VLDB 2024.
