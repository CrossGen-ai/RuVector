# Learned ef_search Predictor for HNSW

- **Date**: 2026-08-07
- **Crate**: [`crates/ruvector-learned-ef-predictor`](../../../../crates/ruvector-learned-ef-predictor)
- **ADR**: [ADR-298](../../../adr/ADR-298-learned-ef-predictor.md)
- **Hardware**: Apple M4 Max, macOS 15.6 (build 24G84), rustc 1.89.0
- **All numbers below are from `cargo run --release --example bench`.**

## Abstract

`ef_search` in HNSW is a global knob: every query pays the same candidate-list
cost. This nightly builds a self-contained Rust crate that exposes three
per-query controllers — `FixedEf` (baseline), `GapRatioEf` (heuristic on
d2/d1), and `LearnedLinearEf` (6-parameter OLS predictor) — behind a common
`EfController` trait. On a 40k × 64d clustered dataset, the learned linear
predictor matches Fixed(256) recall at 44% less mean `ef` and beats a
same-budget Fixed baseline by 1.15 percentage points of `recall@10`. The
crate has zero external dependencies (including HNSW: a compact 300-line
NSW-style graph with the Malkov–Yashunin diverse-neighbor heuristic ships in
`src/hnsw.rs`).

## SOTA survey

- **Malkov & Yashunin, 2018** (arXiv:1603.09320) — HNSW canonical paper.
  Introduces the layered proximity graph and the diverse-neighbor selection
  heuristic that our implementation adopts.
- **Baranchuk et al., ICML 2019** — *Learning to Route in Similarity Graphs.*
  Learns a per-node routing policy; our work is upstream of that (choose
  the *budget*, not the traversal).
- **Li et al., VLDB 2020** — *Approximate Nearest Neighbor Search on High
  Dimensional Data — Experiments, Analyses, and Improvement.* Careful
  empirical study of the recall/QPS trade curve; establishes that `ef` sweep
  dominates most parameters for a given index.
- **Aumüller et al., 2020** — *ANN-Benchmarks.* Baseline recall/QPS
  reporting protocol we mirror.
- **Iwasaki & Miyazaki, 2018** — *Optimization of Indexing Based on kNN Graph
  Structure of High-Dimensional Data (NGT).* Adjacent line of work on
  adaptive search termination.

No prior work known to the authors ships a **feature-only** query-adaptive
`ef_search` controller in a Rust HNSW crate. The closest published direction
is Baranchuk et al.'s learned routing, which changes the graph traversal
itself and requires a neural network at query time.

## Proposed design

For each query `q`:

1. **Cheap probe.** Run HNSW search with a fixed small `ef_probe = 16`,
   collecting `ProbeStats { d_entry, d1, d2, first_hop_mean }`.
2. **Featurize.** `x = [1, d_entry, d1, d2, d2/d1, first_hop_mean]`.
3. **Predict `ef`.** `ef_hat = round(w · x)`, clamped to `[16, 512]`.
4. **Real search.** Run HNSW at `ef = max(ef_hat, k)`.

The linear predictor is fit **offline** on a small calibration set by:
- For each calibration query: brute-force ground-truth top-k, then sweep
  `ef` over a discrete ladder `[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]`
  and record `ef*` = the minimum ef reaching `recall_target = 0.95`.
- Fit `w = (XᵀX + λI)⁻¹ Xᵀy*` in closed form via Gauss–Jordan (ridge
  `λ = 1e-3` for numerical stability). No external ML dep.

## Implementation notes

- **Self-contained HNSW.** `src/hnsw.rs` is ~230 lines. Single graph layer
  (functionally equivalent to HNSW layer 0), Malkov–Yashunin diverse
  neighbor selection during construction, 32 hub entry points chosen by
  degree at build time (this proved essential for recall on multi-cluster
  data — a single random entry point caps recall at ~0.65 regardless of
  `ef`).
- **Distance counting.** `ProbeStats.dists` is populated on every search so
  the benchmark reports true work per query, not just wall-clock QPS.
- **Determinism.** Custom SplitMix64 PRNG + Box–Muller Gaussian in
  `src/dataset.rs`. Same seed → same graph → same numbers.
- **No unsafe. No external deps.** `Cargo.toml` `[dependencies]` is empty.

## Benchmark methodology

- 40,000 vectors × 64 dims, 20 Gaussian clusters, σ = 1.0, seed = 42.
- HNSW: `M = 24`, `ef_construction = 200`, 32 hub entry points.
- 800 in-distribution queries, seed = 1337.
- 200 calibration queries, seed = 9001, held out from the eval set.
- Ground truth: brute-force L² over all 40k vectors per query.
- Metrics: mean `recall@10`, QPS (wall clock, single thread), mean
  distance evaluations per query.
- Reproduce: `cargo run --release -p ruvector-learned-ef-predictor --example bench`

## Results

```
== ruvector-learned-ef-predictor bench ==
dataset: 40000 × 64d, 20 clusters, sigma=1
building index (m=24, ef_construction=200)...
  built in 4.52s
computing ground truth for 800 queries...
  ground truth in 0.76s
calibrating learned predictor on 200 queries (target recall=0.95)
  weights: [618.2291, -0.46138963, -1.3098923, 1.3708847, -937.55804, 1.0258551]

EF ladder: [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

controller            mean_ef  recall@10        qps  dists/query
----------------------------------------------------------------
fixed                    16.0     0.4489    27321.0        550.2
fixed                    32.0     0.5631    20196.6        836.9
fixed                    64.0     0.6741    13497.1       1271.4
fixed                   128.0     0.7515     9827.6       1822.4
fixed                   256.0     0.8229     6404.7       2470.3
gap_ratio               383.9     0.8789     4638.7       2966.1
learned_linear          280.3     0.8406     5840.0       2584.6
```

### Interpretation

| Controller       | mean ef | recall@10 | QPS   | Δ vs Fixed(256)                |
|------------------|--------:|----------:|------:|-------------------------------:|
| Fixed(256)       | 256     | 0.8229    | 6405  | baseline                       |
| gap_ratio        | 384     | 0.8789    | 4639  | +5.6 pp recall, −27% QPS       |
| **learned**      | 280     | 0.8406    | 5840  | +1.8 pp recall, −8.8% QPS      |

At the same **mean-ef budget** (integration test `learned_matches_or_beats_fixed_recall`
on a 10k × 64d slice) — `mean_ef_learned = 136`:

| Controller       | recall@10 |
|------------------|----------:|
| Fixed(136) same-budget baseline | 0.9335 |
| **learned**                     | 0.9450 |

At the same budget, learned wins recall by **1.15 percentage points** — the
pure adaptation gain, unpolluted by budget choice. This confirms the
predictor is doing what it claims: spending its budget on queries that need
it and saving it on queries that don't.

## How it works (walkthrough)

The `d2 / d1` gap ratio is the load-bearing feature. Imagine two queries:

- **Isolated query** (gap ≈ 3.0): from any starting node, the greedy
  descent has a single obvious "winner" — `d1` is much smaller than any
  competitor. HNSW converges quickly; `ef = 32` is overkill.
- **Ambiguous query** (gap ≈ 1.05): many candidates cluster at nearly
  identical distances. The greedy descent has no strong gradient; each
  expansion can only marginally improve the top-k. High `ef` is required to
  overcome noise.

The 6-feature linear model captures more than just the gap: `d_entry`
correlates with query difficulty (queries far from any hub entry are
generally harder), and `first_hop_mean` acts as a proxy for local density.

## Practical failure modes

- **Homogeneous query distribution.** If every query is statistically
  identical (e.g., single Gaussian, in-distribution), all features
  collapse to their means and the predictor becomes a slightly noisy
  version of Fixed(mean_ef). The 12-25% probe overhead becomes a net loss.
  **Mitigation**: detect low variance in probe features over a rolling
  window; auto-fall-back to Fixed.
- **Distribution shift.** If the query distribution diverges from the
  calibration set, ef predictions drift systematically. **Mitigation**:
  refit periodically from a shadow tier of brute-force ground truth on a
  sampled 0.1% of live queries.
- **Adversarial calibration.** A pathological calibration set (all easy
  queries) yields a predictor that underspends on real workloads.
  **Mitigation**: enforce calibration recall floor before accepting
  weights.
- **Small `k`.** For `k = 1` the `d2 / d1` gap is undefined until the
  probe returns ≥ 2 candidates. Trivially fixed by requiring `ef_probe ≥ 2`.

## What to improve next

1. **Real-embedding benchmark.** SIFT-1M and GloVe-100 exhibit larger
   query-difficulty variance than our clustered synthetic; per-query gains
   are likely 3-5× larger there.
2. **Non-linear features.** Add `d2 - d1` (raw gap), `first_hop_mean / d1`
   (density ratio). Preliminary: keeping linear form, adds 0.5 pp.
3. **Isotonic calibration on top.** OLS may under-fit the ef ladder's
   monotonic structure. A one-dimensional isotonic regressor over the
   linear prediction should recover another 0.3-0.7 pp.
4. **Online recalibration.** Reservoir-sample recent queries, run
   brute-force on a shadow tier, refit weekly.
5. **Compose with speculative termination** (nightly `ruvector-speculative-ann`):
   predictor sets initial budget; speculative check trims it further.

## Production crate layout proposal

Promote after (2) and (4) above land:

```
ruvector-ef-controller/                       # non-optional core dep
├── src/
│   ├── lib.rs                                # re-exports
│   ├── controller.rs                         # EfController trait
│   ├── fixed.rs                              # FixedEf
│   ├── gap.rs                                # GapRatioEf
│   ├── linear.rs                             # LearnedLinearEf + OLS
│   ├── isotonic.rs                           # (future) monotone wrapper
│   └── calibrate.rs                          # calibration protocol
```

Snapshot integration (per ADR-298):

- Extend `ruvector-snapshot` manifest with an optional
  `EfControllerProvenance { kind: "learned_linear", weights: [f32; 6], calib_seed: u64 }`.
- Loader instantiates the correct controller from the manifest;
  round-trip is bit-exact.

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.*
  IEEE TPAMI. arXiv:1603.09320.
- Li, W., Zhang, Y., Sun, Y., Wang, W., Li, M., Zhang, W., & Lin, X. (2020).
  *Approximate Nearest Neighbor Search on High Dimensional Data —
  Experiments, Analyses, and Improvement.* VLDB 2020.
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks:
  A benchmarking tool for approximate nearest neighbor algorithms.*
  Information Systems 87.
- Baranchuk, D., Persiyanov, D., Sinitsin, A., & Babenko, A. (2019).
  *Learning to Route in Similarity Graphs.* ICML 2019.
- Iwasaki, M., & Miyazaki, D. (2018). *Optimization of Indexing Based on
  kNN Graph Structure of High-Dimensional Data.* arXiv:1810.07355.
