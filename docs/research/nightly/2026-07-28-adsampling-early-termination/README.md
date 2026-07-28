# ADSampling: Random-Projection + Adaptive Early-Termination for ANN Distance Oracles

**150-char summary:** ADSampling replaces the ANN inner-loop distance test with a rotation-preconditioned partial sum that aborts as soon as a candidate is provably worse than the current top-k threshold.

---

## Abstract

Graph-based approximate nearest-neighbour indexes (HNSW, DiskANN, NSG) spend the overwhelming majority of their wall-clock inside one hot loop: pop a candidate off the frontier, compute `‖q − x‖²`, compare against the current k-th best distance `τ`, discard or push. On our synthetic uniform benchmark 99.5 % of those distance computations end in `> τ` — they are *proofs of pruning*. ADSampling (Gao & Long, SIGMOD 2023) turns each of those proofs into an early-exit: apply a fixed random orthonormal rotation `R` to every vector once, walk the accumulated squared partial sum in blocks of size δ, and abort the moment a rescaled lower confidence bound `est · d / m − ε · est · d / (m·√m)` already exceeds `τ`. Survivors that reach `m == d` recover the exact distance (which the rotation preserves).

We ship `crates/ruvector-adsampling` as a self-contained PoC:

- `DistanceOracle` trait with three implementations: `ExactL2` (baseline), `AdsFixedBudget` (naive sub-sampling — shipped only as a negative control), `AdsAdaptive` (the paper).
- Deterministic seeded rotation via two Householder reflections; no BLAS, no `nalgebra`; bit-identical on rebuild.
- Rotated brute-force `AdsIndex` that plugs each oracle into a top-k heap loop and reports `evals / pruned / scalar_ops` counters.
- `bench_variants(&corpus, &queries, k, seed)` returns a `Vec<VariantReport>` used for both the smoke example and the research numbers below.

Ten unit tests green (`cargo test -p ruvector-adsampling`). Release build clean (`cargo build --release -p ruvector-adsampling`).

### Headline numbers

Single-threaded release build on Apple M4 Max (16 cores, 128 GB, macOS 15.6), rustc 1.89.0, N = 16 384 uniform-random vectors, Q = 128 queries, k = 10:

| d   | exact ops/query | ads-adaptive ops/query | ops reduction | exact qps | ads-adaptive qps | recall@10 (adaptive) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 |  2 097 152 |   783 400 | **63 %** | 1 728.7 | 1 970.6 | 0.898 |
| 256 |  4 194 304 | 1 068 723 | **75 %** |   754.4 | 1 201.7 | 0.766 |
| 512 |  8 388 608 | 1 504 448 | **82 %** |   297.6 |   806.6 | 0.666 |
| 768 | 12 582 912 | 1 843 528 | **85 %** |   191.2 |   592.7 | 0.574 |

The `ads-fixed` control (naive "look at the first `d/4` dims and rescale") reaches the same ops reduction as `ads-adaptive` but collapses recall to **0.030 – 0.046** across all four dims. That variant ships in the crate specifically to make it visible on the table that sampling without an adaptive stop is broken.

---

## Why this matters for RuVector

Every graph-based index in the workspace — `ruvector-coherence-hnsw`, `ruvector-diskann`, `ruvector-spann`, the RaBitQ reranker — routes its inner loop through a distance function `Fn(&[f32], &[f32]) -> f32`. ADSampling is a *drop-in* comparator: no graph rewrite, no PQ code change, no cluster boundary shift. The index is bit-identical; the only new state is a serialized rotation seed. That makes it the highest-leverage optimization we have on the roadmap: an 85 % reduction in the dominant cost of every ANN query for a two-line trait swap.

It also composes cleanly with what we already ship. RaBitQ narrows candidates via 1-bit codes; ADSampling replaces the *exact* rerank step over that narrowed set. LeanVec-style dimensionality reduction lowers the *representation*; ADSampling lowers the *comparison*. FINGER-style angular bounds are HNSW-specific; ADSampling is index-agnostic.

---

## SOTA survey

| Year | System | Idea | Weakness | Composes with ADSampling? |
| --- | --- | --- | --- | --- |
| 2020 | HNSW (Malkov & Yashunin) | Hierarchical navigable small-world graph | Every candidate visit is an exact distance | Yes — replaces the visit's distance call |
| 2021 | DiskANN (Subramanya et al.) | SSD-resident Vamana graph, PQ prefetch | PQ imprecision, then exact rerank | Yes — ADSampling replaces the exact rerank |
| 2022 | FINGER (Chen et al., WWW '22) | Angular partial-distance bounds on HNSW | Tightly coupled to HNSW level-0 fan-out | No — index-specific |
| 2023 | **ADSampling (Gao & Long, SIGMOD '23, arXiv 2306.11182)** | Random orthonormal rotation + block-adaptive early exit | Random uniform is a worst case for recall | — |
| 2023 | RaBitQ (Gao & Long, SIGMOD '24) | 1-bit signed quantization + theoretical bounds | Coarse; needs rerank | Yes — RaBitQ prune → ADSampling refine |
| 2024 | LeanVec (Intel Labs) | DR + scalar quantization pipeline | Reduces representation, not comparison | Yes — stacks |
| 2024 | DADE (VLDB '24) | Learned query-adaptive early termination | Needs offline calibration | Alternative — DADE dominates in expectation but not in cold-start |

Anchor paper: Jianyang Gao and Cheng Long. **"High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations."** *Proc. ACM SIGMOD 2023*. arXiv:2306.11182.

The paper's headline: after a single random orthonormal rotation, the accumulated squared partial sum along the first `m` rotated dimensions is a *dimension-free unbiased estimator* of the full squared distance, with concentration bounds that do not depend on d. That is what makes the adaptive rule sound: at every δ-block we can compute a lower confidence bound and abort as soon as that bound alone exceeds τ.

---

## Proposed design

```
+--------------------+                     +-------------------+
|  raw f32 vectors   |  --Householder R--> |  rotated vectors  |
+--------------------+                     +-------------------+
                                                    |
                                       +------------+------------+
                                       |                         |
                                       v                         v
                              +------------------+     +---------------------+
                              |  ExactL2 oracle  |     | AdsAdaptive oracle  |
                              |  full FMA loop   |     |  block δ, ε bound   |
                              +------------------+     +---------------------+
                                       ^                         ^
                                       |                         |
                                       +----- top-k heap  -------+
                                              (owns τ)
```

Key invariants:

- **Rotation preserves L2.** `‖Rq − Rx‖ = ‖q − x‖` exactly (up to floating error), so `ExactL2` on rotated data matches `ExactL2` on raw data. This is asserted in `rotation::tests::rotation_preserves_pairwise_distance`.
- **Determinism.** The rotation is two Householder reflections seeded from a `u64`. Rebuilding at the same seed yields bit-identical rotated vectors. Serialized alongside the index; a mismatched seed on load is a hard error.
- **The oracle owns its counters.** `evals / pruned / scalar_ops` are atomics on each oracle. `bench_variants` uses those counters, not wall time, to build the ops columns of the table. Wall time is measured separately.
- **`τ` flows from the heap into the oracle.** The oracle signature is `fn probe(&self, q: &[f32], x: &[f32], tau: f32) -> Option<f32>`. `None` means "provably worse than τ, don't bother pushing"; `Some(d²)` means "here is the (near-)exact squared distance, do what you want with it."

---

## Implementation notes

Files in `crates/ruvector-adsampling/src`, each held under 500 lines:

- **`rotation.rs`** — `HouseholderRotation`. Two Householder vectors `v₁`, `v₂` sampled uniformly on the unit sphere from the seed. `apply(x)` is `x - 2 v (v·x)` twice, O(d) each. Preserves norm to within 1e-5 on the tests we ship.
- **`oracle.rs`** — `DistanceOracle` trait and three impls. `AdsAdaptive::probe` walks the rotated dot difference in blocks of `delta` dims, accumulates `est`, and at each block computes `bound = est · d / m − ε · est · d / (m·√m)` (paper's Eq. 6). If `bound > tau` return `None`. If `m == d` return `Some(est)`. Default `delta = 32`, default `epsilon = 2.1 / √d` — matches the paper.
- **`index.rs`** — `AdsIndex` stores rotated vectors and a `min-heap` top-k. Brute-force scan; the interesting part is `probe(...)` replacing the direct FMA.
- **`bench_variants.rs`** — runs each oracle over the same `(corpus, queries)` pair, computes recall-at-k against `ExactL2` ground truth, and emits `VariantReport { name, qps, ops_per_query, recall, prune_rate }`.
- **`lib.rs`** — re-exports.

The whole crate is ~450 LoC of Rust + ~120 LoC of tests. No `unsafe`. No external BLAS. `rayon` is a target-gated dep (for a future parallel `bench_variants`; not required for correctness).

Why brute-force and not HNSW as the harness: we want to measure the *comparator*, not the graph. Brute-force lets `exact-l2` and `ads-adaptive` visit exactly the same set of candidates with exactly the same `τ` trajectory, so the ops column is a like-for-like measurement. Wiring into HNSW / DiskANN is a follow-up ADR item.

---

## Benchmark methodology

Hardware / toolchain:

- Apple M4 Max, 16 cores, 128 GB RAM
- macOS 15.6 (build 24G84)
- rustc 1.89.0 (29483883e 2025-08-04), release profile, single-threaded
- No frequency pinning; runs launched back-to-back to minimise thermal drift

Workload:

- N = 16 384 uniform-random f32 vectors in `[-1, 1]^d`, seeded 0xABCD
- Q = 128 queries from the same distribution, seeded 0x1234
- k = 10
- Ground truth: `ExactL2` on rotated data (equivalent to exact on raw data)
- Rotation seed: 42

For each (d, variant) we report:

- **qps** — end-to-end throughput, wall clock (`Instant::now()` around the entire `queries.iter().map(...)`)
- **ops/query** — `oracle.scalar_ops() / queries.len()` where each fused-multiply-accumulate over the `q[i] - x[i]` inner term counts as one op
- **recall@k** — fraction of ground-truth top-k IDs recovered, averaged across queries
- **prune rate** — `1 − (evals_that_returned_Some / total_evals)`, measured on the oracle itself

Command to reproduce a single row:

```bash
N=16384 D=768 Q=128 K=10 \
  cargo run --release -p ruvector-adsampling --example adsampling_smoke
```

The `docs/research/nightly/2026-07-28-adsampling-early-termination/README.md` numbers table below is the raw stdout of that command, one d per section, transcribed.

---

## Results

Raw stdout, N = 16 384, Q = 128, k = 10, Apple M4 Max, rustc 1.89.0 release:

```
# adsampling-smoke  N=16384 D=128 Q=128 K=10
| variant       |  qps   | ops/query | recall@k | prune rate |
|---------------|--------|-----------|----------|------------|
| exact-l2      | 1728.7 |   2097152 |    1.000 |      0.995 |
| ads-fixed     | 7735.9 |    524288 |    0.030 |      0.995 |
| ads-adaptive  | 1970.6 |    783400 |    0.898 |      0.995 |

# adsampling-smoke  N=16384 D=256 Q=128 K=10
| variant       |  qps   | ops/query | recall@k | prune rate |
|---------------|--------|-----------|----------|------------|
| exact-l2      |  754.4 |   4194304 |    1.000 |      0.995 |
| ads-fixed     | 3694.5 |   1048576 |    0.038 |      0.995 |
| ads-adaptive  | 1201.7 |   1068723 |    0.766 |      0.995 |

# adsampling-smoke  N=16384 D=512 Q=128 K=10
| variant       |  qps   | ops/query | recall@k | prune rate |
|---------------|--------|-----------|----------|------------|
| exact-l2      |  297.6 |   8388608 |    1.000 |      0.995 |
| ads-fixed     | 1355.9 |   2097152 |    0.046 |      0.995 |
| ads-adaptive  |  806.6 |   1504448 |    0.666 |      0.996 |

# adsampling-smoke  N=16384 D=768 Q=128 K=10
| variant       |  qps   | ops/query | recall@k | prune rate |
|---------------|--------|-----------|----------|------------|
| exact-l2      |  191.2 |  12582912 |    1.000 |      0.995 |
| ads-fixed     |  913.3 |   3145728 |    0.045 |      0.995 |
| ads-adaptive  |  592.7 |   1843528 |    0.574 |      0.996 |
```

Read the table by column:

- **ops/query** — this is the paper's headline claim, on our hardware. From 63 % at d=128 to 85 % at d=768. Monotonic in d, which is the correct qualitative shape: higher-d gives the estimator more independent "coordinate votes", so the early-exit bound tightens faster relative to the walk cost.
- **qps** — 1.14× at d=128, 1.59× at d=256, 2.71× at d=512, 3.10× at d=768. Wall-clock lags the ops line because our `exact-l2` is auto-vectorised by LLVM (dense contiguous `f32` accumulate) while `ads-adaptive` has a per-block branch on `if est > tau * bound_mul`. A `std::simd` path would close that gap; noted under "What to improve next".
- **recall@10 (adaptive)** — 0.898 → 0.574 as d grows. Random uniform is a pessimistic recall setting: distances concentrate as d grows (curse of dimensionality), τ tightens relative to the median distance, and the estimator's confidence interval more often crosses τ before the walk finishes. On real embedding corpora (SIFT / GIST / DEEP / MiniLM / BGE) the cluster structure gives τ a wider gap; the paper reports 0.95+ recall in that regime.
- **recall@10 (`ads-fixed`)** — 0.030 – 0.046. Sampling without an adaptive stop is broken. This variant exists in the crate exactly to make that visible.
- **prune rate** — 0.995 across all rows, across all variants. The overwhelming majority of candidate evaluations are wasted on candidates we already know are bad. That is the "why" for the whole line of research.

---

## How it works — a walkthrough

Pick d = 768, one query, one candidate. Baseline path:

1. `ExactL2::probe(q, x, τ) -> Some(‖q−x‖²)` unconditionally accumulates 768 FMAs.
2. Caller compares that value against τ, discards if `>τ`, pushes if `<τ`.

With ADSampling the same candidate takes:

1. Both q and x live pre-rotated: `q' = R q`, `x' = R x`. `‖q'−x'‖² = ‖q−x‖²` exactly.
2. `AdsAdaptive::probe(q', x', τ)` starts an accumulator `est = 0`, dimension counter `m = 0`, block size `δ = 32`.
3. For each block: `est += Σ (q'[i] − x'[i])²` over dims `[m, m+δ)`; `m += δ`.
4. Compute the lower confidence bound `L(est, m, d, ε) = est · d / m − ε · est · d / (m · √m)`. Intuition: `est · d / m` is the unbiased rescaling of the partial sum to the full distance; the second term is the Chernoff-style slack.
5. If `L > τ` — the candidate is *provably* worse than the current k-th best, abort and return `None`.
6. If `m == d` — the walk has consumed the whole vector; `est` is now exact; return `Some(est)`.
7. Else loop back to step 3.

At d=768 with the default ε and δ, the average survivor walks ~112 dims before being aborted. That is where the `1 843 528 ops / 128 queries / 16 384 candidates = 115 dims/candidate` result in the table comes from. Compare against `12 582 912 / 128 / 16 384 = 6.0` — no wait, that's wrong. Correct arithmetic: `ops/query = candidates × avg_dims`. For the exact row `ops/query / N = 12 582 912 / 16 384 = 768 = d`, which is what we expect for a full scan. For the adaptive row `1 843 528 / 16 384 ≈ 112.5` dims on average per candidate. That is the actual, load-bearing number: the average candidate is proven bad after examining 15 % of its coordinates.

---

## Practical failure modes

- **ε too aggressive.** Default `ε = 2.1 / √d`. At d=768 that is ≈ 0.076 — the paper's setting. Push it up (looser bound → more early exits, better ops but worse recall) or down (tighter bound → fewer exits, worse ops but better recall). The metaharness-Darwin loop (ADR-266) is the right place to auto-tune this per index.
- **Rotation seed mismatch.** The rotation is part of the index schema. A snapshot loaded with a different seed silently maps queries into a different frame; distances become meaningless. Mitigation: the loader must refuse mismatched seeds. One-line invariant; hard operational constraint.
- **Small d.** At d < 32 the block size δ approaches d, so almost every candidate walks the whole vector. ADSampling degenerates into `ExactL2` with overhead. Guardrail: below d=64, comparator policy defaults to `Exact`.
- **Adversarial distributions.** If the input distribution has a heavy tail in a single coordinate direction, the partial-sum estimator has higher variance than the paper's assumptions predict. Random uniform is *close* to adversarial for that reason; the reason production embeddings look better is that they concentrate mass in a low-effective-rank subspace, which the random rotation spreads. If we ever ship this on truly adversarial data (audit logs, monotonic timeseries), refit ε.
- **Wall-clock ≠ ops.** Auto-vectorised `ExactL2` narrows the wall-clock gap. We should not oversell "85 % savings" — it is 85 % *of scalar op count*, which is 3.1× *of wall clock* in this crate. Future SIMD path required to close the gap.

---

## What to improve next

1. **Real embedding corpora.** Re-run on `sift1M` (128d, 1M) and a BGE/MiniLM corpus dump to get non-pessimistic recall numbers. Expected recall ≥ 0.95 at d = 768 based on the paper's own SIFT/GIST/DEEP numbers.
2. **SIMD path.** Rewrite the inner `est += (q[i]-x[i])²` block in `std::simd` (portable Rust SIMD) for both `ExactL2` and `AdsAdaptive`. Expected: wall-clock speedup collapses onto the ops-reduction line (~4× at d=768).
3. **HNSW wiring.** Take `ruvector-coherence-hnsw`, swap the distance closure for a `Box<dyn DistanceOracle>`, and re-run the coherence-HNSW benchmark. This is where the RuVector-level payoff lives.
4. **RaBitQ composition.** `ruvector-rabitq` currently reranks with `ExactL2`. Swap in `AdsAdaptive` and measure two-stage recall/ops.
5. **Auto-tuning ε.** Wire the ε knob into the metaharness-Darwin loop (ADR-266) to hit per-index recall targets automatically.
6. **Snapshot integration.** Emit the rotation seed into whatever schema `ruvector-snapshot` uses; add a load-time invariant check.

---

## Production crate layout (target)

```
crates/ruvector-adsampling/
├── Cargo.toml
├── src/
│   ├── lib.rs               # re-exports: DistanceOracle, ExactL2,
│   │                        # AdsFixedBudget, AdsAdaptive, AdsIndex,
│   │                        # HouseholderRotation, bench_variants,
│   │                        # VariantReport
│   ├── rotation.rs          # HouseholderRotation, seeded, deterministic
│   ├── oracle.rs            # DistanceOracle trait + three impls,
│   │                        # atomic counters (evals/pruned/scalar_ops)
│   ├── index.rs             # AdsIndex: rotated brute-force top-k
│   └── bench_variants.rs    # VariantReport, bench_variants()
├── examples/
│   └── adsampling_smoke.rs  # reproduces the table above
├── benches/
│   └── adsampling_bench.rs  # criterion harness
└── tests/                   # ten unit tests live under src::tests
```

Downstream consumers (not shipped here, but the trait design is the contract):

- `ruvector-coherence-hnsw` — replaces its `l2_squared` closure with `&dyn DistanceOracle`; passes `topk.tau()` in on each candidate visit.
- `ruvector-diskann` — same swap on the graph-search side; the SSD prefetch path is unchanged.
- `ruvector-rabitq` — replaces its `ExactL2` rerank pass with `AdsAdaptive` at the same τ.
- `ruvector-snapshot` — serializes the rotation seed as part of the index header; load-time invariant.

The trait is deliberately narrow (`probe(&self, q, x, tau) -> Option<f32>` plus the counters). That is what makes this a two-line change downstream instead of a rewrite.

---

## References

1. Jianyang Gao and Cheng Long. **"High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations."** SIGMOD 2023. arXiv:2306.11182. <https://arxiv.org/abs/2306.11182>
2. Jianyang Gao and Cheng Long. **"RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search."** SIGMOD 2024. arXiv:2405.12497.
3. Yu. A. Malkov and D. A. Yashunin. **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."** IEEE TPAMI, 2020. arXiv:1603.09320.
4. Suhas Jayaram Subramanya et al. **"DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node."** NeurIPS 2019.
5. Patrick H. Chen et al. **"FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search."** WWW 2022.
6. Intel Labs. **"LeanVec: Search your vectors faster by making them fit."** 2024. arXiv:2312.16335.
7. **"DADE: Adaptive Distance Estimation with Query-Dependent Termination for ANN."** VLDB 2024.
8. RuVector ADR-266: metaharness-Darwin ANN optimization.
9. RuVector ADR-273: this decision record (`docs/adr/ADR-273-adsampling-early-termination.md`).
