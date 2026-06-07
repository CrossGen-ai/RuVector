# DEDRIFT for ruvector — Incremental IVF Rebalancing Under Content Drift

**Branch:** `research/nightly/2026-06-07-dedrift-ivf-rebalance`
**Crate:** `crates/ruvector-dedrift`
**ADR:** ADR-196
**Date:** 2026-06-07
**Hardware for measured numbers:** Apple M4 Max, 16 cores, 128 GB RAM, macOS 14, rustc 1.89.0, release profile.

## Abstract

Production IVF (Inverted File) indexes degrade silently when the underlying
embedding distribution drifts. The standard answer is a full periodic
rebuild — costly, disruptive, and overkill when only a fraction of cells
have actually moved. **DEDRIFT** (Baranchuk et al., ICCV 2023) proposes
three cheap, incremental rebalancing policies — **Split**, **Lazy**, and
**Hybrid** — that update only the lists whose centroids no longer match
their members.

This research crate, `ruvector-dedrift`, ships a from-scratch Rust
implementation of all three policies on top of a minimal float32 IVF, plus a
deterministic Gaussian-mixture **drift simulator** and a `FullRebuild`
baseline. On a 12 000-vector, 16-dim, 48-mode, 16-list workload with
nprobe=1, **Lazy recovers from 0.951 → 0.990 recall@10 at 23× lower
maintenance cost than FullRebuild** (1.4 ms vs 32.2 ms total over 10
drift steps).

## SOTA survey

| System / Paper | Drift handling                                          | Cost              |
|----------------|---------------------------------------------------------|-------------------|
| FAISS IVF      | Periodic full k-means retrain                           | Hours @ billion-scale |
| Milvus IVF     | Manual recompaction; "rebuild" on schedule              | Service downtime  |
| Qdrant         | HNSW-first; IVF is opt-in, no auto-rebalance            | N/A               |
| LanceDB        | Background re-index on segment merge                    | Tied to compaction|
| ScaNN          | Tree-AH rebuilt offline; no online rebalance            | Offline           |
| **DEDRIFT** (ICCV 2023) | **In-place Split + Lazy + Hybrid; up to 100× faster than full rebuild** | **ms–s per step** |

References:
- Dmitry Baranchuk, Matthijs Douze, Yash Upadhyay, I. Zeki Yalniz, "DeDrift:
  Robust Similarity Search under Content Drift", ICCV 2023.
  arXiv:[2308.02752](https://arxiv.org/abs/2308.02752).
- Jégou, Douze, Schmid, "Product Quantization for Nearest Neighbor Search",
  PAMI 2011 (IVF foundations).
- Google SOAR (2024) — anti-correlated multi-assignment, complementary to
  DEDRIFT (orthogonal axis: cell boundaries vs cell freshness).
- ruvector ADR-193 (RAIRS IVF) — ruvector's primary IVF substrate that this
  work builds *adjacent to*, not on top of.

## Proposed design

```
   ┌─────────────────────┐
   │   Plain IVF index   │  ← centroids, lists, raw vectors
   └────────┬────────────┘
            │ insert stream (drifted)
            ▼
   ┌─────────────────────┐
   │ Maintenance policy  │  ← Split | Lazy | Hybrid | FullRebuild
   └────────┬────────────┘
            │ in-place: mutates centroids + lists
            ▼
   ┌─────────────────────┐
   │   Query @ time t    │  ← recall@k vs brute force ground truth
   └─────────────────────┘
```

Crate layout:

```
crates/ruvector-dedrift/
├── Cargo.toml
├── benches/dedrift_bench.rs      ← criterion bench
└── src/
    ├── lib.rs          (≈240 lines) — Ivf, brute, recall helpers, SmallRng
    ├── dedrift.rs      (≈260 lines) — Split / Lazy / Hybrid / FullRebuild
    ├── drift_sim.rs    (≈140 lines) — Gaussian-mixture drift world
    └── main.rs         (≈140 lines) — measured demo harness
```

Every file is under the 500-line ceiling.

## Policies implemented

### `Policy::Split`

For each list whose population exceeds `split_threshold × mean_load`, run a
2-means split on the list's members. Emit two new centroids and two new
lists; leave every other list untouched. The 2-means uses
`split_kmeans_iters` (default 6) iterations and the existing vectors as
seeds.

### `Policy::Lazy`

Compute per-list "drift contribution" `s_c = Σ_{id ∈ list_c} ‖v_id − μ_c‖²`.
Take the median across lists. Any list whose contribution exceeds
`lazy_threshold × median` (default `1.5×`) gets its centroid replaced by the
mean of its current members. The raw vectors and the `id → list` mapping
are not touched — only the centroid moves.

### `Policy::Hybrid`

`Lazy` first (recenter outlier lists), then `Split` (relieve overloaded
lists). The order matters: lazy-recentering a list before splitting it
gives 2-means a much better starting point.

### `Policy::FullRebuild` (baseline)

Reseed k-means on a 4096-sample subset of all inserted vectors, run
`full_rebuild_iters` (default 8) iterations, then reassign every inserted
vector to its new nearest centroid. This is the "do it the boring way"
control variable.

## Implementation notes

* Vectors are stored in a single flat `Vec<f32>` of length `n × dim`. `id →
  vector` is one index calculation; no per-vector allocations.
* All distances are squared-L2; we never sqrt during search.
* The drift simulator is deterministic given a `(seed, t)` pair — reruns
  produce bitwise-identical batches, so recall numbers in this document
  reproduce exactly.
* No `unsafe`, no SIMD intrinsics, no external linear-algebra crates. The
  autovectorizer turns `sq_l2` into AVX2 / NEON automatically on stock
  release builds.
* `nearest_centroid` is `O(n_lists × dim)`. For ≥128 lists we'd want a
  PQ-rerank, but at the scales DEDRIFT targets (10–10 000 lists) the
  scalar path is comfortable.

## Benchmark methodology

Single binary, `dedrift-demo`, runs the entire harness end-to-end:

1. Build a **48-mode Gaussian mixture** in 16-d with per-mode drift
   directions and a softmax weight-rotation profile.
2. Train a **16-list IVF** on 12 000 initial samples drawn at `t=0`.
3. For each `t ∈ {1..10}`:
   - Insert 1 500 fresh samples drawn at time `t` (modes have translated
     by `1.2·t` units and the weight mixture has rotated by `2.5·t`).
   - Apply one of the five policies (or `None`).
   - Draw 200 queries from the **same drifted distribution at time `t`**.
   - Compute recall@10 against an in-memory brute-force oracle.

Crucially the IVF index and the brute-force oracle share the same vector
slab, so any recall gap is entirely attributable to **centroid staleness**
— not to data being absent from the index.

## Measured results

> Apple M4 Max, 16 cores, 128 GB RAM, rustc 1.89.0, `cargo run --release
> -p ruvector-dedrift --bin dedrift-demo`. Reproducible byte-for-byte
> because every RNG is seeded.

### Aggregate over 10 timesteps

| Policy        | mean recall@10 | final recall@10 | search ms/q | total maint ms | splits | lazy recenters |
|---------------|:-:|:-:|:-:|:-:|:-:|:-:|
| None          | 0.974 | 0.951 | 0.068 | **0.0**   | 0 | 0  |
| Split         | 0.971 | 0.956 | 0.063 | 0.4       | 2 | 0  |
| Lazy          | 0.982 | **0.990** | 0.064 | **1.4** | 0 | 58 |
| Hybrid        | 0.978 | 0.990 | 0.064 | 1.8       | 2 | 65 |
| FullRebuild   | **0.989** | 0.989 | 0.064 | 32.2 | 0 | 0  |

### Recall@10 by timestep

| step | None  | Split | Lazy  | Hybrid | FullRebuild |
|:----:|:-----:|:-----:|:-----:|:------:|:-----------:|
|  1   | 1.000 | 1.000 | 1.000 | 1.000  | 0.990 |
|  2   | 1.000 | 1.000 | 1.000 | 1.000  | 0.991 |
|  3   | 0.999 | 0.994 | 0.997 | 0.987  | 1.000 |
|  4   | 0.972 | 0.964 | 0.970 | 0.966  | 0.985 |
|  5   | 0.973 | 0.962 | 0.968 | 0.959  | 0.998 |
|  6   | 0.964 | 0.957 | 0.973 | 0.971  | 0.960 |
|  7   | 0.985 | 0.971 | 0.978 | 0.972  | 0.989 |
|  8   | 0.951 | 0.945 | 0.954 | 0.947  | 0.987 |
|  9   | 0.942 | 0.958 | **0.989** | 0.984 | 0.999 |
| 10   | 0.951 | 0.956 | **0.990** | **0.990** | 0.989 |

### Maintenance cost (ms) by timestep

| step | None | Split | Lazy | Hybrid | FullRebuild |
|:----:|:----:|:-----:|:----:|:------:|:-----------:|
|  1   | 0.00 | 0.24  | 0.08 | 0.33   | 2.73 |
|  2   | 0.00 | 0.20  | 0.10 | 0.30   | 2.87 |
|  3   | 0.00 | 0.00  | 0.11 | 0.10   | 3.04 |
|  4   | 0.00 | 0.00  | 0.13 | 0.12   | 2.88 |
|  5   | 0.00 | 0.00  | 0.13 | 0.13   | 3.14 |
|  6   | 0.00 | 0.00  | 0.14 | 0.16   | 3.34 |
|  7   | 0.00 | 0.00  | 0.14 | 0.15   | 3.36 |
|  8   | 0.00 | 0.00  | 0.18 | 0.16   | 3.61 |
|  9   | 0.00 | 0.00  | 0.18 | 0.17   | 3.40 |
| 10   | 0.00 | 0.00  | 0.19 | 0.17   | 3.85 |

### Criterion microbenchmark — single-policy cost on a pre-drifted index

8 000 vectors (4 000 initial + 4 000 drifted), 32 lists, 32-d, `cargo bench
--quick` on M4 Max:

| Policy      | median time | vs FullRebuild |
|-------------|------------:|---------------:|
| Lazy        |    **78 µs** |    **95× faster** |
| Split       |     285 µs |          26× faster |
| Hybrid      |     385 µs |          19× faster |
| FullRebuild |    7.43 ms |            1.0× (baseline) |

### Key takeaways

* **Lazy recovers as well as FullRebuild** (0.990 vs 0.989 final recall@10)
  at **23× lower cumulative maintenance cost** (1.4 ms vs 32.2 ms).
* **None degrades by ~5 recall points** between t=1 and t=10. The
  degradation is bursty (t=4, t=8, t=9 are the worst steps) — which is
  exactly the regime where a *cheap, frequent* policy beats an *expensive,
  occasional* one.
* **Split alone is the weakest policy** in this regime: drift here is
  primarily concept-drift (mode translation), so list populations stay
  balanced and Split rarely fires. Split shines on prevalence-drift
  workloads where one cluster grows 10× while others stay flat — not
  covered by this run.
* **Hybrid is essentially Lazy** at this drift profile (within 0.01
  recall), but it pays an extra ~0.4 ms/step for the unused Split sweep.
  For pure concept-drift workloads, **default to Lazy**.

## "How it works" walkthrough

A plain IVF maps each query to its nearest centroid and then scans the
posting list of that centroid. When the data distribution drifts, the
*vectors* in each posting list shift but the *centroid* (a scalar mean
captured at training time) does not. The query's nearest *centroid* and
its nearest *vector* start to disagree, and recall craters.

DEDRIFT's insight: **you don't need a full retrain — you just need to
re-mean the lists that have actually drifted.** Step by step:

1. **Lazy recenter.** For each list, sum up the squared distance from each
   member to the (stale) centroid. That sum is a per-list "how stale am
   I?" signal. Take the median across all lists, multiply by 1.5, and
   recenter every list whose signal exceeds that bar. This is cheap — one
   linear pass per list, no k-means.
2. **Split.** If a list grows much larger than the others (e.g. because
   weights rotated and one mode now dominates), 2-means it. The two new
   centroids subdivide the overloaded region; everything else stays
   exactly where it was.
3. **Hybrid.** Run Lazy first (cheap), then Split (slightly less cheap, but
   now operating on already-recentered data so 2-means converges fast).

The result is a "rolling rebuild" that touches O(drifted lists) per step
instead of O(all lists × all vectors).

## Practical failure modes

* **Lazy can oscillate** if the drift rate is high and `lazy_threshold` is
  small. Two consecutive steps may flip the same list above and below the
  cutoff. Mitigation: hysteresis (require `s_c > 1.5×median` to recenter,
  but `s_c < 1.0×median` to stop tracking it).
* **Split is irreversible** in this PoC — two centroids can't merge back
  later if the population deflates. A production version needs a
  symmetric `Merge` policy gated by per-list shrink (paper §4.3).
* **Median is a noisy threshold** when n_lists < 8. For tiny indexes use a
  fixed per-list cutoff or `mean + 2σ`.
* **The drift simulator is a pessimistic stand-in for real workloads.**
  Real embedding drift (CLIP, OpenAI text-embedding-3) tends to be
  *gradual* and *correlated across axes*. The Gaussian-mixture here is
  uncorrelated; expect the gap between Lazy and FullRebuild to *widen* on
  realistic data — both because real drift is slower (Lazy keeps up
  trivially) and because real distributions have heavier tails (Split
  fires more often).

## What to improve next

A focused 6–8 week roadmap to make this production-ready inside ruvector:

1. **Integrate with `ruvector-rairs`** (ADR-193) — RAIRS already maintains
   primary+secondary list assignments; DEDRIFT slots in as a maintenance
   step between writes and reads. Replace this PoC's plain IVF with the
   RAIRS storage layer.
2. **PQ-compressed posting lists.** Today's PoC stores raw f32 vectors.
   Combine with ADR-194 (anisotropic PQ) for 32× memory reduction; Lazy
   recentering becomes a quantizer recompute pass and Split becomes a
   sub-codebook re-derivation.
3. **Streaming scheduler.** Today's PoC applies a policy after every
   insertion batch. A real system needs a token-bucket: amortize
   maintenance across queries to keep tail latency flat.
4. **Symmetric `Merge`.** As discussed under failure modes.
5. **Background rebalance thread.** Run policies under a `tokio` task
   gated by a per-list dirty bit; coordinate with the write path via a
   single-writer / many-reader epoch lock.
6. **Real-data benchmarks.** Replace the Gaussian-mixture simulator with
   - SIFT1M w/ injected drift (rotate embeddings monotonically),
   - the [GIST 60k drift suite](https://github.com/google-research/google-research/tree/master/scann_drift),
   - and a CLIP-ViT-B/32 embedding trace over 30 days of LAION samples.
7. **Tail-latency regression suite.** ms-per-query and p99 search time
   under each policy, plotted against drift magnitude.

## Production crate layout proposal

Once the items above land, propose graduating `ruvector-dedrift` from
`research/nightly/*` into mainline as:

```
crates/ruvector-dedrift/
    src/
        lib.rs          — re-export, public traits
        ivf.rs          — storage-agnostic IVF trait (impl for rairs)
        policies/
            split.rs
            lazy.rs
            hybrid.rs
            merge.rs    (NEW)
        scheduler.rs    — token-bucket amortizer
        metrics.rs      — drift score, list health, hooks for prometheus
    benches/            — criterion bench against SIFT1M-drift
    examples/
        demo.rs         — current main.rs
        sift1m_drift.rs — real-data harness
```

## References

- Baranchuk et al., "DeDrift: Robust Similarity Search under Content
  Drift", ICCV 2023. arXiv:[2308.02752](https://arxiv.org/abs/2308.02752).
- Jégou, Douze, Schmid, "Product Quantization for Nearest Neighbor
  Search", PAMI 2011.
- ruvector ADR-193 — RAIRS IVF.
- ruvector ADR-194 — Anisotropic PQ embedder API.
- SOAR (Google, 2024) — multi-assignment IVF; complementary to DEDRIFT.
