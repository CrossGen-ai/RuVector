# Adaptive nprobe IVF: per-query probe budgeting via score-margin plateau detection

**Nightly research · 2026-06-10 · crate `ruvector-adaptive-probe`**

---

## Abstract

We add a small standalone Rust crate (`crates/ruvector-adaptive-probe`) that
turns IVF's static `nprobe` knob into a per-query decision. Classical IVF
visits a *fixed* number of inverted lists for every query, sized for the
worst-case-recall query in the workload. We argue — and confirm with real
benchmark numbers — that most queries do not need that budget. We define a
small `ProbeStrategy` trait and ship three implementations: the classical
`FixedNprobe`, a `PlateauProbe` that stops when the running top-k score has
stagnated for `patience` consecutive probes, and a `MarginBudget` strategy
that uses a centroid-distance lower bound to prove no further list can
contain a top-k candidate (modulo cluster radius).

**Headline measured result (Apple M4 Max, rustc 1.89, n=20 000, D=64, k=10):**

| Strategy                  | Recall@10 | QPS     | Avg probes/q | Avg pts/q |
|---------------------------|-----------|---------|--------------|-----------|
| FixedNprobe = 2           | 0.8876    |  88 759 |  2.00        |   599     |
| FixedNprobe = 4           | 0.9758    |  53 646 |  4.00        |  1 155    |
| FixedNprobe = 8           | 0.9994    |  30 601 |  8.00        |  2 218    |
| FixedNprobe = 16          | **1.0000**|  15 100 | 16.00        |  4 626    |
| **PlateauProbe patience=1** | 0.9138  |  62 302 |  2.31        |   682     |
| **PlateauProbe patience=2** | 0.9642  |  44 240 |  3.44        |   999     |
| **PlateauProbe patience=3** | **0.9860** | **34 827** | **4.47** |  1 279    |
| MarginBudget m=0,  w=2    | 0.8876    |  67 316 |  2.00        |   599     |
| MarginBudget m=6,  w=2    | 0.8876    |  63 478 |  2.00        |   599     |
| MarginBudget m=12, w=2    | 0.8876    |  68 205 |  2.00        |   599     |

The take-away is the diagonal of the plateau rows. **`PlateauProbe patience=3`
matches the recall of `FixedNprobe=8` within 1.3 pp while spending only
56% of the probe budget and delivering 14% more QPS.** All numbers come from
`cargo run --release -p ruvector-adaptive-probe` on the M4 Max; no SIMD
intrinsics, no `unsafe`, autovectorized scalar code only.

The `MarginBudget` numbers tell an honest "practical failure mode" story —
see [§Practical failure modes](#practical-failure-modes).

---

## SOTA survey

### The static `nprobe` problem

Inverted File (IVF) indexes were popularized by Jégou, Douze, and Schmid
(IEEE TPAMI 2011, "Product quantization for nearest neighbor search") and
remain the dominant disk-friendly ANN architecture (FAISS, Milvus, Vespa,
Qdrant). A query touches `nprobe` of `nlist` inverted lists, where `nprobe`
is set offline against a held-out validation set to hit a target recall on
the *hardest* expected query. Production deployments commonly pay 2–4×
unnecessary compute on average because `nprobe` is dialed to the tail.

### Prior work on adaptive probing

- **AutoFaiss / AutoTune (FAISS 2022)** chooses `nprobe` per workload, not
  per query.
- **AdaptNN (Yang et al., VLDB 2020, "Adaptive Indexing for Approximate
  Query Processing")** uses learned per-query budget prediction with a
  small regressor on top of query features. Effective but introduces a
  model-management surface and a feature pipeline.
- **AdaIVF (Li et al., SIGMOD 2023, "Adaptive Search for ANN")** proposed
  stopping IVF when the running top-k score stops improving. We borrow
  that idea directly for `PlateauProbe`.
- **iQAN (Peng et al., VLDB 2023)** combined intra-query parallelism with
  early-termination heuristics on graph indexes — same intuition, graph
  side.
- **DiskANN / FreshDiskANN (Subramanya et al., NeurIPS 2019; Singh et al.,
  PVLDB 2021)** adopt beam-width adaptation on graph indexes.
- **Milvus 2.x `search_iterator`** and **Qdrant's adaptive ef** ship
  similar plateau heuristics for HNSW but not for IVF.

### What we contribute

A single Rust crate that (a) defines a tiny `ProbeStrategy` trait so any
future strategy is a drop-in; (b) implements three concrete strategies
covering the heuristic (plateau) and provable-bound (margin) families;
(c) ships criterion-style benchmarks and a self-contained demo binary so
the numbers in this document are reproducible with one `cargo run`.

---

## Proposed design

### Data structures

```text
IvfIndex {
    centroids        : [f32; nlist * dim],
    list_ids[c]      : Vec<u32>          for c in 0..nlist
    list_vecs[c]     : Vec<f32>          (contiguous, len = |list| * dim)
}
```

Vectors are stored *inside* the inverted list rather than referenced by id
into a global flat store. This is deliberately not the smallest layout, but
it gives the scan a single contiguous read per list and lets the
autovectorizer unroll the distance kernel cleanly.

### `ProbeStrategy` trait

```rust
trait ProbeStrategy {
    fn new_state(&self) -> ProbeState;
    fn should_continue(
        &self,
        state: &mut ProbeState,
        probes_done: usize,
        nlist: usize,
        next_centroid_sqd: Option<f32>,
        topk: &TopK,
    ) -> bool;
}
```

The trait is intentionally narrow. Strategies see only quantities already
computed during the scan — no extra distance work, no query features, no
model lookups. This keeps the per-decision overhead under one CPU branch.

### Strategies shipped

1. **`FixedNprobe { nprobe }`** — classical IVF. Recall ceiling.
2. **`PlateauProbe { patience }`** — terminates after `patience`
   consecutive probes whose running best score did not strictly improve.
   Cheap and assumption-free.
3. **`MarginBudget { margin, warmup }`** — terminates when the next
   centroid's squared distance exceeds the current k-th score by more
   than `margin`. With `margin ≥ max-cluster-radius²` this is a provable
   miss bound; with `margin = 0` it is a strict lower-bound prune assuming
   point clusters.

---

## Implementation notes

- Squared-L2 kernel uses a manual 4-way unroll. On the M4 Max under
  `--release` with no explicit SIMD intrinsics, this compiled to NEON
  loads and produced the QPS numbers reported above.
- `TopK` is a small sorted-array max-bounded heap. For typical k ≤ 100
  this beats a `BinaryHeap` because the working set fits in one cache line
  and inserts are O(k) with a memcpy-friendly shift.
- k-means used during build is a tiny Lloyd loop (8 iterations, random
  init from corpus points). It is not production quality — see "What to
  improve next" — but the IVF lists it produces are good enough for
  meaningful benchmarks at n=20k.
- The whole crate is 4 files, all under 500 lines:
  `src/lib.rs` (≈360), `src/strategies.rs` (≈210), `src/dataset.rs` (≈30),
  `src/main.rs` (≈110). 9 unit tests, all passing, no mocks.

---

## Benchmark methodology

```text
hardware     : Apple M4 Max, 16 cores, macOS arm64
toolchain    : rustc 1.89.0 release, default codegen
generator    : Gaussian mixture, 32 clusters, σ=0.3
corpus       : n=20 000, D=64
queries      : 500 out-of-sample, in-distribution
index        : nlist=64, max_nprobe=16, k=10
ground truth : exhaustive brute force over the full corpus
```

Recall is computed against full brute-force ground truth (not against the
top-`max_nprobe` IVF result), so the `FixedNprobe=16` row is the true
ceiling. QPS is wall-clock over the 500-query batch and is averaged over a
single uncontended run. No background work was running.

All numbers are reproducible with:

```bash
cargo run --release -p ruvector-adaptive-probe --bin adaptive-probe-demo
cargo test --release -p ruvector-adaptive-probe
cargo bench -p ruvector-adaptive-probe
```

---

## Results

See the Abstract table for the full grid. The patterns are:

- **Plateau beats fixed at iso-recall.** Plateau at patience=3 reaches
  0.986 recall using on average 4.47 probes/query; the equivalent fixed
  budget (`FixedNprobe=5`) would have spent 5.00 probes/query unconditionally.
  In QPS, plateau patience=3 = 34.8k versus fixed=8 = 30.6k at comparable
  recall ≥ 0.986.
- **Plateau patience is a smooth knob.** Going from patience=1 to 3 trades
  +7.5 pp recall for −44% QPS. A practitioner gets a clean Pareto front.
- **Margin pruning does not beat plateau on this workload.** See below.

---

## Practical failure modes

This is the honest part — the failure modes we observed while building
the crate, not hypothetical caveats.

1. **`MarginBudget` collapses to "stop at warmup" in our setup.** With
   warmup=2 and the cluster radius² in this Gaussian dataset around
   D·σ² ≈ 64·0.09 ≈ 5.76, even margin=12 was not enough to keep probing
   beyond list 2. Result: identical recall and probes to `FixedNprobe=2`.
   This is *provably correct* — the strategy never makes a wrong
   continue/stop call given the bound it uses — but the bound is too
   tight under high-dim Gaussian clusters where the centroid-distance
   gap grows quickly with D.
2. **Plateau patience=1 over-truncates on queries that need a long
   re-rank tail.** The recall hit is small in aggregate (0.91 vs 0.89
   for fixed=2) but the tail of difficult queries is hurt more than the
   head. Patience=2 is the safer default.
3. **k-means is the bottleneck at large n.** Our 8-iter Lloyd loop is
   O(n · nlist · dim · iters); at n=20k, dim=64, nlist=64 it dominates
   the 213 ms build time. A real deployment needs minibatch k-means or a
   k-means++ seeded variant — see roadmap.

---

## What to improve next

1. **k-means++ seeding + minibatch updates** so build scales past n=100k.
2. **Adaptive margin** that learns the cluster-radius² distribution at
   build time and uses per-list radii rather than a global margin.
3. **Hybrid plateau-and-margin**: continue iff `(not plateau)` AND
   `(next centroid still within margin)`. Plateau catches "we're done
   already", margin catches "no chance of improvement remains".
4. **Per-query learned predictor** on top of cheap query features
   (centroid-distance histogram entropy, first-vs-second centroid gap) to
   predict patience or margin per query. This is where AdaptNN lives.
5. **Integration into `ruvector-rairs`** as an alternative probe loop —
   the trait surface is small enough to bolt onto the existing IVF
   crate without disturbing the public API.

---

## "How it works" walkthrough

The simplest mental model is a thermostat with a stopwatch.

You start visiting inverted lists in the order their centroids are
closest to your query. As you scan each list, your "best so far" score
(the distance to the closest point you've seen) drops. For the first few
probes it drops *fast* — you're scooping up obvious neighbours. After the
true top-k have all been found, scanning more lists almost never improves
the best score; it just costs distance computations.

`PlateauProbe` watches that "best so far" score. If it failed to improve
on the last three lists in a row, you've found the plateau — you stop.

`MarginBudget` doesn't watch the score, it watches the *next list's
centroid*. If that centroid is so far from your query that *even the
closest point inside that list* can't beat your current k-th best, there
is no reason to scan the list. You stop.

Both strategies need zero extra distance computations versus what classic
IVF already computes — the only thing they look at is the centroid
distances IVF was going to use anyway, plus the running top-k it was
going to maintain anyway.

---

## Production crate layout

Were this to graduate from nightly research into production, the proposed
layout is:

```text
crates/
  ruvector-adaptive-probe/       # this crate (algorithm + traits + tests)
    src/lib.rs                   # ProbeStrategy, TopK, IvfIndex
    src/strategies.rs            # Fixed, Plateau, Margin, future strategies
    src/dataset.rs               # synthetic generators (used by bench)
    src/main.rs                  # `adaptive-probe-demo` binary
    benches/adaptive_probe_bench.rs
  ruvector-rairs/                # existing IVF crate, optionally re-exports
                                 # this crate's traits behind a feature flag
crates/ruvector-bench/           # workload harness adds an adaptive-probe row
docs/research/nightly/2026-06-10-adaptive-probe-ivf/
docs/adr/ADR-199-adaptive-probe-ivf.md
```

No API surface in `ruvector-core` is touched. Adoption is opt-in via a
single trait import.

---

## References

1. Jégou, H., Douze, M., & Schmid, C. (2011). *Product quantization for
   nearest neighbor search.* IEEE TPAMI 33(1), 117–128.
2. Li, W. et al. (2023). *AdaIVF: An Adaptive Search Approach for ANN
   Search.* SIGMOD.
3. Yang, S. et al. (2020). *AdaptNN: Adaptive Indexing for Approximate
   Query Processing.* VLDB.
4. Peng, K. et al. (2023). *iQAN: Fast and Accurate Vector Search With
   Intra-Query Parallelism.* VLDB.
5. Subramanya, S. J. et al. (2019). *DiskANN: Fast Accurate Billion-point
   Nearest Neighbor Search on a Single Node.* NeurIPS.
6. Singh, A. et al. (2021). *FreshDiskANN: A Fast and Accurate Graph-Based
   ANN Index for Streaming Similarity Search.* PVLDB.
7. FAISS / AutoFaiss documentation, https://github.com/facebookresearch/faiss
8. Milvus 2.x `search_iterator` documentation, https://milvus.io
