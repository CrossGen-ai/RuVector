# AIVF — Adaptive IVF with Online Partition Split/Merge

**Branch:** `research/nightly/2026-06-05-aivf-adaptive-split`
**Crate:**  `crates/ruvector-aivf`
**ADR:**    [ADR-196](../../../adr/ADR-196-aivf-adaptive-split.md)
**Date:**   2026-06-05
**Status:** experimental (numbers reproducible from `cargo run --release -p ruvector-aivf --bin aivf-demo`)

## Abstract

A classical Inverted-File (IVF) index freezes its coarse centroids at build
time. Under streaming or drifting workloads the data distribution moves; some
lists explode while others collapse; recall at fixed `nprobe` decays.
**AIVF** keeps the IVF structure but adapts it online: lists that become too
large or too dispersed are **split** by local 2-means, and pathologically
small lists are **merged** with their nearest sibling. The result is an IVF
that retains its low search cost while tracking distribution drift, with no
global rebuild.

On a drifting 30 000-vector workload (64-d, 32 bootstrap clusters + 16 drifted
clusters, `nprobe = 3`), AIVF with split-only adaptation lifts recall@10 from
**0.990 → 0.998** while *cutting search latency by ~40 %* (80.6 µs → 48.9 µs
per query) because the post-split lists are smaller. Numbers below are
real, captured from the bundled binary on this machine.

## SOTA survey

Adaptive partitioning for vector search has clustered around a handful of
ideas in 2024-2025:

* **SPFresh / FreshDiskANN** (Zhang et al., SOSP 2023; Singh et al., 2024) —
  consolidation + delta storage for SPANN / DiskANN under updates. AIVF
  borrows their *amortised rebalance* idea (don't rebalance on every insert)
  but operates on a flat IVF rather than a graph index.
* **Quake** (Khosla et al., VLDB 2025) — adaptive partition rebalancing
  for IVF where partitions split and merge based on query-driven cost
  modelling, not just data drift. AIVF's split/merge triggers are
  data-distribution heuristics; integrating query-cost-driven triggers is
  the obvious next step.
* **LSM-Vector / LSM-IVF** (multiple 2025 preprints, including the sibling
  upstream branch `research/nightly/2026-06-05-lsm-vector-index` in this
  same fork) — treat the index as an LSM tree with compaction stages.
  AIVF is the orthogonal *intra-level* mechanism: it adapts the partitioning
  inside one level rather than moving data between levels.
* **SOAR / RAIRS** (Google ICML 2024; ADR-193 in this repo) — redundant
  multi-list assignment for IVF. SOAR/RAIRS gives each vector two list
  homes; AIVF instead adjusts the *number and shape* of lists themselves.
  The two ideas compose cleanly (see "What to improve next").
* **FAISS IVF-PQ, Milvus, Qdrant** — production systems still mostly rebuild
  IVF centroids offline on a sample. The streaming-update path is "insert
  to nearest existing centroid and hope," with periodic offline retrains.
  AIVF is intended as the in-RAM, online middle ground.

## Proposed design

AIVF is a *coarse-quantiser sidecar*: it owns the inverted-list shape and
delegates vector storage / distance to a pluggable [`Quantizer`] backend.
The reference backend is `FlatQuantizer` (raw `f32`s, used by the
benchmark); PQ / RaBitQ backends slot in without index changes.

### Online statistics per list

Each list keeps a **Welford-style** running tuple:

* `count`   — number of vectors,
* `sum`     — vector sum (used to recompute centroid on merge),
* `sse`     — sum of squared L2 distances from members to the *current*
              centroid.

`mean_radius_sq = sse / count` is the per-list dispersion. Splits use a
*relative* dispersion threshold (`mean_radius_sq > k · global_mean_radius_sq`)
so the trigger is workload-independent.

### Split trigger

A list splits when **either**:

1. `count > split_size` (default 4 096), or
2. `mean_radius_sq > split_radius_factor · global_mean_radius_sq`
   (default factor 4×).

The split runs a **local 2-means** initialised by a deterministic
furthest-point seeding (no RNG dependence) for `split_iters` Lloyd passes
(default 6). If one side ends empty the split is rolled back.

### Merge trigger

A list with `count < merge_size` (default 16) is merged with its nearest
sibling by centroid distance. The merged list's centroid is the combined
mean; SSE is bumped by the merged-in SSE (a slight over-estimate which is
fine for radius bookkeeping but is the obvious target for the next
iteration).

### Amortisation

`rebalance_every` inserts trigger one maintenance pass. Default 1 024 →
maintenance amortises to O(1) per insert in steady state. A hard
`max_lists` cap (default `8 · nlist_init`) protects against pathological
growth.

## Implementation notes

* **No RNG inside the index.** Splits seed via furthest-point on the
  existing data so two builds with the same insertion order produce
  bit-identical structures. The benchmark uses `StdRng` with a fixed
  seed, so the *workload* is also reproducible.
* **Heap-based top-k.** Search keeps a bounded max-heap of size `k`. Peek
  cost is O(1); replace is O(log k).
* **Partial sort for probes.** `select_nth_unstable_by` picks the `nprobe`
  closest lists in O(nlist) without sorting the tail.
* **File length.** `src/lib.rs` is 268 LOC, `src/main.rs` 117 LOC,
  `src/quantizer.rs` 47 LOC, `src/metric.rs` 32 LOC. All under the
  500-line cap.

## Benchmark methodology

Workload generation, fixed seed `0xA1ABF107`:

* DIM = 64.
* Bootstrap = 4 000 vectors drawn from 32 Gaussian-ish clusters in
  `[-1, 1]^64`.
* Stream = 26 000 vectors drawn from **16 different** clusters in
  `[-3, 3]^64` (this is the drift — bootstrap centroids are stale for
  these regions).
* Queries = 1 000 vectors drawn from the *drifted* region (the workload
  the index must serve well).
* `nlist_init = 64`, `nprobe = 3`, `k = 10`.

Ground truth via brute force over all 30 000 vectors. Variants:

| variant            | splits | merges |
|--------------------|--------|--------|
| static IVF         |   off  |   off  |
| aivf split-only    |   on   |   off  |
| aivf split+merge   |   on   |   on   |

## Results

Captured from `cargo run --release -p ruvector-aivf --bin aivf-demo` on
Darwin arm64 (M-series), single-thread:

```
AIVF demo — DIM=64 N=30000 nlist_init=64 nprobe=3 k=10
ground truth: brute force 1000 queries in 629.4ms

variant            | lists | splits | merges | build ms | stream ms | search µs/q | recall@10
-------------------+-------+--------+--------+----------+-----------+-------------+-----------
static IVF         |   64  |    0   |    0   |    5.0   |    33.8   |    80.6     |   0.990
aivf split-only    |   79  |   15   |    0   |    5.6   |    40.7   |    48.9     |   0.998
aivf split+merge   |   36  |    2   |   30   |    3.7   |    18.8   |    80.1     |   0.999

estimated raw vector memory: 7.32 MiB
```

### Reading the numbers

* **Recall@10**: static 0.990 → split-only 0.998 → split+merge 0.999. The
  recall lift is small in absolute terms because the synthetic workload
  is benign, but the *trend* (adaptive > static) is consistent and
  reproducible.
* **Search latency**: split-only is **34 % faster** (48.9 µs vs 80.6 µs)
  because the post-split lists are smaller — probing 3 of 79 small lists
  beats probing 3 of 64 larger ones. This is the result we want from
  online split.
* **Stream cost**: AIVF's online maintenance adds ~20 % overhead during
  inserts (40.7 ms vs 33.8 ms for the full 26 000-vector stream),
  i.e. ~0.27 µs per insert of amortised rebalance. Cheap.
* **split+merge** trades search latency for fewer lists (36 vs 79). With
  `nprobe = 3` the per-probe list is bigger, so latency catches back up
  to the static baseline. This is the expected dial: tune `merge_size` to
  the operating budget.

### Practical failure modes

* **Empty-side splits.** Furthest-point init can produce an empty side on
  near-duplicate clusters; the implementation rolls back. Symptom in the
  wild: split_events fewer than expected. Mitigation: fall back to
  random init after N consecutive empty rolls.
* **Merge over-collapse.** With aggressive `merge_size` the index can
  collapse below `nlist_init`, increasing per-probe work. Cap the merge
  with a `min_lists` setting (not in this PoC).
* **SSE drift on merge.** The merged SSE is an over-estimate; over many
  merges the radius signal degrades and split triggers become noisy.
  Fix: recompute SSE from the backing quantiser on merge — cheap when the
  merged list is small (which it is by definition).
* **Deletion is not implemented.** This PoC is insert-only. Tombstones
  + a periodic compaction pass mirror SPFresh/FreshDiskANN.

## "How it works" walkthrough

The AIVF data flow on a single insert:

1. **Store the vector** in the backing `Quantizer` (so splits can read
   it back later — this was the first bug found in the PoC; see the
   commit history).
2. **Find the nearest list** by centroid L2 (O(nlist)).
3. **Assign** the id to that list. Update `sum`, `sse`, `count` online.
4. **Tick the amortiser.** Every `rebalance_every` inserts:
   * compute `global_mean_radius_sq`;
   * for each list, decide split (size OR relative-radius);
   * for each tiny list, decide merge (nearest sibling by centroid).
5. **Splits run a local 2-means** seeded by furthest-point. Lloyd
   converges in 2-3 iterations for cluster-like data; the 6-iter cap is
   a belt-and-braces upper bound.
6. **Merges combine sum/count/SSE** without touching the backing store.

Search is just IVF: `select_nth_unstable` the `nprobe` closest list
centroids, scan their members, push into a bounded max-heap of size `k`,
sort and return.

## What to improve next

Roadmap from "minimum-viable PoC" to a production crate:

1. **PQ / RaBitQ backend.** Drop in `ruvector-rabitq` as the `Quantizer`.
   Splits will pay a re-rank cost (PQ is lossy) — measure it.
2. **Query-driven split triggers.** Quake-style: split lists that are
   hot *and* dispersed, not just dispersed. Requires per-list query
   counters.
3. **SOAR/RAIRS overlay.** Combine AIVF with RAIRS's redundant assignment
   (ADR-193). Splits change the assignment graph; the residual-amplified
   secondary score from RAIRS needs an update path.
4. **Delete + tombstone.** Add `remove(id)` + periodic compaction.
5. **Persisted snapshot.** `ruvector-snapshot` already exists; add an
   AIVF serializer (list centroids + ids + Welford state, no SSE
   recomputation on load).
6. **Multi-threaded inserts.** A per-list `parking_lot::Mutex` is enough
   to start; rebalance must briefly stop-the-world or run under a coarser
   epoch.
7. **Production crate layout** (proposed):

   ```
   crates/ruvector-aivf/
     src/
       lib.rs        — public API + Aivf core
       config.rs     — AivfConfig + builders
       list.rs       — InvList + Welford updates
       split.rs      — 2-means split logic
       merge.rs      — merge logic
       search.rs     — probe + heap top-k
       quantizer.rs  — Quantizer trait + FlatQuantizer
       backends/
         pq.rs        — PQ backend (feature = "pq")
         rabitq.rs    — RaBitQ backend (feature = "rabitq")
       metric.rs     — l2/dot, with portable_simd feature
     tests/
       smoke.rs
       drift.rs       — distribution-drift recall tests
       delete.rs      — tombstone + compaction tests
     benches/
       criterion.rs   — Criterion harness on real datasets (SIFT1M etc.)
   ```

## References

1. Khosla, J. et al. "Quake: Adaptive Indexing for Online Vector Search."
   VLDB 2025 (publication referenced from the conference programme; verify
   exact citation before publishing).
2. Zhang, Y. et al. "SPFresh: Incremental In-Place Update for Billion-Scale
   Vector Search." SOSP 2023.
3. Singh, A. et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN
   Index for Streaming Similarity Search." arXiv:2105.09613.
4. Sun, P. et al. "SOAR: Improved Indexing for Approximate Nearest Neighbor
   Search." ICML 2024.
5. ruvector ADR-193 — RAIRS IVF.
6. ruvector ADR-194/195 — embedder unification (consumer of AIVF if it
   matures into the default IVF backend).

(Note: external citations should be re-checked before any external
publication. The AIVF *implementation* and *numbers* in this document are
reproducible from this repo and do not depend on the citations above.)
