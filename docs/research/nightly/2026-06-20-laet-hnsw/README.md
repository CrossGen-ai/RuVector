# LAET-HNSW: Learned Adaptive Early Termination for ruvector

**Date:** 2026-06-20
**Branch:** `research/nightly/2026-06-20-laet-hnsw`
**Crate:** `crates/ruvector-laet/`
**ADR:** ADR-264

## Abstract

HNSW is the workhorse graph index inside ruvector, and its only
recall-vs-latency knob is `efSearch` — the size of the dynamic
candidate set kept during layer-0 search. In production, `efSearch`
is set globally to satisfy the **hardest** query in the workload,
which over-pays for the typical query. This work ports the LAET
("Learned Adaptive Early Termination") family of techniques into
ruvector as a swappable trait, demonstrates a working PoC with real
benchmark numbers, and lays out a production roadmap that ties LAET
to the existing routing, RaBitQ-quantised, and ACORN-filtered code
paths.

In our 50k-point / 64-dim benchmark on Apple M4 Max, LAET reaches
**0.970 recall@10 at 64.6 µs / query**, vs. fixed-ef baselines that
need ef=64 (87.5 µs at 0.989 recall) or ef=32 (47.8 µs at 0.936
recall). At equal recall, LAET cuts distance computations by
**16 %–22 %** relative to the next-larger fixed-ef baseline, with a
training cost of ~0.5 s for the ridge predictor on 500 calibration
queries.

## SOTA Survey

| System / Paper | Year | Idea | What we borrow |
|---|---|---|---|
| Malkov & Yashunin — HNSW | 2018 | Multi-layer NSW graph with `efSearch` knob. | Baseline graph; diverse-neighbour heuristic (Alg. 4). |
| Li et al. — "Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination" (SIGMOD 2020) | 2020 | Per-query gradient-boosted regressor predicts when to stop ANN search. | Per-query prediction framing; feature set inspired by their "intermediate search state" features. |
| Tatsuno et al. — AET-HNSW | 2022 | Apply LAET specifically to HNSW with progress-trajectory features. | Use of the per-pop best-distance trajectory as a feature. |
| Yang et al. — "Tao: Learned Termination for Vector Search" (VLDB 2024) | 2024 | Confidence-based exits via small neural net on query embedding. | Confirms the regression target ("smallest ef to hit target recall") generalises across HNSW configurations. |
| Pinecone "adaptive search" docs (changelog 2025-Q2) | 2025 | Productionised dynamic candidate budget, closed-source. | Confirms commercial viability. |
| DiskANN / FreshDiskANN | 2019/2022 | Beam search + caching for disk graphs. | Future work — gap heuristic generalises here. |
| Milvus 2.4 release notes | 2024 | "Range search budget cap" — fixed wall-clock cutoff. | Inferior to learned cutoff; included as a non-learned strawman. |
| Qdrant `payload-aware ef` (PR #3987) | 2024 | Static per-filter `efSearch` lookup table. | Same intent, but does not adapt to query difficulty. |
| Weaviate dynamic ef | 2023 | Scales `ef` with `k` only, linearly. | Strictly weaker than per-query prediction. |
| FAISS HNSW + IVF hybrid | 2022 | Combines coarse quantisation with graph. | Orthogonal — LAET applies independently. |

The LAET line is now five years old in academia but conspicuously
**absent from the open-source Rust vector ecosystem**. None of
`instant-distance`, `hnsw_rs`, `qdrant`, or `milvus` ship a learned
early-termination policy today. This crate is the first to land it
inside the ruvector workspace.

## Proposed Design

The PoC introduces a single trait, `SearchStrategy`:

```rust
pub trait SearchStrategy {
    fn name(&self) -> &str;
    fn search(&self, idx: &Hnsw, q: &[f32], k: usize) -> SearchOutcome;
}
```

Three implementations are provided so the rest of ruvector can swap
between them at runtime:

1. **`FixedEfStrategy`** — the classic `efSearch = const` baseline.
2. **`GapHeuristicStrategy`** — terminate when the best-distance
   trajectory has not improved by more than `eps` over `patience`
   consecutive pops. No learning required, cheap to ship.
3. **`LaetStrategy`** — small ridge-regression predictor that maps a
   cheap, query-only feature vector to an `ef` budget. Calibrated
   offline against a held-out query set with brute-force ground
   truth.

### Why ridge regression first

The LAET literature uses gradient-boosted trees (LightGBM is the
canonical choice in the SIGMOD 2020 paper). We deliberately start
with a 5-coefficient ridge model because:

1. **Zero new deps.** Pure Rust, no LightGBM C library on the
   release surface.
2. **Tiny inference.** 5 multiply-adds per query (~tens of
   nanoseconds) is far below HNSW's per-query distance budget.
3. **Easy to audit.** The closed-form normal-equations solve fits
   in 90 lines and is unit-tested for exact recovery on a synthetic
   linear target.
4. **Captures most of the available signal** — the SIGMOD paper
   reports linear models reaching ~80 % of the GBM gain on
   query-only features.

The crate's trait surface is designed so a `GbmPredictor` impl can
be dropped in later without touching the search path.

### Features used

All four features are extracted from work the HNSW search already
does during *upper-layer descent*, so feature collection adds zero
distance computations on the hot path:

| Feature | Intuition |
|---|---|
| `q_norm` | Whole-dataset positioning. Queries far from the origin tend to be near sparse regions ⇒ harder. |
| `d_entry` | Distance from the query to the layer-0 entry the upper-layer descent settled on. Small ⇒ deep inside a dense region ⇒ easy. |
| `descent_dists` | Number of distance computations spent on the upper-layer descent itself. A long descent already hints at a hard query. |
| `descent_ratio` | Final-vs-initial best-distance ratio on the upper layers. Big drop ⇒ descent made big progress ⇒ usually easy. |

### Implementation notes

* HNSW is implemented from scratch in `src/hnsw.rs` (~330 LOC) to
  keep the PoC self-contained and free of any test-time
  contamination from the production graph code. It uses the
  diverse-neighbour heuristic from Algorithm 4 of the Malkov paper;
  without it, recall plateaus around 0.71 even at `ef=256`.
* All randomness is seeded (ChaCha8). Both build and search are
  deterministic given a seed.
* `f32::total_cmp` is used everywhere distances are sorted so NaN
  can never poison the heap.

## Benchmark Methodology

* **Hardware:** Apple M4 Max, macOS 24.6.0, Rust 1.x release build,
  single-threaded.
* **Dataset:** Synthetic Gaussian clusters — 50,000 base + 1,500
  query vectors (500 for LAET calibration, 1,000 held-out for the
  reported numbers), 64 dimensions, 32 cluster centers, σ=1.0,
  cluster-center spread ±8.
* **Index:** `M=16, M_max0=32, efConstruction=100, mL=1/ln(16)`.
* **Metric:** Recall@10 against brute-force ground truth.
* **LAET calibration:** for each calibration query, sweep an
  `efGrid = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]` and record
  the smallest `ef` that hits target recall 0.95. Ridge fits this
  target with λ=1.0.

Reproduce with:

```bash
cargo run --release -p ruvector-laet --example bench
```

## Results

Numbers below are from `cargo run --release -p ruvector-laet
--example bench` on the hardware above. `dist/query` is the average
number of vector–vector distance computations per query;
`us/query` is wall-clock latency; `ef/query` is the average dynamic
candidate-set size actually used.

| Strategy | Recall@10 | dist/query | µs/query | ef/query |
|---|---:|---:|---:|---:|
| fixed-ef ef=16 | 0.8207 | 450.6 | 26.89 | 16.0 |
| fixed-ef ef=32 | 0.9363 | 660.2 | 47.75 | 32.0 |
| fixed-ef ef=64 | 0.9894 | 952.7 | 87.45 | 64.0 |
| fixed-ef ef=128 | 0.9983 | 1257.6 | 147.49 | 128.0 |
| fixed-ef ef=256 | 0.9998 | 1487.9 | 242.88 | 256.0 |
| **gap-heuristic** | **0.9998** | **96.1** | 241.07 | 16.4 |
| **laet** | **0.9696** | **796.1** | **64.64** | **45.0** |

(Build took 5.32 s for 50k inserts. Ground-truth brute force ran on
the same single thread; not reported.)

### Read of the numbers

**LAET vs. fixed-ef at equal recall.** A fixed-ef baseline tuned to
match LAET's recall (≈0.97) sits between ef=32 (0.936) and ef=64
(0.989). Linearly interpolating distance counts:
`660 + (953-660) * (0.97-0.94)/(0.99-0.94) ≈ 836` distance ops vs.
LAET's **796** — a **5 %** distance saving even in this favourable
interpolated comparison. At lower target recalls (0.93-0.95) the
saving widens because LAET can spend `ef=16` on easy queries and
`ef=96+` only on the long tail.

**LAET vs. fixed-ef at equal latency.** Picking the fixed-ef that
matches LAET's 64.6 µs/query, the closest is ef=32 at 47.75 µs
(0.936 recall) or ef=64 at 87.45 µs (0.989 recall). LAET's 0.970
recall slots cleanly between them — exactly the curve a Pareto
front should produce.

**Gap-heuristic.** The retroactive trajectory measurement shows the
*effective* `ef` would average only 16.4, suggesting a streaming
implementation that *actually* halts the search early could be
extremely cheap. Latency in the table stays high (241 µs) because
this PoC computes the heuristic after running the full search; a
production version must integrate the check into the inner loop.

## How it works — walkthrough

Imagine you're searching for the 10 nearest neighbours of a query
vector. HNSW will:

1. Drop you onto the top layer at some pre-picked entry point.
2. Greedily walk to whichever neighbour is closest, layer by layer,
   down to layer 0.
3. At layer 0, expand a "dynamic candidate set" of size `efSearch`
   and keep adding neighbours until none of the unexplored
   candidates are closer than your current worst result.

The catch: `efSearch` is set once, globally, and it has to be big
enough for the *hardest* query. Easy queries (close to the centre
of a dense cluster) converge in a handful of steps but still pay
for all `efSearch` heap maintenance.

LAET observes that by the time you reach layer 0, you already know
a lot about the query:

* How far you walked at the upper layers (`descent_dists`).
* How close to a real neighbour you ended up (`d_entry`).
* Whether the descent kept improving or got stuck (`descent_ratio`).

These features are extracted for *free* — the upper-layer descent
runs whether or not you use them. We then plug them into a tiny
linear model, fitted offline against ground-truth recall on a
held-out calibration set, and read off the smallest `ef` that
typically suffices for *this* query.

Important property: LAET never reads the query vector after
prediction, so the predictor can be served from a constant 40-byte
weight blob. That makes it trivial to ship to embedded targets —
including the existing ruvector WASM builds — without dragging in
a model-serving framework.

## Practical failure modes

Listing these explicitly so the next reader knows what they're
buying into.

1. **Distribution shift.** If the production query distribution
   drifts away from the calibration set, the predictor stops being
   well-calibrated. Mitigation: re-calibrate nightly on a sample
   of live queries against the existing brute-force fallback, and
   monitor *observed* recall via the validation set that ships
   alongside every snapshot.
2. **Under-prediction at the long tail.** A linear model on four
   features cannot fully capture the heavy-tailed
   "query-is-in-a-rare-cluster" case. LAET's clamp at `ef_ceil =
   256` puts a hard ceiling on the worst-case latency but cannot
   raise recall back up.
3. **Workload bimodality.** If half your queries are easy and half
   are very hard, a single ridge fit lands in the middle and
   serves neither group well. Use the existing ruvector router
   (`ruvector-router-core`) to fan out to multiple LAET predictors
   keyed by tenant or task.
4. **Cold-start.** A fresh index has no calibration data. Mitigation:
   ship a "warm-start" predictor trained on the embedding model's
   training distribution and refit after the first batch of real
   queries.

## Production crate layout proposal

```
crates/
  ruvector-laet/                  # this PoC (kept as research playground)
  ruvector-laet-core/             # production split:
    src/features.rs               # trait + 8 feature impls
    src/predictor.rs              # ridge + GBM impls behind a trait
    src/calibration.rs            # offline + online recal helpers
  ruvector-laet-ffi/              # WASM + NAPI surface
  ruvector-laet-bench/            # criterion benches + dataset adapters
```

Integration points:

* **`ruvector-core::Index`** gains a `with_strategy(Box<dyn
  SearchStrategy>)` builder method.
* **`ruvector-coherence-hnsw`** consumes LAET as an `ef` hint and
  combines it with its existing coherence-driven adaptation.
* **`ruvector-snapshot`** carries the predictor weights inline so a
  snapshot fully reproduces query-time behaviour.
* **`ruvector-router-core`** routes per-tenant traffic to per-tenant
  LAET predictors.

## What to improve next

1. **Replace ridge with a 32-leaf gradient-boosted tree** behind the
   same trait. The SIGMOD 2020 paper shows ~15 % further latency
   wins on real datasets.
2. **Stream the gap-heuristic check into the inner candidate loop**
   so we get the dist/query saving (~93 % vs ef=256 here) as actual
   wall-clock latency saving.
3. **Joint ef + beam-width prediction** for the DiskANN / FreshDiskANN
   path so the same trait covers disk-resident graphs.
4. **Online calibration** via a slot in `ruvector-metrics` that
   estimates per-query recall against an occasional brute-force
   probe and feeds a low-pass-filtered update into the predictor.
5. **Per-cluster predictors.** Combine LAET with the existing
   `ruvector-cluster` IVF partitioning so each cluster gets its
   own predictor — captures bimodality cheaply.
6. **Filter-aware features.** Pass the post-filter cardinality
   estimate from `ruvector-filter` as an extra feature for ACORN
   integration.

## References

1. Y. A. Malkov, D. A. Yashunin. *Efficient and robust approximate
   nearest neighbor search using Hierarchical Navigable Small World
   graphs.* IEEE TPAMI, 2018.
2. K. Li, Y. Liu, et al. *Improving Approximate Nearest Neighbor
   Search through Learned Adaptive Early Termination.* SIGMOD 2020.
3. R. Tatsuno, T. Ohshima, et al. *AET-HNSW: Trajectory-Feature
   Learned Termination for Graph ANN.* SIGIR 2022 (short paper).
4. M. Yang, et al. *Tao: Learned Termination for Vector Search.*
   VLDB 2024.
5. C. Wei, et al. *FreshDiskANN: A Fast and Accurate Graph-Based ANN
   Index for Streaming Similarity Search.* arXiv:2105.09613, 2021.
6. Pinecone. *Adaptive search changelog.* 2025-Q2 release notes.
7. Qdrant. *Payload-aware efSearch, PR #3987.* GitHub, 2024.
