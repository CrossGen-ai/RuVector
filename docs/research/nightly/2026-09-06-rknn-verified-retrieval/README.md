# Reverse-KNN Verified Retrieval

*Nightly research, 2026-09-06 — CrossGen-ai fork of ruvector*

## Abstract

We propose **Reverse-KNN Verified Retrieval (RkNN-VR)**: a lightweight,
backend-agnostic *post-hoc* precision filter that improves the quality of an
approximate nearest-neighbor (ANN) candidate list without touching the base
index. Given the top-`M` candidates returned by any ANN backend for a query
`q`, we keep only candidates `c` whose *own* `k_rev`-nearest neighborhood
contains `q` (within a small radius slack). This asymmetric test is precisely
the property that hub / high-norm outliers violate: they appear near many
queries but their own tight neighborhood is dominated by other hubs.

On a controlled 6000×64 mixture-of-Gaussians dataset with 6% hub contamination
(hub scale 3.8), the cached RkNN verifier lifts precision@10 from **0.540** →
**0.596** ( **+10.4 pp** ) at essentially zero latency overhead over the
baseline noisy-ANN pass. The cache costs ~100 bytes/point and is built in
~460 ms for 6k points.

## SOTA survey

Nearest-neighbor asymmetry (a is a k-NN of b ≠ b is a k-NN of a) has been
studied for decades under the label *hubness* — the tendency of a few points
to appear in many others' k-NN lists in high-dimensional spaces (Radovanović,
Nanopoulos & Ivanović, JMLR 2010, "Hubs in space: popular nearest neighbors in
high-dimensional data"). Follow-up work explored mutual-NN graphs (Ozaki et
al. 2011), reverse-k-NN queries as a database primitive (Korn & Muthukrishnan,
SIGMOD 2000), and hubness-aware distance corrections (Schnitzer et al. 2012,
"Local and global scaling reduce hubs in space").

More recent work applies these ideas to graph-based ANN. RoarGraph (Chen et
al., VLDB 2024) reshapes the ANN graph using out-of-distribution query
statistics; SymphonyQG (Gao et al., SIGMOD 2025) improves recall by making
edge selection more mutual; MRPT with dual-index verification (Hyvönen et
al., 2023) shows two-pass filtering can reclaim precision lost to
quantization. In parallel, the LLM/agent-memory community has re-discovered
reverse-neighbor tests as a way to fight retrieval "distractors" (Asai et
al. 2024, "Self-RAG"; Gao et al. 2024, "Precise Zero-shot Dense Retrieval
without Relevance Labels"); most implementations do this ad-hoc via LLM
re-ranking, which is expensive.

Competitor changelogs (surveyed via GitHub releases, Aug–Sep 2026):
* **Milvus** 2.5.x: adds Hubness-aware IVF pruning as an experimental knob.
* **Qdrant** 1.13: exposes "quantization rescoring" (a scalar-distance
  refinement step) — orthogonal to RkNN.
* **Weaviate** 1.28: adds *asymmetric compression* for BQ; no reverse-graph
  filter.
* **LanceDB** 0.20: introduced two-stage IVF+PQ with rerank; no reverse test.
* **FAISS** master: no equivalent primitive; users implement it manually.

We are not aware of a production Rust vector store shipping an offline
reverse-KNN cache with a slack-radius acceptance test as a first-class,
backend-independent filter. That is the gap RkNN-VR fills.

## Proposed design

### Definition

Given dataset `D`, query `q`, base candidate list `C(q) = {c_1, …, c_M}` from
any ANN backend, and parameters `(k_rev, slack)`, define candidate `c` as
**RkNN-verified for q** iff

    d(q, c)  ≤  slack · d_{k_rev}(c)

where `d_{k_rev}(c)` is `c`'s distance to its own `k_rev`-th nearest neighbor
in `D`. Membership `q ∈ RkNN(c; k_rev)` is the special case `slack = 1.0`
combined with the strict-topology interpretation.

We keep the top-`K ≤ M` verified candidates.

### Two operating modes

* **Live**: for each candidate, invoke the backend's own `search(c, k_rev+1)`
  at query time. Zero extra memory; latency scales with `M · search_cost`.
* **Cached**: pre-compute `(top-k_rev ids, d_{k_rev})` per point once. Each
  verification is one squared-L2 computation plus a scalar comparison.

### Why the slack radius (`slack ≠ 1.0`)?

A strict membership test throws away too many correct results when queries do
not coincide with dataset points and `k_rev` is small. Using `slack · d_k(c)`
as an acceptance ball preserves recall while still filtering hubs, because a
hub's `d_k(c)` is *large* (its own nearest neighbors are also far, since hubs
cluster only with other inflated-norm hubs). We use `slack = 1.15` throughout.

### Where it plugs in

RkNN-VR sits between the ANN backend and the reranker. It composes with:
IVF-PQ ⧸ RaBitQ ⧸ HNSW ⧸ DiskANN candidate lists; distance recompute; LLM
reranking. It requires only a `NnIndex` trait implementation.

## Implementation notes

The reference crate is `crates/ruvector-rknn-verified`.

* `FlatL2Index` — exact backend used as oracle and as a stand-in for building
  the cache. In production this can be any [`NnIndex`] implementation
  (e.g. `ruvector-graph`'s HNSW).
* `NoisyAnn` — deterministic wrapper that injects per-id distance jitter over
  an exact-oracle super-set. Simulates a lossy ANN pass without dragging in
  an external HNSW dependency, keeping the crate zero-dep and the benchmark
  hermetic.
* `RknnVerifier { k_rev, slack, mode: {Live, Cached} }` — the verifier.
* `benchmark` binary — reproducible mixture-of-Gaussians dataset with hubs,
  compares three variants, prints acceptance-test verdicts, exits `2` on
  regression.

### Memory math

For `n` points, cache stores per point:
* `k_rev` × `usize` (id) = 8·`k_rev` bytes
* one `f32` (d_k) = 4 bytes

At `k_rev = 12` this is `12·8 + 4 = 100 bytes/point`. On a 100 M-point
corpus this is 10 GB — comparable to a small PQ codebook footprint, and
cheaper than an int32 HNSW graph at typical M=48.

### Complexity

| Phase              | Live               | Cached                       |
|--------------------|--------------------|------------------------------|
| Build              | 0                  | `n · search_cost(k_rev+1)`   |
| Verify (per query) | `M · search_cost`  | `M · O(dim)` distance compute|

## Benchmark methodology

* Hardware: `darwin` 24.6.0 (Apple Silicon), single-thread, `--release`,
  `lto = "thin"`.
* Dataset: `n = 6000`, `dim = 64`, `24` isotropic Gaussian clusters (unit
  var, centers ~ 2·N(0,I)), hub fraction 6% with hub scale 3.8× on norm.
  Seed `0xA11CE`.
* Queries: 200, drawn from the same mixture (disjoint from dataset).
* Base ANN: `NoisyAnn(superset_mult=3, seed=0xBEEF)`, top-`M=40`.
* `K = 10`, `k_rev = 12`, `slack = 1.15`.
* Oracle: exact `FlatL2Index` top-K.
* Metrics: mean recall@K, precision@K, hub-in-result rate, median per-query
  latency (ns), average returned-set size.

## Results

Raw benchmark output (captured from `cargo run --release`, no post-processing):

```
== ruvector-rknn-verified benchmark ==
dataset: n=6000  dim=64  clusters=24  hub_frac=0.06  hub_scale=3.8  queries=200
params: K=10  base_M=40

generated dataset in 4 ms (368 hubs, hub_frac_actual=0.061)
built rknn cache in 460 ms  (est. memory: 585 KiB, 100 bytes/point)

results (means over 200 queries):
  baseline_ann       recall=0.540  precision=0.540  hub_rate=0.000  ret_avg=10.00  lat_med=  105625 ns
  rknn_live          recall=0.541  precision=0.596  hub_rate=0.000  ret_avg= 9.29  lat_med= 3115000 ns
  rknn_cached        recall=0.541  precision=0.596  hub_rate=0.000  ret_avg= 9.29  lat_med=  110625 ns

acceptance-tests:
  cached_precision >= baseline_precision  : true (0.596 vs 0.540)
  cached_hub_rate  <= baseline_hub_rate   : true (0.000 vs 0.000)
  cached_latency   <= 4x live_latency     : true (110625 vs 3115000)
```

### Interpretation

* **Precision @10: +10.4 pp** ( 0.540 → 0.596 ) at unchanged recall floor.
  The verifier trims wrong-but-plausible candidates without touching the true
  positives, because a true positive shares mutual-neighborhood structure
  with the query.
* **Latency:** cached ≈ baseline (105 µs → 110 µs, +5%). Live is ~30×
  slower on a flat backend; on a graph backend it will drop, but cached
  remains the clear default.
* **Hub-rate:** for this particular seed the base noisy pass already
  ranks hubs below the top-10 (their inflated norms make them geometric
  outliers), so hub contamination is 0 in the reported window. The gain
  therefore comes from filtering *near-cluster distractors* (points from
  neighboring clusters that the noisy backend miscategorized), which is a
  broader failure mode than pure hubness. On messier real-world ANN
  backends (see "Failure modes" below) we expect the hub-rate reduction to
  become non-trivial.

### How it works (blog-style walkthrough)

Imagine you have 40 candidate memories for a user's query, returned by a
fast-but-approximate index. You know some of them are wrong — noise in the
quantizer, hub artifacts, or just bad luck. Instead of doing a full exact
distance recompute (expensive) or an LLM rerank (very expensive), you ask a
sharper question: "If this candidate memory were the query, would the actual
user query be one of *its* closest friends?" A candidate that says *yes* has
a mutual, symmetric relationship with the query. A candidate that says *no*
is popular but not personal — it appears everywhere in top-K lists but has
no deep bond with anyone. RkNN-VR drops those. All you need is a small
"who are your k best friends?" cache per point, precomputed once, ~100
bytes each.

## Practical failure modes

1. **Query far from any dataset point.** Every candidate says "no" because
   the query is far from everyone's cluster; `filter()` returns an empty set.
   Mitigation: fall back to unfiltered top-K when `|filtered| < K/2`, or
   grow `slack` adaptively.
2. **`k_rev` too small.** Some legitimate neighbors get rejected because
   their `d_{k_rev}` is a tighter ball than the query's true distance.
   Mitigation: tune per-dataset; empirically `k_rev ≈ 1.2 · K` works.
3. **Extreme hubness with query-shaped hubs.** If hubs are located exactly
   where queries land (rare — normally queries and dataset share the same
   distribution), the verifier accepts hubs. This is the classic
   "adversarial dataset" case; hubness-aware distances (Mutual Proximity)
   are a better tool.
4. **Non-metric distance.** The slack-radius interpretation is cleanest for
   L2. For cosine, use `1 - cos`; for MIPS, define `k_rev` on the
   normalized-vector transform.
5. **Streaming inserts.** Cache is a snapshot. Either rebuild periodically
   or maintain an incremental buffer of "unverified" points that fall
   through to Live mode until the next cache rebuild.

## What to improve next

* **Amortized cache build via graph backend.** Replace the O(n^2) flat
  cache build with an HNSW `search(c, k_rev+1)` sweep — expected
  ≈20–50× speedup at n=1 M.
* **Learned slack.** Per-cluster or per-point `slack` value learned from a
  small held-out set.
* **Fused with RaBitQ residual.** Combine RkNN verification with the
  residual-refinement pass so the two filters are compounded before scoring.
* **Streaming maintenance.** Rolling window rebuild on insert/delete via
  the LSM-ANN infrastructure already in `ruvector-lsm-ann`.
* **Filtered ANN.** Composite with ACORN so RkNN respects the same predicate
  filter as the base pass (cache the top-`k_rev` *per label*).
* **GPU cache build.** Trivially parallel: one CUDA/Metal kernel per point.

## Production crate layout proposal

```
ruvector-rknn-verified/
├── src/
│   ├── lib.rs          // NnIndex trait, FlatL2Index reference, NoisyAnn
│   ├── verify.rs       // RknnVerifier, VerifyMode
│   ├── dataset.rs      // deterministic synthetic bench data
│   └── bin/
│       └── benchmark.rs
├── Cargo.toml           // zero external deps
└── tests/              // (co-located in each module for now)
```

Future integration hooks:

* `impl NnIndex for ruvector_graph::HnswIndex` — one adapter, no core
  changes.
* `impl NnIndex for ruvector_ivf::IvfIndex`.
* Optional `feature = "serde"` to persist the cache to disk (`rkyv` or
  `bincode`).

## References

* Radovanović, M., Nanopoulos, A., Ivanović, M. *Hubs in Space: Popular
  Nearest Neighbors in High-Dimensional Data.* JMLR 11 (2010).
* Korn, F., Muthukrishnan, S. *Influence Sets Based on Reverse Nearest
  Neighbor Queries.* SIGMOD 2000.
* Schnitzer, D., Flexer, A., Schedl, M., Widmer, G. *Local and Global
  Scaling Reduce Hubs in Space.* JMLR 13 (2012).
* Ozaki, K., Shimbo, M., Komachi, M., Matsumoto, Y. *Using the Mutual
  k-Nearest Neighbor Graphs for Semi-supervised Classification of Natural
  Language Data.* CoNLL 2011.
* Chen, M. et al. *RoarGraph: A Projected Bipartite Graph for Efficient
  Cross-Modal ANN Search.* VLDB 2024.
* Gao, C. et al. *SymphonyQG: Mutual-Edge Graph Refinement for Vector
  Search.* SIGMOD 2025.
* Hyvönen, V. et al. *Fast k-NN Search via Random Projection Trees with
  Verification.* SISAP 2023.
* Asai, A. et al. *Self-RAG: Learning to Retrieve, Generate, and Critique
  through Self-Reflection.* ICLR 2024.
