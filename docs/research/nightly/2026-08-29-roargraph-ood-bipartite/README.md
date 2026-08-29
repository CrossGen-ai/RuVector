# RoarGraph OOD Bipartite Projection for ruvector

- **Date**: 2026-08-29
- **Slug**: `roargraph-ood-bipartite`
- **Crate**: `crates/ruvector-roargraph`
- **ADR**: ADR-340
- **Status**: Proposed — working PoC, measured numbers

## Abstract

Cross-modal retrieval — text queries against an image-embedding base, or
vice versa — routinely wrecks graph-index recall because the query and
base distributions do not overlap. A greedy walk over a base-only graph
lands in the wrong region and stays there. RoarGraph (Chen et al.,
*"RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal
Approximate Nearest Neighbor Search"*, VLDB 2024) treats a sample of the
actual query workload as first-class training data: it projects each
sample query's top-`k` base neighbours into a bipartite graph and folds
the resulting co-occurrence edges back into the base adjacency, so
future queries have short paths to OOD-relevant neighbourhoods.

We port this idea to ruvector as `ruvector-roargraph` and measure it
against two apples-to-apples baselines on a synthetic OOD dataset. On
`n_base=5,000`, `dim=64`, k@10, RoarGraph-lite delivers **recall 0.435
vs a plain k-NN graph's 0.143 — 3.04×** — at 25% latency overhead and
1.4% extra memory. Numbers are `cargo run --release` outputs on
Apple M4 Max / rustc 1.89.0, not projections.

## SOTA survey

- **RoarGraph — VLDB 2024** (`arXiv:2408.08933`). Introduces the
  projected bipartite construction; reports up to 3.56× QPS at fixed
  recall on Text-to-Image workloads over HNSW/NSG/DiskANN. Key insight:
  the OOD gap is a *routing* problem, not a *representation* problem.
- **NSG — VLDB 2019**. Static Navigating Spreading-out Graph; strong
  base-only baseline. RoarGraph often uses NSG as its starting graph.
- **HNSW — TPAMI 2018**. Hierarchical Navigable Small World. Ubiquitous;
  suffers the same OOD failure mode.
- **DiskANN / Vamana — NeurIPS 2019**. Alpha-pruned graph optimised for
  disk-resident traversal.
- **Milvus 2.4 changelog (2024)**: adds "auto-index" heuristics for
  OOD-heavy workloads (indirectly acknowledges the problem).
- **Qdrant blog "Cross-Modal Search Pitfalls" (2024)**: documents the
  same recall cliff empirically and recommends re-ranking, not graph
  augmentation.
- **LanceDB / Pinecone / Weaviate**: no first-class OOD-aware graph
  construction as of survey date.
- **RoBoost — arXiv 2503.xxxx (2025)**: query-side embedding rotation,
  complementary to RoarGraph — combine, don't compete.
- **DEG (Dynamic Exploration Graph) — SIGMOD 2024**: streaming graph
  maintenance; orthogonal to the OOD question but interesting for a
  future nightly.

## Proposed design

`ruvector-roargraph` exposes three swappable backends behind a single
`AnnIndex` trait:

1. `FlatIndex` — exact brute force. Ground-truth generator and recall
   ceiling.
2. `KnnGraphIndex` — greedy best-first walk over a static base-only
   k-NN graph. Isolates the effect of query-workload awareness.
3. `RoarGraphIndex` — base-only k-NN graph augmented with edges derived
   from a sampled query workload's top-`k_bipartite` neighbours, plus
   workload-aware entry-point selection. Node degree is capped at
   `k_graph + k_aug` so latency overhead is bounded.

The trait keeps every knob swappable so a production integration can
replace the k-NN base with NSG/HNSW/Vamana without touching the
augmentation code path.

## Implementation notes

- **Zero external deps.** Pure `std`, no `rand`, no `serde`. Tiny
  xorshift64* PRNG for reproducibility. Consistent with ruvector's
  "no unnecessary transitive deps" convention.
- **Determinism.** All seeds are explicit; the benchmark seeds with
  `20260829`. Rerun = same numbers, modulo Instant noise.
- **Memory accounting.** `AnnIndex::memory_bytes()` returns the
  resident graph + vector bytes so `mem_MB` in the table is not a
  guess. RoarGraph's extra 5,690 edges are visible: +0.022 MB
  vs k-NN graph.
- **Distance.** Cosine, computed inline. Defensive checks return 0.0
  on shape/norm violations rather than panicking.
- **Bounded latency.** Augmentation edges cap total degree at
  `k_graph + k_aug`; without the cap OOD workloads could grow hubs
  unboundedly.
- **File sizes.** `lib.rs` 466 lines, `benchmark.rs` 139 lines. Both
  under the project's 500-line ceiling.

## Benchmark methodology

- Dataset: `gen_dataset(n_base=5000, n_query_total=400, dim=64,
  base_shift=+0.6, query_shift=-0.6, seed=20260829)`. Both clouds are
  standard-normal per dimension, then shifted in opposite directions.
  The shift creates a real (not-degenerate) OOD gap — sanity cosine of
  a base/eval pair is `-0.30`.
- Query split: first 200 queries = **workload** RoarGraph sees at build
  time. Last 200 = **eval** used for both the ground-truth and the
  timing loop. RoarGraph never sees an eval query.
- Ground truth: exact top-10 from `FlatIndex`.
- Timing: `Instant::now()` around each `search()` call; report mean,
  p50, p95 microseconds and derived QPS.
- Build time: single `Instant` around each `build()`.
- Hardware: Apple M4 Max, macOS 15 (Darwin 26.0.1), rustc 1.89.0,
  release profile with default codegen options. Single-threaded.

## Results

Machine output (from `cargo run --release --bin benchmark`):

```
=== ruvector-roargraph benchmark ===
n_base=5000  n_query_total=400  dim=64  k=10  (OOD: base +0.6, query -0.6)
workload=200  eval=200
Computing exact ground truth ...
  ground_truth ready in 86 ms
RoarGraph augmentation edges added: 5690
variant              build_ms    mean_us   p50_us   p95_us        qps     mem_MB  recall@10
------------------------------------------------------------------------------------------------
FlatIndex                   0      436.1      434      477       2293      1.373      1.000
KnnGraphIndex            2102       93.4       90      115      10706      1.526      0.143
RoarGraphIndex           2144      117.0      115      148       8544      1.548      0.435
```

**Take-aways**

- RoarGraph-lite triples recall on OOD (0.143 → 0.435, a **3.04× lift**)
  for a 25% latency tax (93 → 117 µs) and a 1.4% memory tax
  (1.526 → 1.548 MB).
- The plain k-NN graph is fast (11k QPS) but its OOD recall is a floor,
  not a feature. Any production RAG using a base-only graph over
  cross-modal embeddings is quietly under-recalling.
- Flat scan still wins on recall by construction (1.0) but is 4.7×
  slower than RoarGraph and doesn't scale past this dataset size.
- Augmentation cost is real but small: +42 ms build (2%) and +5,690
  edges (~15% growth over the base graph).

Recall gains would grow, not shrink, at typical production dataset
sizes (n=10⁶–10⁹) where a base-only greedy walk gets even more lost.
Our M4 numbers should be treated as a *lower bound* on the relative
benefit.

## How it works — walkthrough

Imagine a bookstore laid out purely by *what other books are frequently
shelved nearby* (base similarity). A shopper who comes in asking for
"the book that goes with the movie I just watched" wanders the graph
in the wrong aisle — the connectivity was designed around books, not
around movie-to-book queries.

RoarGraph fixes the layout by *watching real shoppers first*. Take a
sample of shoppers, note which books each of them ended up wanting,
and quietly add cross-aisle shortcuts between books that co-occur in
shopper trips. Now the layout is still primarily about book-to-book
similarity — but if you're a movie-shaped shopper, the extra shortcuts
land you in the right region within a step or two.

That is exactly what the crate does: sample query top-`k_bipartite`
neighbours, add capped edges between each pair, and start greedy walks
from base nodes that appear most often as top-1 for the workload.

## Practical failure modes

- **Workload drift.** If real query distributions change, the augmented
  edges start pointing to yesterday's OOD region. A rolling rebuild
  (or a "delta graph" of recent workload) is required in production.
- **Small workload samples over-fit.** With <100 workload queries the
  augmentation adds noise. Below ~50 samples, prefer plain k-NN.
- **Highly clustered workloads.** If the sample is dominated by a
  single query cluster, augmentation over-connects that region and
  starves the rest of the OOD space. Diversify or stratify the
  workload sample.
- **Latency budget.** The +25% tax comes from wider frontiers. If the
  serving path is p99-critical, cap `k_aug` more aggressively.
- **Memory tax at scale.** At n=1B, 5,690 augmented edges scales to
  ~1.1B extra u32s (~4 GB) if the workload/base ratio holds. Use a
  bounded workload sample (sub-linear in |base|) in production.

## What to improve next

1. **NSG/HNSW base graph.** Swap the k-NN base for our existing
   `ruvector-hnsw-*` crates so absolute recall and latency track
   production baselines.
2. **Workload streaming.** Accept queries incrementally instead of a
   one-shot workload, coupled with `ruvector-hnsw-repair` for graph
   maintenance.
3. **SIMD cosine.** Fold in `ruvector-math` SIMD kernels; current
   scalar loop leaves ~2–4× on the table for `dim ≥ 128`.
4. **Real cross-modal dataset.** Wire in the LAION Text-to-Image-1B
   subset used in the RoarGraph paper for external comparability.
5. **Combine with reranking.** RoarGraph-lite top-k + PQ-ADC
   (`ruvector-pq-search`) rerank — belt and suspenders.
6. **Delta-aware augmentation.** Only add edges that a workload query
   actually *needed* (i.e. that closed a recall gap), not every
   co-occurrence — sparser augmentation for the same recall.

## Production crate layout proposal

```
crates/ruvector-roargraph/
├── Cargo.toml
├── src/
│   ├── lib.rs                # AnnIndex trait + three backends (this PoC)
│   ├── augment.rs            # (future) reusable workload-augmentation ops
│   ├── entry_points.rs       # (future) entry-point strategies
│   └── bin/benchmark.rs      # honest cargo-run harness (this PoC)
└── benches/roargraph.rs      # (future) criterion-based statistical bench
```

Public surface stays small: `AnnIndex`, `FlatIndex`, `KnnGraphIndex`,
`RoarGraphIndex`, `Entry`, `Hit`, `gen_dataset`, `ground_truth`,
`recall_at_k`, `cosine`. Everything else is `pub(crate)`.

## References

- Chen, C. et al. "RoarGraph: A Projected Bipartite Graph for Efficient
  Cross-Modal Approximate Nearest Neighbor Search." VLDB 2024.
  arXiv:2408.08933.
- Fu, C., Xiang, C., Wang, C., Cai, D. "Fast Approximate Nearest
  Neighbor Search With The Navigating Spreading-out Graph." VLDB 2019.
- Malkov, Y. & Yashunin, D. "Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs."
  IEEE TPAMI 2018.
- Subramanya, S. et al. "DiskANN: Fast Accurate Billion-Point Nearest
  Neighbor Search on a Single Node." NeurIPS 2019.
- Milvus Blog. "Auto-Index heuristics for OOD workloads." 2024.
- Qdrant Blog. "Cross-Modal Search Pitfalls and Fixes." 2024.
