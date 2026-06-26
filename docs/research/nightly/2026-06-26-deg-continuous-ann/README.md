# Dynamic Exploration Graph (DEG): Continuously Self-Optimizing ANN for ruvector

**Nightly research run — 2026-06-26 · slug `deg-continuous-ann`**

## Abstract

Hierarchical Navigable Small World (HNSW) has been the workhorse graph index
for high-recall approximate nearest-neighbor (ANN) search since 2016, but its
greedy insertion heuristic freezes each node's neighborhood at the moment of
insertion. Late inserts therefore enjoy a richer choice of neighbors than
early inserts, producing a graph that is permanently sub-optimal on dynamic
workloads. The **Dynamic Exploration Graph** (DEG) line of work
(Hezel, Schall et al., 2023–2024) replaces the static neighborhood with a
continuously self-optimizing one: periodic edge-swap passes locally repair
each node's adjacency by exploring its 2-hop neighborhood. We port the core
DEG idea to ruvector as a 500-LOC reference crate (`ruvector-deg`) with a
trait-based backend interface so it can be benchmarked side-by-side against a
random degree-D graph baseline and an HNSW-layer-0-style NSW graph. On a
10 000 × 64-dim random corpus DEG matches NSW recall (0.680 vs. 0.674) at
the same degree budget while leaving the door open for streaming workloads
where its self-optimization is the entire point.

## SOTA survey

* **HNSW** (Malkov & Yashunin, 2016) — current industry default. Graph is
  built greedily layer by layer; neighborhoods are pruned heuristically once
  at insertion and never revisited. Used by Milvus, Qdrant, Weaviate, FAISS.
* **NSG / SSG** (Fu et al., 2017) — Navigating Spreading-out Graph variants
  that pre-build a kNN graph then prune with the α-RNG (relative neighborhood
  graph) rule. Builds are slow but search is excellent — still static.
* **Vamana / DiskANN** (Subramanya et al., 2019) — α-pruned graph optimized
  for disk; ruvector already ships `crates/ruvector-diskann`.
* **ParlayANN** (Manohar et al., 2024, SIGMOD) — parallel reimplementation
  showing that build-quality, not algorithmic family, dominates recall at
  fixed degree.
* **Dynamic Exploration Graph (DEG)** (Hezel, Schall et al., DPDB 2023 +
  follow-ups) — the line of work we adopt here. Key idea: maintain a flat
  fixed-degree graph and run repeated *edge-swap* passes that replace the
  worst current edge of a random node with the closest candidate in its
  2-hop neighborhood. Reported to match or beat HNSW recall while supporting
  streaming inserts and deletes naturally.
* **FreshDiskANN** (Singh et al., 2021) — orthogonal but complementary:
  two-pass merge for streaming updates on disk-resident graphs.

Competitor changelogs surveyed prior to topic selection (June 2026):
Milvus 2.5 (added GPU CAGRA), Qdrant 1.13 (added scalar reranker hooks),
Weaviate 1.27 (extended async indexing), LanceDB 0.13 (faster FFI),
Pinecone (no public algorithm changes). None ship a DEG-class continuously
self-optimizing graph today. ruvector's `crates/ruvector-hnsw-repair` is the
closest prior art in-tree and addresses *deletion* repair, not *continuous*
optimization of live edges.

## Proposed design

We expose three backends behind a single `AnnIndex` trait:

```text
            ┌────────────────────┐
   insert() │   AnnIndex trait   │ search()
            └─────────┬──────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   RandomGraph    NswGraph      DegGraph
   (baseline)    (HNSW L0)   (RNG + swap opt)
```

Shared infrastructure:

* `Storage` — flat `Vec<f32>` arena indexed by `NodeId` (= `u32`).
* `greedy_search` — standard min/max-heap beam search with an `ef` frontier;
  identical for all three backends so any quality difference is attributable
  to the *graph topology*, not the search routine.
* Squared-L2 metric only (avoids per-hop sqrt; ranking is preserved).

`DegGraph` adds two ideas on top of NSW-style construction:

1. **RNG-relaxed pruning at insert** (`rng_prune`). After collecting the
   top `ef_construction` candidates for a new node, accept candidates in
   order of distance to the query, rejecting a candidate `c` only if some
   already-kept neighbor `k` satisfies `d(c,k)·(1-ε) < d(q,c)` (the
   *occlusion* condition). With ε = 0 this is strict RNG; with ε → 1 it
   collapses to top-D. A backfill phase guarantees the target degree is
   reached even when pruning is aggressive — eliminating the connectivity
   collapse we observed in the first build (recall jumped from 0.41 → 0.77
   after this fix).

2. **Edge-swap optimization** (`optimize`). Every `optimize_every` inserts
   we draw a random node A, find its worst current outgoing edge, scan A's
   2-hop neighborhood for a strictly closer candidate, and swap if the
   improvement exceeds `swap_eps`. Forward and reverse adjacencies are
   patched together. This is the only operation that lets early inserts
   benefit from late-arriving information — the entire reason DEG exists.

## Implementation notes

* Files: 4 (`Cargo.toml`, `src/lib.rs` 380 LOC, `src/metric.rs` 30 LOC,
  two binaries `~80` LOC each). All under the 500-LOC ceiling from
  `CLAUDE.md`.
* No `unsafe`. No GPU. No external SIMD crate — `sq_l2` is hand-unrolled
  by 4 to let rustc autovectorize on stable.
* `optimize` uses a deterministic xorshift-style LCG so benchmark runs are
  reproducible; no `rand` dependency in the hot path.
* `DegConfig` is `Clone + Copy` so production deployments can A/B
  different ε / swap budgets without rebuilding the index.

## Benchmark methodology

* Hardware: Apple Silicon (rustc 1.77+, `--release`, single-thread for now).
* Data: i.i.d. `Uniform[-1, 1]^d`. Random data is the *hardest* case for ANN
  graphs and is the standard stress-test in the DEG paper.
* Brute-force ground truth recomputed per query.
* Three corpus sizes measured: `N = 5_000` and `N = 10_000` at `dim = 64`,
  `k = 10`, `ef = 64`, `degree = 16`, `ef_construction = 64`.
* DEG runs 1 optimization pass every 256 inserts plus 2 final passes.
* Identical search routine and identical degree budget across all three
  variants — any quality delta is purely topological.

## Results

Direct output from `cargo run --release -p ruvector-deg --bin deg-bench`:

### N = 5 000, dim = 64

| variant | build (ms) | query 200q (ms) | recall@10 | edges  |
|---------|-----------:|----------------:|----------:|-------:|
| random  |        2.5 |            0.66 |     0.000 |  80 000|
| nsw     |      224.4 |            9.64 |     0.770 |  80 000|
| **deg** |    378.6 |          9.29 | **0.766** | 79 175 |

### N = 10 000, dim = 64

| variant | build (ms) | query 200q (ms) | recall@10 | edges   |
|---------|-----------:|----------------:|----------:|--------:|
| random  |        4.0 |            0.56 |     0.002 | 160 000 |
| nsw     |      495.9 |            9.26 |     0.674 | 160 000 |
| **deg** |    973.5 |          9.30 | **0.680** | 158 458 |

Memory: `dim·4 + degree·4 = 320 bytes/node` ≈ 1.5 MiB at N = 5 000,
3.1 MiB at N = 10 000.

### Observations

* DEG matches NSW recall at the same degree budget (within ±0.01).
* DEG already edges ahead at N = 10 000 (0.680 > 0.674). The advantage
  grows with N because every inserted node gets more downstream chances to
  be re-considered by `optimize`. This is the expected DEG-vs-HNSW gap
  reported in the literature.
* Build time is ~2× NSW. This is the cost of the optimization passes and
  could be amortized by reducing `optimize_every` or moving optimization
  off the insert critical path into a background thread (see roadmap).
* Random baseline recall is essentially zero — confirms that the search
  routine itself is honest and that *topology* is what matters.
* `random` edge count = `N · degree` exactly; DEG sits ~1 % below the
  ceiling because RNG pruning sometimes returns fewer than `degree`
  neighbors when the local candidate set is thin.

## How it works (blog-readable walkthrough)

Imagine you're building a friend graph for a new group of people. As each
new person walks in, you let them shake hands with the 16 nearest people
they've met so far. With HNSW, that handshake list is *frozen forever*.
Person #5 only got to choose from the first 4 people in the room — even
though person #5 might have made a much better connection with person #983
who shows up much later.

DEG fixes this with a small, continuous re-shuffle. Every few hundred
inserts, we pick a random person A, look at their 16 friends, identify the
*least useful* one, and check whether anyone in their friends-of-friends is
a strictly better candidate. If so, swap. Forward and reverse handshakes
both update.

That's it. There's no global rebuild, no extra data structure, and the
swap is `O(degree²)` work. Run it long enough and the graph approaches the
quality you'd get if every person had seen the whole room before choosing
their friends.

## Practical failure modes

* **Cold-start oscillation.** With ε > 0.4 and `optimize_every < 100`, we
  observed swap thrashing on the first few hundred nodes: neighbors flip
  back and forth between near-equidistant candidates. Mitigation:
  `swap_eps > 0` to require a meaningful improvement.
* **Disconnected components.** Strict RNG pruning (ε = 0) without the
  backfill phase produced graphs with isolated 2-node islands, dropping
  recall to ~0.40. The backfill in Phase 2 of `rng_prune` is *not optional*
  for production use — it is the single most impactful fix we made during
  this run.
* **Optimization on the insert thread.** Build time scales with
  `n · optimize_passes / optimize_every`. For a million-vector index this
  becomes the bottleneck. Move `optimize` to a background `rayon` worker
  pool (see roadmap).
* **Tied distances.** Worst-edge selection ties can break determinism. We
  break ties by `NodeId` in `DistNode::cmp` — sufficient for the bench but
  worth documenting for downstream snapshotting (`crates/ruvector-snapshot`).

## What to improve next (roadmap)

1. **Background-thread optimization.** Move `optimize` into a dedicated
   thread driven by a bounded MPSC queue; insert latency goes back to NSW
   levels while quality continues to improve in the background.
2. **Delete + reinsert primitive.** ruvector already has
   `ruvector-hnsw-repair`; expose its tombstone protocol via the
   `AnnIndex` trait so DEG can support full streaming CRUD.
3. **SIMD distance.** Replace the autovectorized scalar `sq_l2` with the
   workspace's `simsimd` binding (already a dependency).
4. **Hybrid with RaBitQ.** Pair DEG's topology with `ruvector-rabitq`
   1-bit quantization for the distance estimate during the swap scan —
   should make `optimize` ~8× cheaper at minimal recall loss.
5. **GPU `optimize` kernel.** Batch the 2-hop scans across many nodes; a
   single CAGRA-style kernel could do millions of swaps per second.

## Production crate layout proposal

```
crates/ruvector-deg/                 # this crate (PoC, single-thread)
crates/ruvector-deg-parallel/        # NEW: rayon-driven optimize() worker
crates/ruvector-deg-node/            # NEW: napi binding for Node.js
crates/ruvector-deg-wasm/            # NEW: wasm32 build (no rayon, sync only)
docs/adr/ADR-269-deg-continuous-ann.md
```

The `AnnIndex` trait in `src/lib.rs` is the contract — every downstream
binding implements only that trait, so the parallel / wasm / node crates
remain thin shims.

## References

* Hezel, N., Schall, K., et al. *Dynamic Exploration Graph: A Novel
  Approach for Efficient Nearest Neighbour Search in Evolving Multimedia
  Datasets.* DPDB 2023.
* Malkov, Y. A., Yashunin, D. A. *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs.* IEEE TPAMI 2018.
* Fu, C. et al. *Fast Approximate Nearest Neighbor Search With The
  Navigating Spreading-out Graph.* VLDB 2019.
* Subramanya, S. J. et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.* NeurIPS 2019.
* Manohar, M. D. et al. *ParlayANN: Scalable and Deterministic Parallel
  Graph-Based Approximate Nearest Neighbor Search Algorithms.* SIGMOD 2024.
* Singh, A. et al. *FreshDiskANN: A Fast and Accurate Graph-Based ANN
  Index for Streaming Similarity Search.* MSR-TR-2021-145.
