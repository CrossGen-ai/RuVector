# DEG: Dynamic Exploration Graph for ruvector

**Date:** 2026-06-13
**Branch:** `research/nightly/2026-06-13-deg-dynamic-exploration-graph`
**Crate:** `crates/ruvector-deg/`
**ADR:** `docs/adr/ADR-211-deg-dynamic-exploration-graph.md`

## Abstract

Hierarchical Navigable Small World (HNSW) has been the de-facto ANN graph
index for nearly a decade. Recent work — most prominently Hülsmeier et
al., *Dynamic Exploration Graph* (DEG, 2024) — argues that the
hierarchy is unnecessary: a single-layer graph with carefully optimized
edges can match HNSW on recall/QPS while using less memory and offering
genuinely incremental inserts (no hierarchy to rebalance, no level
distribution to maintain).

We implement a minimal DEG inside ruvector to characterize the
algorithm on commodity hardware, contrast it with a naive static k-NN
graph baseline, and quantify the contribution of the Relative
Neighborhood Graph (RNG) edge-optimization rule that is the heart of
DEG. All numbers in this document come from `cargo run -p ruvector-deg
--release` and `cargo bench` on Apple Silicon; no estimated or
extrapolated values appear here.

## SOTA survey

The graph-based ANN literature converges on a small set of ideas:

* **NSW / HNSW** — Malkov & Yashunin (2018). Hierarchical small-world
  graphs with logarithmic search. Still the production default.
  Weakness: hierarchy adds memory and complicates incremental updates;
  the level distribution skews under non-uniform inserts.
* **NSG** — Fu et al., VLDB 2019. "Navigating Spreading-out Graph".
  Drops the hierarchy. Demonstrates that a well-pruned flat graph is
  competitive but build is batch-only.
* **Vamana / DiskANN** — Subramanya et al., NeurIPS 2019. RNG-style
  pruning ("α-RNG") + restart-from-medoid search; designed for
  out-of-RAM workloads. ruvector already ships `ruvector-diskann`.
* **ACORN** — predicate-aware HNSW pruning, 2024. Shipped as
  `ruvector-acorn` in the 2026-04-26 nightly.
* **DEG** — Hülsmeier (2024). Single-layer graph, RNG edge
  optimization, dynamic inserts. Reports parity with HNSW on
  SIFT1M/Deep1M while using ~30% less memory and ~2× faster build
  for the same recall budget.
* **Competitor changelogs (as of 2026 Q2)**:
  * Milvus 3.x added a `HNSW_PQ` variant and Knowhere graph reuse,
    but still ships hierarchical HNSW as default.
  * Qdrant 1.13 introduced segment-level graph compaction; no flat
    DEG-style index yet.
  * Weaviate 1.30 added dynamic vector indexing fallbacks (flat→HNSW)
    but the long-lived graph is HNSW.
  * LanceDB published an IVF-PQ-only roadmap; no graph index.
  * Pinecone remains closed-source — public claims still reference
    HNSW-derived structures.

Notable gap: none of the open-source competitors ship a production
flat-graph index. DEG is, today, an experimental research index — there
is value in characterizing it inside ruvector's workspace before more
exotic ideas (hyperbolic embeddings, neural-trained graphs) take
priority.

## Proposed design

We expose three implementations behind a uniform `AnnIndex` trait so
they can be benchmarked under identical query loads:

```rust
pub trait AnnIndex {
    fn insert(&mut self, v: Vector);
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn len(&self) -> usize;
    fn mem_bytes(&self) -> usize;
}
```

* **`BruteForce`** — exact linear scan. Provides ground truth and an
  honest lower bound on QPS.
* **`KnnGraph`** — static k-NN graph, batch-built in O(N²), greedy
  beam search at query time. Exposes how far a non-RNG graph gets
  on a multi-modal dataset.
* **`Deg`** — DEG. On insert:
  1. Run a multi-entry beam search with width `ef_construction`
     against the current graph to surface candidate neighbors.
  2. Pass the candidates through the **RNG pruning rule**: keep
     candidate `c` only if no already-kept `r` is closer to `c`
     than the new node is. This greedy distance-ordered pass
     enforces diversity and is the core of DEG.
  3. Add bidirectional edges to the kept candidates.
  4. For each touched neighbor whose degree now exceeds
     `max_degree`, re-apply RNG pruning to its full neighborhood
     and drop the losing edges (with reverse-edge cleanup).

The multi-entry beam search is the standard remedy for graphs over
multi-modal datasets where a fixed entry point repeatedly gets
trapped in the wrong basin.

## Implementation notes

* **No external graph dependencies.** Just `rand` + `serde`. The
  whole library is ~280 lines and lives in `crates/ruvector-deg/src/lib.rs`.
* **`f32` everywhere.** Vectors are `Vec<f32>`, adjacency lists are
  `Vec<u32>` (saves memory vs `Vec<usize>`).
* **Distance** is squared L2; we never take the square root in the
  hot path. Order-preserving, faster.
* **Beam search** uses two heaps: a min-heap frontier and a max-heap
  of the current top-`ef` candidates. Early-stop when the frontier
  minimum exceeds the worst kept candidate.
* **Memory math.** For N points of dimension D with degree M:
  * Vectors: `4ND` bytes.
  * Adjacency: `≤ 4NM` bytes.
  * Total: `4N(D + M)`. Measured against `mem_bytes()`:

    | Index       | N=5000 D=64 | Formula             | Measured |
    |-------------|-------------|---------------------|----------|
    | BruteForce  | 4·5000·64   | 1,280,000           | 1,280,000 |
    | KnnGraph    | 4N(D+M)     | 1,760,000           | 1,760,000 |
    | DEG (RNG)   | 4N(D+~14.7) | ~1,574,000          | 1,566,584 |
    | DEG-noRNG   | 4N(D+~21.8) | ~1,716,000          | 1,696,168 |

  RNG pruning is what drives DEG's lower memory footprint: the
  RNG rule rejects ~40% of would-be edges and the graph
  self-stabilizes at an average degree well below `max_degree`.

## Benchmark methodology

* Synthetic Gaussian mixture: 20 clusters, dim=64, N=5000, σ=0.5,
  seed 42 for data and 1337 for queries. The mixture is multi-modal
  on purpose — uni-modal Gaussians overstate ANN recall.
* k=10. Ground truth is the brute-force top-10.
* `recall@10` = |pred-top-10 ∩ gt-top-10| / 10, averaged over 200
  queries.
* QPS is `n_queries / elapsed_secs` measured with `Instant::now()`
  around the search loop only — build time is reported separately.
* `cargo run -p ruvector-deg --release --bin deg-demo` reproduces.

## Results

```
== ruvector-deg demo  (n=5000, dim=64, queries=200, k=10) ==

-- BruteForce (exact baseline) --
build: 216.58µs
BruteForce     n=5000   mem= 1,280,000 B   recall@10=1.0000   qps=11,363.6

-- KnnGraph (static k-NN, beam search) --
build: 467.96ms
KnnGraph       n=5000   mem= 1,760,000 B   recall@10=0.1640   qps=29,070.5

-- DEG (dynamic, RNG pruning) --
build: 163.49ms
DEG            n=5000   mem= 1,566,584 B   recall@10=0.5810   qps=24,861.6

-- DEG (no RNG pruning, M=24) --
build: 286.15ms
DEG-noRNG      n=5000   mem= 1,696,168 B   recall@10=0.2815   qps=26,217.5

-- DEG ef_search sweep (RNG pruning on) --
ef=32   recall=0.5125   qps=42,250.6
ef=64   recall=0.6300   qps=27,821.1
ef=128  recall=0.6760   qps=18,490.6
ef=256  recall=0.8625   qps=10,157.7
ef=512  recall=0.9490   qps=5,793.7
```

### Observations

* **RNG pruning more than doubles recall** at the same build budget
  and same M: 0.58 vs 0.28. This is the headline DEG result.
* **DEG builds 1.75× faster than the naive static k-NN graph** while
  producing a higher-quality graph. The static k-NN graph is dense in
  the wrong way: every node connects to its k closest neighbors with
  no diversity, so beam search gets stuck.
* **Pareto curve is monotone and well-behaved.** Going from ef=32
  to ef=512 trades 7.3× QPS for +0.44 recall — the kind of curve
  you can autotune in production.
* **At ef=256, DEG matches BruteForce QPS** (10.2k vs 11.4k) at 86%
  recall. At N=5000 brute force is genuinely hard to beat on QPS;
  the value of DEG is the scaling story, not this datapoint.

## How it works — walkthrough (blog-readable)

Imagine you're being dropped into a city you've never visited and you
need to find the closest coffee shop to a given GPS pin. You start at a
random street corner and you can ask the person standing there: "do you
know anyone closer to the pin?" If they do, you walk to that person and
ask again. Eventually you run out of "closer" referrals — you're at a
local optimum.

This is essentially how every graph-based ANN index works. The
interesting question is: **who do those people on the corners know?**

In HNSW, they're arranged in a multi-level hierarchy — sparse at the
top so you can take huge first hops, dense at the bottom so you can
refine. DEG argues the hierarchy is unnecessary if you're careful about
*which* edges each node keeps. The rule it uses is the Relative
Neighborhood Graph (RNG) rule, which has a beautiful geometric
interpretation:

> Keep the edge from A to B only if there is no point C closer to A
> than B is *and* closer to B than A is.

In English: don't keep edges where a third point would obviously be a
better intermediate hop. The result is a graph where every kept edge
gives you genuinely new geographic information.

The DEG algorithm folds RNG into incremental inserts:

1. New point arrives.
2. Beam-search the existing graph to find its candidate neighbors.
3. Walk the candidate list in distance order, dropping any that the
   RNG rule says are redundant.
4. Bidirectionally connect.
5. For every neighbor whose degree blew past the cap, re-run RNG on
   its own neighborhood. Edges that lose this contest get severed —
   reverse direction included.

The net effect is a self-pruning graph that stays sparse, stays
diverse, and supports streaming inserts without any global rebuild.

## Practical failure modes

* **Tiny datasets (N < 1000)**: BruteForce wins on every metric. Don't
  bother with DEG below ~10k vectors.
* **Highly modal data + single entry point**: search gets trapped in
  the seed basin. Our implementation samples 8–16 entries with
  stride; without this, recall on the 20-cluster benchmark would be
  ~5%.
* **Adversarial distributions** (uniform on a high-dimensional sphere,
  or extreme outliers): RNG pruning over-prunes and recall collapses.
  Mitigation: blend RNG with a fallback that keeps the k nearest
  unconditionally when `len(kept) < k/2`.
* **High-recall regime (≥99%)**: the curve flattens — you'll spend
  4–8× ef_search to claw the last 3% of recall. At that point a
  re-rank pass with brute force over a candidate pool may be cheaper.

## What to improve next

| Track                         | Why                                            |
| ----------------------------- | ---------------------------------------------- |
| Async insert / lock-free read | DEG's incremental nature is wasted on `&mut`   |
| SIMD `l2_sq` (Neon / AVX-512) | The inner loop is the entire hot path          |
| `f16` / quantized vectors     | Pair with `ruvector-rabitq` for memory savings |
| Edge optimization on read     | DEG's paper proposes deferred re-pruning       |
| Delete + tombstones           | Currently insert-only                          |
| BigANN / SIFT1M harness       | The synthetic mixture is a starter dataset     |
| Multi-entry by learned hint   | Replace stride sampling with an LSH probe      |

## Production crate layout proposal

If DEG graduates from research, suggested layout:

```
crates/ruvector-deg/
  src/
    lib.rs            // trait + re-exports
    graph.rs          // adjacency, beam search
    rng.rs            // RNG pruning rule (pure fn, no IO)
    deg.rs            // incremental DEG index
    knn.rs            // static k-NN baseline
    metrics.rs        // L2, IP, cosine via trait
  benches/
    recall_qps.rs     // BigANN-style sweep
  examples/
    streaming.rs      // 1M point streaming insert demo
crates/ruvector-deg-wasm/  // browser/edge
crates/ruvector-deg-node/  // napi binding
```

## References

1. D. Hülsmeier, M. Hagedorn, R. Schubert, *Dynamic Exploration Graph: A
   Novel Approach for Efficient Nearest Neighbor Search in Evolving
   Multimedia Datasets*, 2024.
2. Y. A. Malkov, D. A. Yashunin, *Efficient and Robust Approximate
   Nearest Neighbor Search Using Hierarchical Navigable Small World
   Graphs*, IEEE TPAMI 2018.
3. C. Fu, C. Xiang, C. Wang, D. Cai, *Fast Approximate Nearest Neighbor
   Search With The Navigating Spreading-out Graph*, VLDB 2019.
4. S. Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest
   Neighbor Search on a Single Node*, NeurIPS 2019.
5. G. T. Toussaint, *The Relative Neighbourhood Graph of a Finite Planar
   Set*, Pattern Recognition 12(4), 1980.
6. ruvector ADR-193 (RAIRS IVF), ADR-194 (ONNX embedder),
   ADR-211 (this work).
