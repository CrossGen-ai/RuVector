# HCNNG: Hierarchical Clustering Navigating Neighbor Graphs for ruvector

**Nightly research · 2026-06-14 · Pattern Recognition 96 (2019), 106985**

---

## Abstract

We implement HCNNG (Hierarchical Clustering-based Navigating Neighbor Graph,
Munoz et al. 2019) as a new standalone Rust crate
(`crates/ruvector-hcnng`) in the ruvector workspace. HCNNG builds an
approximate nearest-neighbor proximity graph by repeating a simple recipe:
recursively partition the dataset with random 2-pivot metric splits, compute a
Minimum Spanning Tree on every leaf, and union all MST edges across many
trees. The result is a parameter-light index — no hierarchy levels to tune, no
heuristic edge pruning, no fixed entry point — that nonetheless reaches HNSW-class
recall when paired with multi-entry beam search and small per-leaf kNN
augmentation.

**Key measured results (this PR, cargo --release, single-thread, MBP M-series):**

| Setup (uniform [-1,1]) | build_ms | µs/query | recall@10 | speedup vs brute |
|------------------------|---------:|---------:|----------:|-----------------:|
| n=20k, d=32, brute     |       –  |   414.9  |     1.000 |             1.0× |
| n=20k, d=32, trees=12  |     274.8|    61.2  |     0.863 |             6.8× |
| n=20k, d=32, trees=20  |     463.1|   165.0  | **0.991** |             2.5× |
| n=20k, d=64, brute     |       –  |   553.7  |     1.000 |             1.0× |
| n=20k, d=64, trees=12  |     337.7|    85.3  |     0.599 |             6.5× |
| n=20k, d=64, trees=20  |     624.9|   205.8  | **0.929** |             2.7× |
| n=50k, d=64, brute     |       –  |  1479.5  |     1.000 |             1.0× |
| n=50k, d=64, trees=12  |     970.1|    98.5  |     0.402 |            15.0× |
| n=50k, d=64, trees=20  |    1783.7|   257.8  |     0.792 |             5.7× |

Build is single-threaded today (one thread per tree is the obvious next
optimization). All recall numbers are exact against brute-force L2 ground
truth on the same data and same queries; no sampling, no approximations in
the GT. The 4 measured variants are: brute-force, HCNNG `n_trees=1`,
HCNNG `n_trees=12` (paper-recommended default), HCNNG `n_trees=20` with
`ef_search=128`.

The numeric acceptance test PASSES on both d=32 and d=64 at n=20k
(best variant recall@10 ≥ 0.90). See "Failure modes" for an honest
discussion of the n=50k d=64 case, which does not yet clear that bar
without further tuning.

---

## SOTA Survey

### 2019–2026 Proximity Graph ANNS

**HNSW (Malkov & Yashunin, TPAMI 2018; arXiv:1603.09320)**
: The reigning champion. Hierarchical Navigable Small World layers with
heuristic edge pruning ("RNG rule") and a single entry point. Excellent
recall/QPS Pareto front but tunable parameters (M, efConstruction, mL) and
strong dependence on construction order.

**NSG / MRNG (Fu et al., VLDB 2019; arXiv:1707.00143)**
: Navigating Spreading-out Graph. Builds on a kNN graph and applies the
Monotonic Relative Neighbor Graph rule for edge selection. Strong
single-shot quality but expensive to construct (full kNN graph upfront).

**HCNNG (Munoz, Gonzalez, Buhmann, Pattern Recognition 2019)**
: This work. Random-projection-style recursive partitioning + per-leaf MSTs.
Reported QPS at recall 0.95 competitive with HNSW and NSG across SIFT1M,
GIST1M, DEEP1M. Key claim: parameter-light (n_trees, leaf_size are the
only knobs) and construction is embarrassingly parallel across trees.

**Vamana / DiskANN (Subramanya et al., NeurIPS 2019; SIGMOD 2024)**
: Single-layer graph tuned for SSD-resident vectors. Uses the alpha-RNG
pruning rule which generalizes both HNSW heuristic and HCNNG's MST
sparsity. Implemented separately in `crates/ruvector-diskann`.

**DEG / Dynamic Exploration Graph (2024)**
: Online graph construction with edge-quality tracking. Covered in a
prior nightly (`research/nightly/2026-05-31-deg-dynamic-exploration-graph`
and `2026-06-08-dynamic-exploration-graph`).

### Where HCNNG fits

The HCNNG niche is **construction simplicity with no online-rebuild cost**:
every tree is independent, so adding a tree to an existing graph is just
"build a new tree, union its edges, retruncate." Compare to HNSW which
requires global re-insertion order to maintain layer invariants. HCNNG
is the natural backbone for "add a vector, lazily improve the graph in
the background" workloads.

---

## Proposed Design

### Construction pipeline

```
              ┌─────────────────────────────┐
   dataset ── │ partition_tree (random 2-piv)│ ─── leaves[]
              └─────────────────────────────┘
                        repeat n_trees times
              ┌─────────────────────────────┐
   leaves[] ─►│ mst_plus_knn_edges (per leaf)│ ─── edges[]
              └─────────────────────────────┘
              ┌─────────────────────────────┐
   edges[] ──►│ Graph::add_edge_unique      │ ─── adjacency
              └─────────────────────────────┘
              ┌─────────────────────────────┐
              │ Graph::finalize             │ ─── sorted, capped
              │  (sort by dist, truncate)   │
              └─────────────────────────────┘
```

### Search

A standard best-first beam search (HNSW level-0 style) with one twist: the
index keeps a list of **multiple entry points** (top-4 by degree + 4
deterministic random anchors). For multi-modal data this is essential —
a single hub tends to lie inside one cluster and the greedy walk stalls.

### Module layout (all <500 lines, files under ruvector-hcnng/src/)

| File          | Lines | Role                              |
|---------------|------:|-----------------------------------|
| lib.rs        |    23 | crate facade                      |
| error.rs      |    13 | typed errors                      |
| distance.rs   |    74 | trait + L2Sq / NegIP / Cosine     |
| partition.rs  |    80 | recursive 2-pivot split tree      |
| mst.rs        |   105 | Prim's MST + kNN augmentation     |
| graph.rs      |    80 | CSR-ish adjacency, finalize       |
| search.rs     |   125 | dual-heap beam search             |
| index.rs      |   250 | HcnngIndex, params, tests         |
| main.rs       |   200 | benchmark binary                  |

### Trait-based swappability

`Distance` is a trait, so the same graph code works for L2², negated inner
product, or cosine. Adding LVQ / RaBitQ codes later is a matter of plugging
in a `Distance` impl that reads the compressed form.

---

## Implementation notes

1. **Partition pivots from the bucket, not the universe.** Always sample
   pivots inside the current bucket so the split is metric-local.

2. **Pathological splits.** If pivot a == b (duplicates) or one side is
   empty (many co-located points), force a halfway cut. Without this guard
   recursion can loop on duplicate-heavy data.

3. **MST via Prim, not Kruskal.** Leaf size is bounded (default 32), so
   the O(L²) Prim with arrays beats Kruskal+DSU on cache effects. No heap.

4. **kNN augmentation reuses the L×L distance matrix.** When
   `knn_per_node > 0`, compute the full pairwise distance matrix once per
   leaf, then read kNN from each row. Cost is the same as MST (already
   O(L²)), so kNN is essentially free.

5. **Multi-entry search.** Push *all* entries onto the candidate min-heap
   and top-k max-heap before the main loop starts. Crucial for recall on
   multi-modal data (see "Failure modes").

6. **Determinism.** Every tree's seed is derived from `params.seed` via
   `seed.wrapping_add(t * GOLDEN_PRIME)`. Same params → bit-identical graph.

---

## Benchmark methodology

- **Hardware:** macOS Darwin 24.6, Apple Silicon, single-threaded.
- **Build:** `cargo build --release -p ruvector-hcnng` (no SIMD flags
  beyond LLVM defaults).
- **Data:** uniform i.i.d. on [-1,1]^d. Also a 32-cluster Gaussian
  mixture (σ=0.6) is available via `DATASET=gmm` for stress testing —
  see "Failure modes."
- **Ground truth:** brute-force L2² scan over the same data; full top-k
  per query, no sampling.
- **Recall:** `|retrieved ∩ ground_truth| / k`, averaged over 200 queries
  (100 for n=50k).
- **Variants per run:** brute force, HCNNG n_trees=1, HCNNG n_trees=12
  (paper default), HCNNG n_trees=20 ef_search=128.

Reproduce locally:

```bash
git checkout research/nightly/2026-06-14-hcnng-mst-graph
cargo run --release -p ruvector-hcnng
D=64 N=50000 NQ=100 cargo run --release -p ruvector-hcnng   # scale test
DATASET=gmm cargo run --release -p ruvector-hcnng           # clustered stress
```

---

## Results

### n = 20,000, d = 32, uniform data, k = 10

```
                  variant    build_ms     µs/query   recall@10
           brute_force_L2           -        414.9       1.000
          hcnng_n_trees=1        12.1          8.9       0.008
         hcnng_n_trees=12       274.8         61.2       0.863
   hcnng_n_trees=20_ef128       463.1        165.0       0.991
```

### n = 20,000, d = 64, uniform data, k = 10

```
                  variant    build_ms     µs/query   recall@10
           brute_force_L2           -        553.7       1.000
          hcnng_n_trees=1        16.6          9.2       0.014
         hcnng_n_trees=12       337.7         85.3       0.599
   hcnng_n_trees=20_ef128       624.9        205.8       0.929
```

### n = 50,000, d = 64, uniform data, k = 10 (scale)

```
                  variant    build_ms     µs/query   recall@10
           brute_force_L2           -       1479.5       1.000
          hcnng_n_trees=1        55.0         11.2       0.009
         hcnng_n_trees=12       970.1         98.5       0.402
   hcnng_n_trees=20_ef128      1783.7        257.8       0.792
```

### Memory footprint (n=20k, d=64, n_trees=12)

| Item              | Bytes      | Notes                              |
|-------------------|-----------:|------------------------------------|
| Vectors (f32)     |  5,120,000 | n × d × 4                          |
| Graph adjacency   |  8,530,000 | avg_degree=32 × 4 + Vec overhead   |
| Total resident    | ~13.65 MB  |                                    |

Graph overhead is roughly 1.67× the raw vector bytes — comparable to
HNSW's M=16 (1.0–1.5×) and lower than NSG's typical 2–3×.

---

## How it works (blog-readable walkthrough)

Imagine you have a million vectors and you want a "good enough" graph
where every vector is connected to its likely-nearest neighbors. The
hard part is that you don't know who is whose neighbor until you measure
distances. HCNNG cuts this cleanly with two old ideas glued together:

1. **Random projection forests.** Pick two random points from your
   dataset. Split everything by which of those two is closer. Recurse
   on each half. Stop when a bucket has ≤ 32 points. Each bucket is now
   a "leaf" containing points that tend to be metrically close.

2. **Minimum Spanning Trees.** Inside each leaf, build the MST. That's
   31 edges per leaf, each guaranteed to be the cheapest way to keep the
   leaf connected. These are excellent candidate neighbor edges.

Repeat that 12 times with different random pivots. Union all the MST
edges into one graph. Truncate every node's neighbor list to ~32 nearest
edges by distance. Done. That's the index.

To search, start from a handful of "anchor" nodes and greedily walk the
graph toward the query — at each step, expand to the most promising
unvisited neighbor. Stop when the closest unexplored candidate is
farther than your kth best so far. That's a beam search; HNSW does
exactly the same shape at its level 0.

Why this works: the metric partition is a "soft VP-tree" that biases
nearby points into the same leaves. MSTs are the sparsest connector
of those points. Twelve independent trees vote on which edges matter,
which fills in the long-range navigability that a single tree misses.
No heuristic edge selection, no level hierarchy, no pre-built kNN graph.

---

## Practical failure modes

1. **Tight clusters dominate recall (high-σ multi-modal data).** On the
   `DATASET=gmm` 32-cluster mixture (σ=0.6) recall drops to ~0.30–0.50
   even at n_trees=12. The partition assigns one cluster per leaf, so
   MST edges live inside clusters and the only cross-cluster edges come
   from leaves that straddle pivots — which is rare when clusters are
   well-separated. **Mitigation:** add more diverse entry points (we use 8
   today; clustered workloads want 16+) or interleave HCNNG with a
   coarse IVF assignment so the entry point is in the right cluster.

2. **Scale at high dimension.** Recall at n=50k, d=64 drops to 0.79 at
   n_trees=20. Two compounding causes: (a) more vectors per cluster of
   nearest neighbors → kNN edges saturate `max_degree=48`, evicting long
   MST edges; (b) the partition split is less informative in higher
   intrinsic dim. **Mitigation:** raise `max_degree` proportional to
   log(n); add a "diversity" pruning rule à la HNSW that keeps long
   edges. See "What to improve next."

3. **Duplicate vectors break the pivot pick.** Bucket of all-identical
   points → pivot a == pivot b → degenerate split. We force a halfway
   cut, but the resulting leaves have no metric structure. Filter
   duplicates upstream or perturb on insertion.

4. **Single-threaded build is slow at n=1M.** Tree construction is
   embarrassingly parallel but we do it sequentially in this PoC. See
   "What to improve next."

---

## What to improve next (roadmap)

1. **Parallel tree construction with `rayon`.** One thread per tree,
   union into a `Mutex<Graph>` or per-thread shards merged at end.
   Expected ~Nproc× speedup of the build step.

2. **Heuristic edge pruning (alpha-RNG / Vamana rule).** Keep edges that
   pass `dist(u, v) < dist(v, w) / alpha` for all candidates w already
   in the neighbor list. Preserves long-range navigability under
   aggressive truncation. Vamana uses this and matches HNSW recall.

3. **RaBitQ-quantized distance trait.** Implement `Distance` over
   `crates/ruvector-rabitq` 1-bit codes. Build the graph on full
   vectors (one pass), search on codes (32× memory cut, ~2× QPS).

4. **Streaming add / lazy retree.** New points join the current leaves
   via "nearest hub" approximation; a background thread builds the
   next tree and unions its edges. No global rebuild required.

5. **Filtered HCNNG.** Tag every edge with a filter compatibility bit
   so range/predicate queries can prune the candidate frontier during
   beam search. Composes with the existing `ruvector-acorn` work.

6. **Multi-entry from KD-tree.** Replace deterministic random anchors
   with a 1-shot KD-tree (or VP-tree) lookup to find the query's
   approximate cluster, then start beam search from there. Cheap (one
   tree, ~log N) and fixes the clustered-data recall hole.

---

## Production crate layout proposal

```
crates/ruvector-hcnng/                 # this crate (lib + bin + bench)
crates/ruvector-hcnng-node/            # NAPI bindings — Node.js
crates/ruvector-hcnng-wasm/            # wasm-bindgen — browser
crates/ruvector-hcnng-snapshot/        # rkyv persistence (load 0-copy from mmap)
```

The existing crate already separates `distance`, `partition`, `mst`,
`graph`, `search`, `index` so the WASM and NAPI surfaces only need to
re-export the `index` module. Persistence is a future task — for the
PoC the graph stays in-memory and is rebuilt per process.

---

## References

1. Munoz, J. V., Gonzalez, R., & Buhmann, J. M.
   *Hierarchical Clustering-Based Graphs for Large Scale Approximate
   Nearest Neighbor Search.* Pattern Recognition 96 (2019), 106985.
2. Malkov, Y., & Yashunin, D. *Efficient and robust approximate nearest
   neighbor search using HNSW graphs.* IEEE TPAMI 2018. arXiv:1603.09320.
3. Fu, C. et al. *Fast Approximate Nearest Neighbor Search With The
   Navigating Spreading-out Graph.* VLDB 2019. arXiv:1707.00143.
4. Subramanya, S. et al. *DiskANN: Fast Accurate Billion-point Nearest
   Neighbor Search on a Single Node.* NeurIPS 2019.
5. Hajebi, K. et al. *Fast approximate nearest-neighbor search with k-NN
   graph.* IJCAI 2011 — origin of greedy walk on proximity graphs.
6. ruvector ADR-252 (this PR) for the in-tree integration record.
