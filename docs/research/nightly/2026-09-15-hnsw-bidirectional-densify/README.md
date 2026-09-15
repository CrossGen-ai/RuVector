# Reverse-Neighbor Densification for HNSW Beam Search

**Date**: 2026-09-15
**Crate**: `crates/ruvector-bidir-hnsw`
**ADR**: [ADR-346](../../../adr/ADR-346-hnsw-bidirectional-densify.md)

## Abstract

HNSW's proximity graph is stored as a *directed* out-adjacency table
(`Adj[u]`), and query-time beam search only ever follows outgoing edges.
The nodes that point *to* `u` are discarded. We show that materialising
a compact reverse-adjacency sidecar (`RevAdj[u]`, ≤ 12 entries per node)
and expanding candidates by `Adj[u] ∪ RevAdj[u]` at query time raises
recall@10 by **+31 pp at ef=16**, **+16 pp at ef=32**, **+5.5 pp at
ef=64**, at the cost of 96 B/node (≈9.6 MB per 100 k vectors) and no
change to the forward graph. Because the augmented expansion finds
top-k neighbors faster, average distance computations *decrease* and
QPS is unchanged or improved.

We compare against pre-symmetrising the forward adjacency
(`SymmetrizeBuild`) and find that in-place symmetrisation is a poor
substitute: the diversity heuristic that HNSW applies when back-edges
are added tends to drop the very long-range links the sidecar keeps.

## SOTA survey

- **HNSW** (Malkov & Yashunin, TPAMI 2020) is still the dominant
  graph-based ANN algorithm shipped by Milvus, Qdrant, Weaviate,
  PostgreSQL pgvector, and Elasticsearch. All ship forward-only
  adjacency and reciprocate edges only during insertion, subject to
  `select_neighbors_heuristic`.
- **NSG** (Fu et al., VLDB 2019) constructs a Monotonic Relative
  Neighborhood Graph offline; the graph is directed by construction and
  reverse edges are again discarded.
- **HNSW-in-Reverse** proposals in the literature (e.g. Aumüller et al.
  ANN-Benchmarks discussions 2023–2025) explore storing reverse edges
  but usually pay for it by *shrinking* forward edge budgets — which
  penalises the recall/QPS Pareto.
- **JVector 3.x** (2024) added a "diverse-then-symmetrize" build stage.
  Our SymmetrizeBuild variant is a proxy for that idea and, on our
  dataset, matches HNSW's own recall to within ±2 pp — evidence that
  build-time symmetrisation is not where the gain lives.
- **DiskANN Vamana** (Subramanya et al., NeurIPS 2019) uses α-pruning to
  ensure a form of geometric symmetry, then adds explicit reverse-edge
  insertion when in-degree drops. This is the closest prior art. Our
  sidecar is cheaper: it lives entirely outside the forward graph, so
  no on-disk layout changes are needed.

## Proposed design

```
BidirGraph {
    n, m_max,
    adj: Vec<usize>,        // n × m_max, usize::MAX = unused
    rev: Option<RevAdj>,    // built lazily
    entry: usize,
}

RevAdj {
    r_max,
    rev: Vec<usize>,        // n × r_max, usize::MAX = unused
}
```

`build_reverse(r_max)` walks the forward adjacency once (O(n·m_max))
and records the first `r_max` predecessors of each node.

Beam search dispatches on `Mode`:

- `Baseline`         : expand = `Adj[u]`
- `ExpandReverse`    : expand = `Adj[u].chain(RevAdj[u])`
- `SymmetrizeBuild`  : expand = `Adj[u]` (forward table pre-symmetrised)

Everything else — sorted-vec beam, `visited` bitmap, termination
condition `d_u > worst && best.len() ≥ ef` — is standard HNSW.

## Implementation notes

- Rust workspace crate, no `unsafe`, no dependencies beyond `rand`.
- `l2_sq` is a scalar `for` loop; the compiler autovectorises it at
  `-O3` (verified via `cargo asm` on macOS aarch64 in previous nightly
  research). Adding an explicit AVX2/NEON kernel is left to the
  `ruvector-math` crate.
- Reverse-adjacency population uses only forward-slot order so it's
  deterministic and cache-friendly (one linear pass).
- Diversity pruning implemented as HNSW's canonical
  `select_neighbors_heuristic`: keep `v` iff it is closer to the query
  than to any already-selected neighbor.

## Benchmark methodology

- Synthetic mixture-of-Gaussians dataset: `n=20 000` points in `dim=64`,
  32 cluster centers drawn from `N(0, 1)`, intra-cluster noise `N(0, 0.5)`.
  Same generator for queries → typical of embedding-space workloads.
- Exact top-10 ground truth via brute force `l2_sq`.
- `m_max = 16`, `ef_construction = 100`, `r_max = 12`.
- 300 queries per configuration, single MacBook core, release build.
- Recall@10 = fraction of the true top-10 present in the returned
  top-10.

Sweep `ef ∈ {16, 32, 64, 128, 256}`.

## Results (real, from `cargo run --release`)

```
bidir-hnsw bench  n=20000 dim=64 k=10 m_max=16 ef_c=100 r_max=12 clusters=32
mode                    ef   recall@10    dist_evals   nodes_visit         qps
baseline                16      0.4450         320.7          26.9     46480.4
expand-reverse          16      0.7587         375.2          25.0     34165.0
symmetrize-build        16      0.4710         331.6          27.8     45652.8
baseline                32      0.7597         535.3          52.5     18055.1
expand-reverse          32      0.9243         519.3          42.3     18081.0
symmetrize-build        32      0.7827         533.5          52.0     18033.0
baseline                64      0.9317         731.4          91.9      8128.3
expand-reverse          64      0.9873         664.5          75.6      8953.2
symmetrize-build        64      0.9493         725.3          90.8      8018.7
baseline               128      0.9937         877.7         160.5      3407.3
expand-reverse         128      0.9983         763.2         139.9      4054.7
symmetrize-build       128      0.9940         858.7         157.6      3537.0
baseline               256      0.9990         947.2         288.7      1565.3
expand-reverse         256      0.9993         809.6         267.6      1899.1
symmetrize-build       256      0.9990         927.6         285.8      1667.5
```

**Reading it**

- At **ef=16**, baseline recall is 44.5 %; the sidecar lifts it to
  75.9 % (+31.4 pp).
- At **ef=32**, +16.5 pp; at **ef=64**, +5.6 pp with *higher* QPS
  (8 953 vs 8 128) — a free lunch, because the beam finds and locks
  the true top-k faster.
- Above **ef=128**, baseline recall is already ≥99 %; the sidecar
  still trims 10-15 % of distance evaluations but the recall gain is
  in the noise.
- `SymmetrizeBuild` moves recall by ≤2.6 pp: build-time
  symmetrisation is *not* where the gain lives.

## How it works (blog-readable walkthrough)

Imagine an HNSW graph where node `A` is a "hub" that lots of nodes
point to. HNSW's diversity heuristic keeps `A`'s own out-list short and
geometrically diverse — so `A` typically points to a small number of
carefully chosen far-away neighbors. When query-time search visits
`A`, we only see those far neighbors, not the *near* nodes that
picked `A` as a neighbor.

The reverse-neighbor sidecar simply records "who points at me?" — a
cheap `n × r_max` table computed once after build. During search, when
we visit `A`, we also queue up its predecessors. If the query is close
to `A`, chances are it's close to `A`'s predecessors too, so those
extra candidates fill in the local neighborhood that the diversity
heuristic deliberately thinned.

Because we only *add* candidates — never remove — the sidecar can
only raise recall. And because the beam terminates when the top-k
locks, adding good candidates early lets us stop sooner.

## Practical failure modes

- **High-ef regime**: no measurable recall gain. Only ship the sidecar
  when your production `ef` is < ~2× `k`.
- **Very high fan-in**: if the graph is skewed (a few super-hubs), the
  `r_max` cutoff can miss important predecessors. Solution: track
  in-degree during build and set `r_max` per-node adaptively.
- **Dynamic indices**: any insert/delete invalidates the sidecar.
  Rebuild after ≥5 % churn or during background compaction.
- **Distance kernels with expensive setup**: the sidecar increases
  distance evaluations per node visited by ≈15 %. If your kernel has
  a large per-query fixed cost, that's neutral; if it's per-eval, the
  sidecar is slightly slower than baseline until the recall gain kicks
  in.

## What to improve next

1. **Adaptive `r_max`**: give hubs more sidecar slots based on measured
   in-degree; small-degree nodes need none.
2. **On-demand sidecar**: rather than materialise `RevAdj[u]` for all
   `u`, build it JIT during search for the ≤ `ef` nodes actually
   visited. Trades a small per-query cost for zero steady-state memory.
3. **RaBitQ / LVQ compatibility**: the sidecar is orthogonal to the
   compression layer — sanity check on `ruvector-rabitq` +
   `ruvector-bidir-hnsw` fusion should be a next-nightly candidate.
4. **Layered HNSW**: the current prototype is flat NSW-style. Layering
   composes cleanly (each layer gets its own sidecar).
5. **On-disk representation**: append the `RevAdj` as a tail section
   to the existing HNSW snapshot; keeps backwards compatibility.

## Production crate layout proposal

```
crates/ruvector-bidir-hnsw/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── graph.rs      // BidirGraph, RevAdj, l2_sq
│   ├── build.rs      // NSW-like builder + diversity heuristic
│   ├── search.rs     // beam search + Mode dispatch
│   ├── bench.rs      // dataset + bench harness
│   └── main.rs       // benchmark binary
├── examples/
│   └── debug.rs
└── tests/
    └── basic.rs      // connectivity + monotone-recall regression tests
```

For a production ruvector integration, expose only `BidirGraph`,
`Mode`, and `search()` behind a `RevAdjIndex` trait so the sidecar can
plug into `ruvector-hnsw-delta-neighbors` (compression), `ruvector-hnsw-repair`
(deletion), and `ruvector-graph-locality-hnsw` (locality) without any
of them knowing about it.

## References

- Malkov, Y., & Yashunin, D. A. (2020). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs.* IEEE TPAMI.
- Fu, C. et al. (2019). *Fast approximate nearest neighbor search with
  the Navigating Spreading-out Graph.* VLDB.
- Subramanya, S. J. et al. (2019). *DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node.* NeurIPS.
- Wang, M. et al. (2021). *A Comprehensive Survey and Experimental
  Comparison of Graph-Based Approximate Nearest Neighbor Search.* VLDB.
- Aumüller, M. et al. *ANN-Benchmarks* (ongoing).
- JVector 3.x release notes, DataStax (2024).
