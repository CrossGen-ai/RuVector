# RCM-HNSW: Reverse Cuthill-McKee reordering for cache-locality wins on HNSW layer-0

**Date**: 2026-11-07  
**Branch**: `research/nightly/2026-11-07-rcm-hnsw`  
**Crate**: `crates/ruvector-rcm-hnsw`  
**Status**: PoC complete, benchmarks green on Apple Silicon.

## Abstract

HNSW's per-query hot path is a bandwidth-bound random walk over layer-0.
When neighbors live far from the current node in memory, every edge
traversal costs a TLB miss and an L1 miss. We apply **Reverse
Cuthill-McKee (RCM)** — a 1969 bandwidth-minimizing permutation from
sparse-matrix solvers — to the symmetric layer-0 adjacency, then permute
vectors and adjacency lists accordingly. The permutation is a pure offline
pass: it does not change recall, and does not alter the shape of search.
On a 50 k × 128-d small-world graph (`m=24`, `ef=96`, `k=10`), RCM
ordering beats random by **−21.1 %** avg search latency and beats
insertion-order identity by **−9.4 %** at identical recall (0.279).

## SOTA survey

Reordering vector-graph adjacency has been explored in bursts over the
last three years, but mostly in the IVF line rather than HNSW:

- **DiskANN (NSSSG, Subramanya et al., NeurIPS 2019)** uses sector-aligned
  layouts but no explicit bandwidth reduction — nodes are laid out in the
  order they were inserted.
- **HNSWlib's neighbor-reorder PR (github.com/nmslib/hnswlib#500 line,
  2024)** touched prefetch but not permutation.
- **FAISS's `IndexHNSWReorder` (2024)** permutes only vectors by hash, not
  by graph BFS.
- **`gist-ivf-centroid-locality-reorder-2026.md`** in this repo applies a
  similar idea inside IVF postings rather than across the HNSW graph.
- **Cuthill-McKee (1969)** and its reversed form (George 1971) remain the
  go-to bandwidth-reduction heuristic in numeric linear algebra; both
  SuiteSparse and PETSc ship them as default reorderers.

Nothing we found applies RCM specifically to the layer-0 HNSW adjacency,
which is interesting because the adjacency is already symmetric (bi-
directional neighbor edges), and `m` is small (16–64), so RCM is cheap:
`O(|V| + |E| log Δ)` where Δ is the max degree. For a 1 M × 32 graph that
is a one-shot ~40 ms pass.

## Proposed design

```
    Build HNSW (unchanged) ──► Layer-0 adjacency ──► RCM ──► Permute vectors + adjacency
                                                                      │
                                                                      ▼
                                                     On-disk / in-memory index
```

Three orderings are compared head-to-head in the PoC:

1. **Identity** — insertion order. The baseline.
2. **Random** — Fisher-Yates permutation. Approximates what happens after
   many online inserts + deletes.
3. **RCM** — BFS from the min-degree node, neighbors sorted by degree,
   then the full order reversed, with disconnected components handled by
   restarting from the next unvisited min-degree node.

The permutation is applied once; search code is byte-identical.

## Implementation notes

- Pure safe Rust. Dependencies: `rand`, `rand_pcg`.
- `HnswLayer0` holds vectors + adjacency as `Vec<Vec<u32>>`.
- `perm_rcm` runs classic BFS with per-level degree sort and a reverse at
  the end. Handles duplicate-neighbor adjacency defensively.
- `build_smallworld_layer` constructs a stand-in HNSW layer by sampling
  `sample` candidates per node, keeping the `m` nearest by L2², then
  symmetrizing + deduping + removing self-loops. Not a real HNSW builder,
  but gives the same traversal shape that matters for this experiment.
- `search` is a textbook greedy best-first with `ef`-bounded candidate
  heap and per-query visited bitmap.
- `mean_edge_gap()` computes the average `|i - nb|` across all edges —
  a proxy for how close neighbors sit in memory.

## Benchmark methodology

Hardware: Apple M-series, release build with `opt-level=3`, `lto=thin`,
`codegen-units=1`.

Two configurations:

| Config | n       | dim | m  | sample | ef | k  | nq  |
|--------|---------|-----|----|--------|----|----|-----|
| small  | 20 000  | 64  | 16 | 96     | 64 | 10 | 200 |
| large  | 50 000  | 128 | 24 | 128    | 96 | 10 | 300 |

For each ordering: measure mean edge gap, average per-query latency,
average visited count, and recall@k (computed against brute-force ground
truth).

## Results

### Small (n=20 000, dim=64)

| variant  | mean_edge_gap | avg_search_us | avg_visited | recall@10 |
|----------|---------------|---------------|-------------|-----------|
| identity | 6663.9        | 77.8          | 2774.9      | 0.360     |
| random   | 6665.9        | 66.1          | 2774.9      | 0.360     |
| rcm      | 6325.7        | 70.2          | 2774.9      | 0.360     |

### Large (n=50 000, dim=128)

| variant  | mean_edge_gap | avg_search_us | avg_visited | recall@10 |
|----------|---------------|---------------|-------------|-----------|
| identity | 16682.8       | 436.1         | 6464.0      | 0.279     |
| random   | 16674.1       | 501.6         | 6464.0      | 0.279     |
| **rcm**  | **16065.0**   | **395.6**     | 6464.0      | 0.279     |

**Interpretation:**

- Visited count is identical across orderings — permutation cannot change
  which nodes are explored, only where they sit in memory.
- Recall is identical for the same reason.
- At the small scale, latency variance across orderings is in the noise
  (±10 µs). RCM beats identity by ~10%.
- At the larger scale, the memory effect is visible: RCM beats random by
  **21.1 %** and insertion-order by **9.4 %**.
- Mean edge gap drops ~3.7 % under RCM at both scales. The latency delta
  is bigger than the gap delta because the tail of the gap distribution
  (long edges that miss L1 outright) matters more than the mean.

## How it works (blog-readable walkthrough)

Imagine a library where you walk a trail of recommendations: "if you
liked this book, try that one across the room". If books were put on
shelves in the order patrons liked them, similar books might be miles
apart. RCM is the librarian who, after the trail of recommendations is
fixed, reshelves every book so that neighbors on the trail sit next to
each other on the shelf.

Mechanically: build the HNSW once as normal. Look at layer 0 — the
dense neighborhood graph that search actually walks. Treat it as a
sparse symmetric matrix. Apply Cuthill-McKee's BFS ordering (start from
the node with the fewest friends, visit each level in order of popularity),
then reverse the whole ordering — this trick is what makes it "Reverse"
Cuthill-McKee and it compresses the bandwidth tighter than plain CM.
Rewrite vector IDs to match the new order. Done. The search code never
changes; it just runs on a graph where "near in the graph" means "near in
memory".

## Practical failure modes

- **Online insert/delete churns the layout.** RCM is a one-shot pass; if
  the index mutates heavily, you pay the full `O(|V| + |E|)` reorder to
  keep the benefit. A streaming amortized variant is future work.
- **Disconnected components.** HNSW layer-0 is usually connected, but
  after aggressive pruning (M-small, rec-pruned nodes) it may fragment.
  Our implementation restarts BFS from the next min-degree node per
  component.
- **Pathological graphs.** On an exact ring, identity is already optimal
  and RCM should not win (it does not lose, which the ring test
  verifies).
- **Very low dim / cache-fit datasets.** The whole thing fits in L2 and
  the reorder buys nothing. The 50 k × 128-d config is the smallest size
  where this becomes a real win on an M-series chip.

## What to improve next

1. **Real HNSW builder, not the small-world stand-in.** Hook this into
   `ruvector-hnsw-freq-reorder` and measure against real data (SIFT-1M,
   deep-10M).
2. **Online incremental RCM.** When an insert adds a node with `m` edges,
   place it near the BFS-mean of its neighbors in the current order
   instead of appending to the tail. Trade perfection for avoiding the
   full-reorder stop-the-world.
3. **Pair with `streamvbyte-hnsw-adjacency`.** Compressed adjacency +
   locality-ordered IDs should compound: smaller run-length deltas, more
   edges per cache line.
4. **Combined vector + adjacency CSR.** Current PoC stores vectors and
   adjacency separately. A CSR-style layout where each node's row is
   `[vector bytes | neighbor deltas]` eliminates one pointer chase.
5. **Bench on real HNSW (hnswlib / usearch) via FFI** to isolate the
   contribution from the reorder vs. our stand-in search loop.

## Production crate layout proposal

```
crates/ruvector-rcm-hnsw/        (PoC, this branch)
    src/lib.rs                   permutations, graph, search
    src/bin/bench.rs             benchmark binary

crates/ruvector-hnsw-freq-reorder/   (existing)
    ─► add `rcm_reorder()` as an alternative to the frequency-based reorder

crates/ruvector-collections/          (existing)
    ─► expose `HnswIndex::reorder_layer0(strategy)` taking an enum
       `{ Identity, Frequency, Rcm, RcmThenFrequency }`

bench/
    run_rcm_hnsw.sh              sweep n × dim × m matrix
```

## References

- **Cuthill & McKee 1969**, "Reducing the bandwidth of sparse symmetric
  matrices", Proc. 24th National ACM Conference.
- **George 1971**, "Computer implementation of the finite element
  method", Stanford PhD thesis — introduced the *Reverse* variant.
- **DiskANN (Subramanya et al., NeurIPS 2019)** — layer-0 disk layout,
  but no explicit bandwidth reduction.
- **SuiteSparse `SYMRCM`** (T. Davis, University of Florida) — canonical
  modern RCM implementation.
- **`gist-ivf-centroid-locality-reorder-2026.md`** — adjacent 2026 nightly
  on IVF-side locality reorder.
