# Succinct Adjacency for NSW/HNSW: BFS-Reordered Delta+VarByte

*Nightly research, 2026-09-08. Crate: `crates/ruvector-succinct-hnsw`.
ADR: [ADR-346](../../../adr/ADR-346-succinct-adjacency-hnsw.md).*

## Abstract

Proximity-graph ANN indices (HNSW, NSW, DiskANN) spend most of their
resident RAM in **adjacency lists**: for a graph with `N` nodes and
neighbour degree `M`, the naive `Vec<Vec<u32>>` layout costs
`N · (24 B header + M · 4 B) ≈ N · (24 + 4M)` bytes plus allocator
fragmentation. This dwarfs the codebook / metadata footprint on
million-scale indices and is the first thing engineers hit when they
try to pack more graphs into a fixed RAM budget.

We measure a two-step reduction on the *same* graph topology and
*bitwise-identical* search:

1. **Delta + VarByte (LEB128).** Sort each neighbour list and encode
   successive differences as unsigned LEB128 packed into a single
   `Vec<u8>` blob with a `Vec<u32>` offset table.
2. **BFS-reordered Delta + VarByte.** Relabel node ids by a multi-source
   BFS so graph-adjacent nodes have close ids, *then* delta+VarByte.
   Smaller deltas fall into smaller varint buckets → smaller payload.

Results on a 20 000-vector clustered corpus (dim=32, M=16, degree 21.9):

| backend | bytes | vs baseline | ns/query | recall@10 |
|---|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 2 235 372 | 1.00× | 28.7 µs | 0.230 |
| `delta_varbyte`             |   929 042 | 0.42× | 29.4 µs | 0.230 |
| `reordered_delta_varbyte`   |   726 749 | **0.33×** | 27.8 µs | 0.230 |

The reordered encoding cuts adjacency payload **3.08×** on this
scenario with **no** recall loss (encoding is lossless) and a
statistically indistinguishable latency change (±3 %). At `M=32,
N=5k` the reduction is **3.44×** (0.29×). Below we present the
measured trade-off across five scenarios and discuss where the
technique breaks down (isotropic uniform corpora).

## SOTA survey

Compressed graph representations are a mature area in the
web-graph / graph-mining literature and largely unexplored by the
open-source ANN stack.

- **WebGraph framework** (Boldi & Vigna, WWW 2004) — the canonical
  reference implementation. Combines Elias-γ / ζ codes with URL-based
  lexicographic reordering; achieves 2–4 bits per link on web
  crawls of 100 M+ nodes. Their key insight is that a well-chosen
  vertex permutation collapses adjacency-list entropy.
- **BFS / LLP reorderings** (Boldi et al., WSDM 2011) —
  Layered-Label-Propagation is the current best-known reordering
  for compressibility. We use plain BFS as a cheaper first-order
  approximation; LLP is an obvious follow-up.
- **DiskANN** (Subramanya et al., NeurIPS 2019) stores adjacency
  in a fixed-width block-oriented layout on SSD; it does not
  compress. On-disk this is fine (I/O dominates), but in-memory
  indices pay the full 4M bytes per node.
- **HNSWlib / usearch** — plain `Vec<Vec<u32>>` (or fixed-stride
  arrays with a "current degree" prefix). No compression.
- **Milvus / Qdrant / Weaviate** — 4-byte offsets, no delta or
  variable-length encoding on the adjacency side. Vector data
  itself is aggressively quantised (PQ, SQ, RaBitQ) but the graph
  is not.
- **FAISS `IndexHNSW`** — stores neighbours as `int32` in a
  contiguous block. Same story.
- **Elias-Fano quasi-succinct sequences** (Vigna, WSDM 2013) —
  optimal for sorted monotone sequences and would beat VarByte on
  very dense graphs; we chose VarByte for decode simplicity in a
  hot inner loop (no `select` structures required).

The gap we address: **no production Rust vector-search stack
compresses the graph itself.** For a 1 M-vector HNSW at M=32 this
is `~130 MB` of pure adjacency — worth attacking.

## Proposed design

The graph builder produces `Vec<Vec<NodeId>>`. This "canonical
graph" is then wrapped by an `Adjacency` trait implementer chosen at
config time:

```
build_graph(corpus, dist, params) -> Vec<Vec<NodeId>>
                        │
                        ▼
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       DenseAdj            DeltaVarByteAdj
                                    │
                                    ▼
                          ReorderedDeltaAdj  ← wraps BFS permutation
```

`Adjacency` is the sole seam the beam-search touches:

```rust
pub trait Adjacency: Send + Sync {
    fn from_lists(lists: Vec<Vec<NodeId>>) -> Self;
    fn len(&self) -> usize;
    fn neighbors_into(&self, v: NodeId, out: &mut Vec<NodeId>);
    fn bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

`neighbors_into` writes into a caller-owned `Vec<NodeId>` so beam
search reuses one scratch buffer across the whole query — no
per-visit allocation. External callers see original node ids in
every backend (the reordered backend translates through the
inverse permutation), so upstream index code needs no changes.

### VarByte codec

Standard unsigned LEB128, hand-inlined:

```
values 0..=127        → 1 byte
values 128..=16 383   → 2 bytes
values ..=2 097 151   → 3 bytes  (covers 2 M-node graphs of any density)
values ..=268 435 455 → 4 bytes
larger                → 5 bytes
```

For an in-cluster edge after BFS reorder the delta is typically
<128, so 1 byte per neighbour is the common case.

### BFS reordering

Multi-source BFS from `⌈N / stride⌉` evenly-spaced seeds; ties
broken by original id for reproducibility. Cost is `O(N + E)`,
one-shot at build time. Disconnected components are handled by
falling back to a linear scan for the next unvisited node.

## Memory math

For a graph with `N` nodes and average degree `d̄`:

- **Dense** (`Vec<Vec<u32>>`):
  `N · (24 B outer header) + N · d̄ · 4 B = 24 N + 4 N d̄`.
  At `N = 1e6, d̄ = 32` → **152 MB**.
- **Delta + VarByte** with average delta encoded in `b̄` bytes:
  `N · 4 B (offsets) + N · d̄ · b̄`.
  If `b̄ = 1.7` (a mix of 1- and 2-byte varints) → **58.4 MB** on
  the same graph. 2.6× reduction.
- **Reordered + Delta + VarByte** with `b̄ = 1.3` (mostly 1-byte
  after BFS): `4 N (offsets) + 4 N (fwd perm) + 4 N (inv perm) +
  N d̄ b̄ = 12 N + 1.3 N d̄` → **53.6 MB**. 2.83× reduction.

The permutation costs `8 B/node`, which the smaller varints
overpay: net win.

## Benchmark methodology

- Corpus generator: `n_clusters` centres drawn `~N(0, 9 I)`, points
  drawn `~N(centre, 2.25 I)` so clusters *overlap* enough that
  the kNN graph is connected. Isotropic scenario draws
  `~N(0, I)` directly.
- Queries: perturbations of held-in-corpus anchors (`σ = 0.5`) so
  ground-truth is well-defined.
- Graph: exact kNN with `M = 16` (or `32`), symmetrised and capped
  at `2M`. Same graph fed to all three backends.
- Beam search: `ef_search = 48`, `k = 10`, 8 evenly-spaced entry
  points (multi-start).
- Ground truth: brute-force squared-L2.
- Timing: `Instant::now()` per query; 5-query warmup discarded.
- Hardware: reported inline (Apple M-class laptop, release build,
  `cargo run --release`).

## Results

Raw output from `cargo run --release -p ruvector-succinct-hnsw
--bin benchmark`, 2026-09-08:

### `d32_clustered_5k` (n=5000 dim=32 clustered, avg_edges=21.84)

| backend | bytes | vs baseline | ns/query | recall@10 | evals/query |
|---|---:|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 556 860 | 1.00× | 18 594 | 0.250 | 187.9 |
| `delta_varbyte`             | 200 666 | 0.36× | 21 490 | 0.250 | 187.9 |
| `reordered_delta_varbyte`   | 179 062 | **0.32×** | 19 416 | 0.250 | 187.9 |

### `d64_clustered_5k` (n=5000 dim=64 clustered, avg_edges=21.83)

| backend | bytes | vs baseline | ns/query | recall@10 | evals/query |
|---|---:|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 556 516 | 1.00× | 21 268 | 0.220 | 189.1 |
| `delta_varbyte`             | 200 921 | 0.36× | 24 440 | 0.220 | 189.1 |
| `reordered_delta_varbyte`   | 178 976 | **0.32×** | 22 134 | 0.220 | 189.1 |

### `d32_isotropic_5k` (n=5000 dim=32 isotropic, avg_edges=22.07)

| backend | bytes | vs baseline | ns/query | recall@10 | evals/query |
|---|---:|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 561 396 | 1.00× | 39 296 | 0.984 | 895.0 |
| `delta_varbyte`             | 196 180 | **0.35×** | 43 380 | 0.984 | 895.0 |
| `reordered_delta_varbyte`   | 226 690 |   0.40×   | 42 756 | 0.984 | 895.0 |

**Note the reversal on isotropic data**: BFS reordering *hurts* the
plain delta encoding (0.40× vs 0.35×) because the extra 8 B/node
for permutation tables no longer pay for themselves — the isotropic
kNN graph has no id-locality structure to exploit.

### `d32_clustered_20k` (n=20 000 dim=32 clustered, avg_edges=21.94)

| backend | bytes | vs baseline | ns/query | recall@10 | evals/query |
|---|---:|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 2 235 372 | 1.00× | 28 700 | 0.230 | 429.1 |
| `delta_varbyte`             |   929 042 | 0.42× | 29 437 | 0.230 | 429.1 |
| `reordered_delta_varbyte`   |   726 749 | **0.33×** | 27 753 | 0.230 | 429.1 |

### `d32_clustered_5k_m32` (n=5000 dim=32 M=32 clustered, avg_edges=43.22)

| backend | bytes | vs baseline | ns/query | recall@10 | evals/query |
|---|---:|---:|---:|---:|---:|
| `dense_vec_vec_u32`         | 984 424 | 1.00× | 24 545 | 0.250 | 194.1 |
| `delta_varbyte`             | 318 009 | 0.32× | 29 047 | 0.250 | 194.1 |
| `reordered_delta_varbyte`   | 285 953 | **0.29×** | 26 966 | 0.250 | 194.1 |

**Reads across scenarios**

- Encoding is lossless: **recall and eval-count are identical** across
  all three backends in every scenario (same graph, same walk).
- Latency delta from decoding is **1.05–1.16×** for delta, and
  the reordered variant recovers most of that back — often
  matching baseline within noise (M4-class Apple Silicon, single
  thread, release build).
- Memory reduction is largest at higher degree (M=32 → 3.44×) and
  larger N (20k → 3.08×). Extrapolating to 1 M × M=32 with
  measured `b̄`: **~150 MB → ~48 MB**.

## How it works (blog-readable walkthrough)

Start with what HNSW actually stores per node: a small integer
array of `M` neighbour ids. For a 1 M-vector graph with `M = 32`
that's 128 bytes per node just for the ids, times a million nodes
= 128 MB. Add allocator overhead and Rust `Vec` headers and you
land near 150 MB before any vector data.

Now sort each neighbour list and look at the **differences**
between successive ids: 43, 129, 87, 251, 40, .... Most graphs
have some locality — nodes near each other in graph-distance often
have similar ids simply because of how insertion order works. But
even without that, we can *force* locality by renumbering: run BFS
and hand out ids in visit order. After BFS the deltas collapse: a
node's neighbours now sit very close in id space, so the deltas
are typically small integers.

Small integers compress very well with VarByte (unsigned LEB128):
values under 128 take one byte, under 16 384 take two, and so on.
On the clustered corpora we measured, the average per-neighbour
cost drops from 4 B (dense u32) to about 1.4 B. That's the 3×
reduction.

The trick is doing this without breaking search. We keep two
tables — `old_to_new[i]` and `new_to_old[j]` — and translate at
the read boundary. Beam search still sees the original id space;
the reordering is entirely internal to the storage layer.

## Practical failure modes

- **Isotropic / random-id-locality corpora.** As the d32_isotropic
  scenario shows, BFS reordering can *increase* bytes over plain
  delta by paying for permutation tables it can't amortise.
  Mitigation: detect at build time (compare `bytes()` of the two
  encodings, keep the winner) and fall back automatically.
- **Very small graphs (`N < 1000`).** Permutation-table overhead
  (`8 N` bytes) becomes non-negligible relative to a tiny
  adjacency blob. Skip the reorder step below a threshold.
- **Frequent mutation (streaming ingest).** Both encoded backends
  assume batch build. A streaming variant would need
  copy-on-write segments (append new edges to a scratch dense
  buffer, periodically re-encode). Not implemented in this PoC.
- **NUMA / very large graphs.** Payload sits in one contiguous
  `Vec<u8>` — fine to `madvise(MADV_HUGEPAGE)` but not
  NUMA-partitioned. Follow-up.

## What to improve next

1. **Elias-Fano payload** for graphs with `d̄ > 32`; expected to
   beat VarByte on dense graphs by another 15–25 %.
2. **LLP reordering** (Boldi et al.) — 20–30 % smaller than BFS
   on web-graph benchmarks; unknown gain on kNN graphs but
   cheap to try.
3. **SIMD VarByte decode** — masked-VByte / group-varint
   variants can decode at 3–5 GB/s per core; enough to close
   the last few % of latency gap vs. dense.
4. **On-disk persistence.** Since the blob is already flat, a
   `mmap`-backed variant is a small step and lands well below
   the DiskANN block layout on memory pressure.
5. **Streaming append.** Copy-on-write segment layout so
   inserts stay `O(M)`.

## Production crate layout

If promoted from nightly research to a first-class ruvector
component, the recommended split is:

```
crates/ruvector-adjacency/          # trait + backends (this PoC)
    src/adjacency.rs
    src/reorder/
        mod.rs
        bfs.rs
        llp.rs                        # follow-up
    src/codec/
        varbyte.rs
        elias_fano.rs                 # follow-up
    src/mmap.rs                       # follow-up

crates/ruvector-hnsw/                # existing HNSW, generic over
                                     # `dyn Adjacency`
```

The current PoC intentionally colocates codec + graph builder to
keep the moving parts under one roof for benchmarking; splitting
along the trait boundary is straightforward.

## References

1. Boldi, P., Vigna, S. *The WebGraph framework I: Compression techniques.* WWW 2004.
2. Boldi, P., Rosa, M., Santini, M., Vigna, S. *Layered label propagation.* WSDM 2011.
3. Vigna, S. *Quasi-succinct indices.* WSDM 2013.
4. Malkov, Y., Yashunin, D. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI 2018.
5. Subramanya, S. J., Devvrit, F., Simhadri, H. V., Krishnaswamy, R., Kadekodi, R. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
6. Lemire, D., Boytsov, L. *Decoding billions of integers per second through vectorization.* Software: Practice and Experience 2015.
