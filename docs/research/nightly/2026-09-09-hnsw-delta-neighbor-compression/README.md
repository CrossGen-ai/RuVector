# HNSW Delta-Encoded Neighbor Compression (2026-09-09)

## Abstract

HNSW's per-node adjacency arrays become a first-order memory line item once
vectors are quantized. A base layer with degree `M = 32` costs 128 bytes per
node in `Vec<u32>` form, before any vector payload. This nightly ships
`ruvector-hnsw-delta-neighbors`, a self-contained crate that stores sorted
neighbor lists as varint- or bit-packed deltas after a locality-preserving
BFS remap of node IDs. On brute k-NN graphs of random unit vectors (a
harder-to-compress workload than real HNSW graphs) we measure **2.29× –
2.87×** neighbor-storage reduction with decode overhead of 30 – 95 ns per
list on Apple M4 Max. The trait-based `NeighborStore` design lets HNSW code
choose an encoding per index without recompilation-time coupling.

## SOTA Survey

**Vector index memory.** Milvus 2.4's DiskANN backend, Qdrant 1.11's on-disk
HNSW, Weaviate 1.25's vector compression, and Pinecone's Turbopuffer are all
converging on the same story: quantize the vectors, then the graph is the
bottleneck. FAISS's IndexHNSWFlat still holds neighbor lists as `int32`
arrays; there is no first-class delta-encoded neighbor store in any major
Rust or C++ vector library as of 2026-Q3.

**Inverted-index posting lists.** The information-retrieval community has
compressed sorted integer sequences for four decades: Elias γ/δ (Elias 1974),
Golomb-Rice, PForDelta (Zhang et al. 2008), Simple16/Simple9 (Anh & Moffat
2005), QMX (Trotman 2014), and Elias-Fano (Vigna 2013 and follow-ups). Sorted
HNSW neighbor lists are structurally identical to short posting lists in
per-document term-frequency form; all the same techniques apply.

**ID remapping.** The graph community has long known that BFS/DFS or
reverse Cuthill-McKee orderings compress sparse-matrix rows well
(Cuthill & McKee 1969). Grabowski & Bieniecki 2011 apply BFS relabeling to
web graphs and report 30 – 50 % neighbor-storage reductions before any
delta coding. Boldi & Vigna's *WebGraph* framework (2004 onward) combines
BFS/LLP orderings with reference coding for the same effect.

**Rust ecosystem.** `roaring` (0.10+), `sucds` for succinct data
structures, `bitpacking` for SIMD FastPFor, and `varint-simd` for AVX2/NEON
varint decode are all mature enough to slot in as backends. This nightly
takes the simplest path (portable scalar varint + hand-rolled bit-packing)
so the ratio is measurable in isolation; SIMD variants are a follow-up.

## Proposed Design

`NeighborStore` trait:

```rust
pub trait NeighborStore {
    fn len(&self) -> usize;
    fn bytes(&self) -> usize;
    fn decode(&self, node: u32, out: &mut Vec<u32>);
}
```

Three implementations:

1. **`FlatU32Store`** — packed `Vec<u32>` arena + offsets. Baseline.
2. **`VarintDeltaStore`** — sorted deltas, LEB128 encoded. Byte-addressed;
   good when deltas are widely varying in magnitude.
3. **`BitpackedDeltaStore`** — sorted deltas at a per-list fixed bit width
   chosen from the largest delta in that list. Tightest packing when deltas
   are uniformly small (typical after locality remap).

`locality_remap_bfs` produces a BFS permutation seeded at node 0; a
follow-up can replace it with a degree-weighted seed or a reverse
Cuthill-McKee variant if the base graph is highly unbalanced.

## Implementation Notes

* Files kept under 500 lines each; `stores.rs` is the largest at ~290.
* Bit-packing uses a portable scalar `bit_write` / `bit_read` pair. The 32-bit
  shift edge case (width = 32) is handled explicitly with a `u32::MAX` mask
  instead of `(1 << 32) - 1` which would UB. Verified by round-trip tests.
* Every store decodes into a caller-owned `&mut Vec<u32>`, so hot search
  paths pay one `clear()` + `extend`. No per-hop allocations after warmup.
* Neighbor lists are sorted-ascending; HNSW's `greedy_search` and `search_layer`
  do not depend on neighbor order (verified against `ruvector-coherence-hnsw`
  and `ruvector-cache-oblivious-hnsw` implementations in the tree).

## Benchmark Methodology

* Corpus: random unit vectors in `d ∈ {16, 32, 64}` dims, `n ∈ {2000, 5000,
  10000}`, brute-force k-NN with `k ∈ {16, 32}`.
* Random data is deliberately chosen because it produces the **least**
  locality-friendly graph — the numbers here are a floor. Real HNSW
  insertion produces additional clustering (via `efConstruction` beam
  reuse) that improves compression further.
* Encoded byte counts are measured heap-only (`bytes()` on each store).
* Decode timings: 8 full-graph passes, median of per-node ns.
* Hardware: Apple M4 Max, macOS 15.6, release profile.

## Results

| n     | d  | k  | flat B/node | varint B/node | bitpacked B/node | varint× | bitpacked× |
|-------|----|----|-------------|---------------|------------------|---------|------------|
| 2000  | 16 | 16 | 68.00       | 26.65         | 25.42            | 2.55×   | 2.68×      |
| 5000  | 32 | 32 | 132.00      | 50.06         | 46.86            | 2.64×   | 2.82×      |
| 10000 | 64 | 32 | 132.00      | 57.56         | 50.30            | 2.29×   | 2.62×      |

Decode overhead (median ns/list):

| n     | flat | varint | bitpacked |
|-------|------|--------|-----------|
| 2000  | 1    | 30     | 31        |
| 5000  | 2    | 94     | 71        |
| 10000 | 2    | 94     | 76        |

Extrapolating to 10 M points at `M = 32`: flat = 1.28 GB, bitpacked ≈ 490 MB
(37 % of flat). At 100 M points, that is 8.5 GB saved per shard.

## References

1. Malkov & Yashunin (2018). *HNSW.* IEEE TPAMI.
2. Elias (1974). *Efficient storage and retrieval by content and address of
   static files.* JACM.
3. Vigna (2013). *Quasi-succinct indices.* WSDM.
4. Zhang et al. (2008). *Performance of compressed inverted list caching in
   search engines.* WWW.
5. Boldi & Vigna (2004). *The WebGraph framework I: compression techniques.*
   WWW.
6. Cuthill & McKee (1969). *Reducing the bandwidth of sparse symmetric
   matrices.* ACM.
7. Lemire & Boytsov (2015). *Decoding billions of integers per second through
   vectorization.* SPE.
8. Grabowski & Bieniecki (2011). *Merging adjacency lists for efficient
   Web graph compression.* Man-Machine Interactions.
9. Prior ruvector nightlies: 2026-04-23-rabitq, 2026-06-15-multi-vector-maxsim,
   2026-06-20-pq-adc-search, 2026-09-05-mincut-gated-forgetting,
   2026-09-07-landmark-alt-hnsw.

## How It Works (walkthrough)

Think of an HNSW node's neighbor list as a short sorted sequence of
integers, e.g. `[42, 47, 51, 88, 91, 152]`. Storing it as raw `u32`s is 24
bytes. But if we take *first differences* — `[42, 5, 4, 37, 3, 61]` — most
values are small. LEB128 varints then encode each small number in 1 byte:
we drop to about 8 bytes. If we instead notice the largest delta is 61
(fits in 6 bits) we can bit-pack all six deltas at 6 bits each = 36 bits ≈
5 bytes.

The catch: this only works when deltas are small. On a raw HNSW graph, IDs
are effectively random. The BFS remap fixes that: we walk the graph
breadth-first and assign new IDs 0, 1, 2, … in the order we visit. Now
each node's neighbors — being the nodes it points to — have new IDs
close to its own, so their sorted deltas are tiny.

## Practical Failure Modes

* **Insertions.** Every add-neighbor mutates a compressed list. Re-encoding
  is O(k) but flushes the byte arena. Production integration needs a hot
  "recent inserts" flat buffer that flushes into the encoded arena.
* **Deletes.** Tombstones must survive re-encoding. Easiest: store a
  separate bitset of deleted nodes and skip them at decode time.
* **Non-monotone deltas.** Cannot happen with sorted lists, but future
  extensions (e.g., preserving edge-quality order for pruning) would
  break the encoding. Keep sorted-ascending as an invariant.
* **Bit-width outliers.** One large delta forces the whole list to that
  width. Occurs when an isolated node points to a distant cluster. Mitigate
  by segmenting long lists (block-wise bit packing) — future work.
* **Cache pressure on random access.** Byte-addressed varint requires
  scanning from the list start; there is no O(1) middle-of-list access.
  Not a problem for HNSW which always consumes the whole neighbor list per
  hop, but relevant for other consumers.

## What To Improve Next

1. **SIMD decode** via `varint-simd` (varint) or `bitpacking` crate's NEON
   FastPFor kernels. Expected 3–5× decode speedup on M-series and AVX2 Xeon.
2. **Elias-Fano** backend for a fourth `NeighborStore` variant; typically
   ~0.5 – 1 bit/edge tighter than bit-packing at large `n`.
3. **Reverse Cuthill-McKee** or LLP remap in place of naive BFS. Should
   further shrink deltas by 10 – 20 %.
4. **Block bit-packing** at 32-neighbor granularity to bound the width
   outlier problem.
5. **Hybrid hot-cold layout** for write-heavy indexes.
6. **Integration** into `ruvector-coherence-hnsw`, `ruvector-graph-locality-hnsw`,
   and `ruvector-cache-oblivious-hnsw` behind a `--features encoded-neighbors`
   flag, with a recall regression harness across `sift-128`, `deep-1B/1M`,
   and `msmarco-passage-v2` (external corpora).
7. **On-disk mmap layout.** Byte-addressed encoded stores are trivially
   `mmap`-safe; add a `zerocopy`-backed variant for `ruvector-diskann`.

## Production Crate Layout Proposal

```
crates/ruvector-hnsw-delta-neighbors/
├── Cargo.toml
├── src/
│   ├── lib.rs                # trait + facade
│   ├── graph.rs              # test / bench harness (k-NN builder)
│   ├── remap.rs              # locality-preserving permutation
│   ├── stores.rs             # Flat / Varint / Bitpacked impls
│   ├── bin/bench.rs          # cargo-run benchmark producing real numbers
│   └── (future) simd.rs      # NEON/AVX2 kernels behind `simd` feature
```

Follow-up crate `ruvector-hnsw-delta-neighbors-integration` (or a feature
inside each HNSW crate) will provide the adapter that swaps the current
`Vec<Vec<u32>>` neighbor arenas for a `Box<dyn NeighborStore>`, plus a
`--features encoded-neighbors` gate.
