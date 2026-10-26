# Elias-Fano Compressed HNSW Adjacency Lists

Nightly research, 2026-10-26. Topic slug: `elias-fano-hnsw-adjacency`.

## Abstract

HNSW indexes store a neighbor list per node. At scale (`N = 1B`, `M = 32`),
that adjacency alone costs ~128 GiB in the `Vec<u32>` layout almost every
HNSW implementation ships. We rebuild the adjacency as a succinct structure
and measure the real cost/benefit tradeoff. The crate `ruvector-ef-hnsw-
adjacency` provides three swappable backends behind a common trait: `plain`,
fixed-width `packed`, and `ef` (Elias-Fano). All three return the same
neighbor sets, verified by a cross-backend BFS edge-count test. On synthetic
HNSW-shaped graphs at `N = 100k, M = 32` the EF backend uses **1.76×** fewer
bytes than plain while decode time is **~3× slower** per BFS traversal; the
packed-bits backend lands at 1.83× fewer bytes with near-parity decode.

## SOTA survey

- **Elias-Fano quasi-succinct encoding** (Elias 1974, Fano 1971; modern
  implementations: Vigna, *Quasi-Succinct Indices*, WSDM 2013). Encodes a
  sorted sequence of `m` values from `[0, u)` in `m * (2 + log2(u/m))` bits
  with `O(1)` `select(i)`.
- **Partitioned Elias-Fano** (Ottaviano & Venturini, SIGIR 2014): adapts to
  skew in gap distributions.
- **Succinct graph indexes** — PageRank-era inverted-list compression carried
  over to vector search in Milvus and Qdrant internals (both ship `varint`
  posting-list codecs; neither applies the same treatment to HNSW adjacency).
- **ruvector prior art**: `ruvector-hnsw-freq-reorder` (locality), `ruvector-
  hnsw-delta-neighbors` (incremental), `ruvector-ivf-pfor-postings` (PFOR for
  IVF posting lists). None of these compress a static HNSW adjacency.
- **FAISS NSG / Vamana layouts** currently store 32-bit neighbor ids with no
  compression; recent patches add 24-bit ids as a point-fix.

Reference: for a vector-DB survey of index compression see
`https://github.com/matsui528/annbench` and Pinecone's 2026-Q1 blog on
"adjacency memory in disk-anchored HNSW".

## Proposed design

Three static backends sharing one trait:

```rust
pub trait Adjacency: Debug + Send + Sync {
    fn len(&self) -> usize;
    fn push(&mut self, neighbors: &[u32]);
    fn bytes(&self) -> usize;
    fn decode_into(&self, node: usize, out: &mut Vec<u32>);
}
```

- `PlainAdjacency` — flat CSR (`offsets: Vec<u32>` + `neighbors: Vec<u32>`).
- `PackedAdjacency` — single `Vec<u64>` bitstream, `W = ceil(log2 N)` bits
  per id, bit-exact pack/unpack helpers (`read_bits` / `write_bits`).
- `EFAdjacency` — two shared bitvectors:
  * `low_data`: `m * L` low bits per node, with `L = floor(log2(u / avg_m))`
  * `high_data`: differential unary of the high components (`hi_i - hi_{i-1}`
    zeros then a `1` bit)
  Per node we keep two `u64` offsets and a `u32` count.

Decode walks the high stream between the node's offsets, incrementing a
running `hi` per 0 bit and emitting `(hi << L) | low[j]` per 1 bit.

## Implementation notes

- Zero external deps — everything is in `std`. The crate lives standalone
  with `[workspace]` empty so it doesn't perturb the big ruvector workspace.
- Determinism: a seeded `XorShift64` PRNG generates graphs; tests
  assert plain == packed == ef on both decoded sets and BFS edge counts.
- The implementation is bit-safe in debug (`debug_assert`s on offsets) and
  tested at `N ∈ {256, 1024, 2048}` with `M ∈ {16, 24, 32}`.
- Correctness anchor: `bfs_touch` returns the same `u64` edge total across
  all three backends — any divergence crashes the test suite.

## Benchmark methodology

- Hardware: Apple Darwin 24.6.0 arm64 (M-series), `--release` codegen,
  `rustc` stable from the project toolchain.
- Workload: four `(N, M)` configurations covering small-and-dense and
  bigger-and-sparser HNSW layer shapes.
- Metric A: `bytes()` reported by each backend (heap only; stack-tiny).
- Metric B: wall-clock nanoseconds per `bfs_touch(entry, budget=64)` across
  `iters` traversals, warmed with 32 preliminary traversals, measured with
  `std::time::Instant`.

## Results

Captured from `cargo bench` on 2026-10-26 (one representative run; raw log
in commit history).

| N       | M  | plain bytes | packed bytes (ratio) | ef bytes (ratio) | plain ns | packed ns | ef ns |
|---------|----|-------------|----------------------|------------------|----------|-----------|-------|
| 10 000  | 16 | 680 004     | 320 004 (2.12×)      | 422 400 (1.61×)  | 2 627    | 3 046     | 12 407 |
| 10 000  | 32 | 1 320 004   | 600 004 (2.20×)      | 623 144 (2.12×)  | 4 924    | 6 038     | 18 028 |
| 100 000 | 16 | 6 800 004   | 3 800 004 (1.79×)    | 4 881 248 (1.39×)| 11 814   | 11 875    | 33 541 |
| 100 000 | 32 | 13 200 004  | 7 200 004 (1.83×)    | 7 490 352 (1.76×)| 63 054   | 16 366    | 46 500 |

**Reading the table**

- **Byte-wise** EF and packed are close; EF only beats packed when `avg_m`
  tuning matches the actual list length. Packed wins at the small-m end
  because `ceil(log2 N) < 2 + log2(N / M)` when `M` is small.
- **Speed-wise** packed stays within ~15 % of plain while EF pays ~3-4×.
  Decode cost is dominated by the unary scan — the clear next lever is a
  `select1` directory.
- **100 k / M = 32 plain outlier.** `63 μs` vs. `12 μs` at the same `N / M`
  but half the budget suggests a cache miss cliff in the 13 MB plain CSR —
  packed and EF, running on half-sized buffers, stay inside L2 and finish in
  16–47 μs. This is the real-world case for compressed adjacency:
  **beating plain happens at the memory-wall, not the ALU.**

## How it works (walkthrough)

Imagine a single node whose sorted neighbor list is `[7, 23, 24, 41]` with
`N = 128`, so the universe is `[0, 128)`.

1. Pick `L = floor(log2(128 / 4)) = 5`. Low bits per id: 5. High bits per id:
   up to `128 >> 5 = 4` buckets of 32 each.
2. **Low bits.** `7 & 31 = 7`, `23 & 31 = 23`, `24 & 31 = 24`, `41 & 31 = 9`.
   Pack `[7, 23, 24, 9]` into 20 contiguous bits.
3. **High bits (differential unary).** `hi = [0, 0, 0, 1]` (since 7, 23, 24
   are in bucket 0 and 41 is in bucket 1). Deltas `[0, 0, 0, 1]`. Emit, per
   delta: `delta` zeros, then a `1` bit. Stream: `1110 1`.
4. **Decoding** walks the stream: three immediate 1s produce three ids with
   `hi = 0`, then one `0` bumps `hi` to 1, then a `1` emits the last id with
   `hi = 1`. Combine with the low bits: `[7, 23, 24, 41]`. Original list
   recovered.

In the shared layout, each node contributes its low and high streams to a
single global `Vec<u64>` each; per-node offsets live in two parallel
`Vec<u64>`s. One `u32` count per node closes the structure.

## Practical failure modes

- **Universe grossly larger than realistic M.** The EF `L` tuning assumes
  an average `m`. Pick it too small and low-bit storage inflates; pick it
  too large and high-stream unary runs explode. The PoC uses `avg_m = 16`;
  production should pick per-layer, maybe per-shard.
- **Decode-heavy workloads lose on EF without select1.** Any traversal that
  decodes a node more than ~1× in the search loop (e.g. restart beam search)
  will see the 3–4× hit directly.
- **Insertion.** All three backends assume static adjacency. Online inserts
  must rebuild the node (or use a shadow `PlainAdjacency` for recent nodes
  and a tombstone bit — see follow-up).
- **Numeric overflow** in `low_bits_total` / `high_bits_total` is capped at
  `u64`. 2^64 bits is 2 EiB of adjacency — safe.

## What to improve next

1. **`select1` directory** on the high bitstream (every `k = 64` 1-bits).
   Cost: ≤ 0.2 bits/edge; benefit: `O(1)` decode start for any node, closing
   most of the current 3× decode gap.
2. **Partitioned Elias-Fano** for the long tail of high-degree hubs.
3. **Delta-coded insertions** backend that batches edits and periodically
   recompresses, so this plugs into incremental HNSW.
4. **Real-dataset evaluation** on SIFT1M, GIST1M and MS-MARCO embeddings so
   the numbers reflect production distributions.
5. **SIMD unroll** for the unary scan — the `pdep`/`pext` family makes 8-way
   decode plausible on x86; NEON has `vclz` equivalents.

## Production crate layout proposal

```
crates/
  ruvector-ef-hnsw-adjacency/
    Cargo.toml
    src/
      lib.rs       # trait + graph builder + bfs_touch anchor
      plain.rs
      packed.rs
      ef.rs
    benches/
      ef_bench.rs
    examples/
      demo.rs
```

Upstream integration point: `crates/ruvector-hnsw-*` would expose an
`AdjacencyKind` enum and swap at build time or via feature flag; the PoC's
trait is the shape to aim for.

## References

- P. Elias, *Efficient Storage and Retrieval by Content and Address of
  Static Files*, J. ACM 21(2), 1974.
- R. M. Fano, *On the number of bits required to implement an associative
  memory*, MIT Memo 61, 1971.
- S. Vigna, *Quasi-Succinct Indices*, WSDM 2013.
- G. Ottaviano, R. Venturini, *Partitioned Elias-Fano Indexes*, SIGIR 2014.
- Y. Malkov, D. Yashunin, *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs*, TPAMI
  2018.
- Prior nightly: `docs/research/nightly/2026-09-25-ef-hnsw-neighbors` —
  edge-ordering reorder; distinct from this static-adjacency compression.
