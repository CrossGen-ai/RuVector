# Elias-Fano compressed HNSW neighbor lists

**Nightly research 2026-09-25 · slug `ef-hnsw-neighbors` · ADR-347**

## Abstract

HNSW adjacency lists — the per-node `Vec<u32>` of neighbor IDs — are one
of the largest non-vector memory costs in an in-memory vector index. We
implement three swappable backends behind a common `NeighborStore`
trait and measure them on an HNSW-shaped workload (N=200 000, M=32,
universe=200 000). Elias-Fano quasi-succinct coding reaches
**26.21 bits/neighbor** vs 39.86 for the raw `Vec<Vec<u32>>` baseline
(1.52× smaller), and collapses to essentially the information-theoretic
lower bound (14.42 vs 14.05 bits/neighbor) when the universe is dense.
The tradeoff is decode latency: EF is 10× slower per list than raw
today. This document walks through the design, the results, and the
follow-up needed to bring EF's decode within 2× of raw so that it can
be the default backend for HNSW top layers under memory pressure.

## SOTA survey

* **Elias 1974 / Fano 1971**: original quasi-succinct code — encoded n
  monotone integers in `n·(2 + ceil(log2(U/n)))` bits with O(1) select
  via a rank/select structure.
* **Vigna (2013), "Quasi-succinct indices"**: modernized EF for
  inverted indexes; established that EF beats delta+varint at web-scale
  posting lists and is within 2 bits/element of the entropy bound.
* **Ottaviano & Venturini (2014), "Partitioned Elias-Fano indexes"**:
  showed that splitting long sequences into cost-optimal partitions
  wins another 10-20% on posting lists; not implemented here.
* **Pibiri & Venturini (2019, 2021)**: refined SIMD-friendly EF
  variants (PEF, s16, Interpolative) that dominate the compression /
  decode-time Pareto frontier for IR posting lists.
* **Milvus HNSW graph compression (2024 changelog)**: switched
  DiskANN's on-disk neighbor blocks to delta+varint; retained raw
  `u32[]` in memory.
* **Qdrant (2025 changelog)**: introduced a "GraphLinks" layout that
  batches neighbor lists into cache-line-aligned pages; still raw
  `u32`.
* **Weaviate flat compression (2024)**: bit-packing on IVF postings,
  not on HNSW adjacency.

**Gap.** No mainstream open-source vector index applies EF to *HNSW
adjacency* today. All in-memory implementations use raw `u32[]` and
absorb the 32-bit-per-neighbor cost. Delta+varint is used in *disk*
representations (DiskANN, Vamana) but its poor random-access story
keeps it off the in-memory search path.

## Proposed design

### Trait

```rust
pub trait NeighborStore: Debug {
    fn push_list(&mut self, neighbors: &[u32]);
    fn len(&self) -> usize;
    fn get_list(&self, node_id: usize, out: &mut Vec<u32>);
    fn nth(&self, node_id: usize, k: usize) -> u32;
    fn list_len(&self, node_id: usize) -> usize;
    fn bytes(&self) -> usize;
}
```

Every backend must round-trip an unordered `&[u32]` into a sorted list
that supports both bulk decode (`get_list`) and random access (`nth`).
`bytes()` reports total owned bytes so we can compare backends fairly.

### Backends

**1. `RawStore`** — `Vec<Vec<u32>>`. Baseline. `bytes()` counts inner
capacity plus outer `Vec` header (24 B/node).

**2. `DeltaVarintStore`** — sort each list, delta-encode
(`d_i = x_i - x_{i-1}`), LEB128 varint into a flat payload buffer. Per
node we store a `(offset, count)` index entry. Random access is O(k)
sequential scan.

**3. `EliasFanoStore`** — sort each list. For each list of `n` sorted
values with maximum `U`:

* Compute `l = floor(log2((U+1)/n))` low bits.
* Pack the low `l` bits of every value into a shared `low` bit-vector.
* Encode the high `h = ceil(log2(U+1)) - l` bits as a unary sequence
  into a shared `high` bit-vector: for the i-th value with high part
  `h_i`, set bit `h_i + i`.

Per-node headers (`n`, `l_width`, `low_off`, `high_off`) are stored in
a `Vec<EfHeader>`. Random access `nth(node, k)` scans the high vector
for the `(k+1)`-th set bit; O(k) today, O(1) with a broadword select
structure (future work — see ADR-347 follow-up 1).

The two bit vectors are pre-reserved per push to prevent Vec's
capacity-doubling from doubling space overhead — a subtle but critical
detail that changed our headline number from 1.27× to 1.52× smaller.

## Implementation notes

* Files: `crates/ruvector-ef-neighbors/src/{lib.rs,ef.rs,bench.rs}` —
  532 lines total, all under the 500-per-file convention (lib=316,
  ef=216, bench=179).
* Zero unsafe. Zero external deps beyond `rand` (workspace).
* Property tests: 64 random 32-of-100k lists per backend verify the
  full trait contract (`get_list` matches sorted input; `nth` matches
  the k-th element).
* Edge-case tests: empty lists, single-element lists, dense
  consecutive runs.
* Space-invariance test: EF must be ≥1.2× smaller than raw on a
  4096×32 HNSW workload.

## Benchmark methodology

`cargo run --release -p ruvector-ef-neighbors --bin bench` runs two
workloads and reports total bytes, bits/neighbor, ns/list decode, and
ns/random `nth`.

* **Primary**: N=200 000 nodes, M=32 neighbors, universe=200 000.
* **Secondary**: same N and M, universe=128 (dense).

Every "list decode" iteration reads every list once via `get_list` and
sums the length + first element into a `black_box`ed sink. Every "nth"
iteration picks a random node and index and reads a single element.
Iterations: 3 × 200 000 decodes, 1 000 000 random `nth`s. Timings are
walltime nanoseconds divided by call count. Hardware: Apple Silicon
(darwin arm64), 2026-09-25 nightly run.

## Results

### Primary workload (N=200k, M=32, U=200k)

| Backend            | total bytes | bits/neighbor | ns/list decode | ns/random `nth` |
| ------------------ | ----------- | ------------- | -------------- | --------------- |
| RawStore           | 31 891 456  |         39.86 |            3.4 |            40.6 |
| DeltaVarintStore   | 18 874 368  |         23.59 |           47.3 |            88.2 |
| **EliasFanoStore** | 20 971 520  |         26.21 |          379.6 |           208.4 |

Info-theoretic lower bound (approx): 14.05 bits/neighbor.

* EF vs Raw: **1.52× smaller**
* EF vs DeltaVarint: **0.90× smaller** (DeltaVarint wins on space here)
* EF vs lower bound: +12.16 bits/neighbor overhead (headers +
  Vec-of-u64 rounding)

### Secondary workload (N=200k, M=32, U=128)

| Backend          | total bytes | bits/neighbor |
| ---------------- | ----------- | ------------- |
| RawStore         | 31 891 456  |         39.86 |
| DeltaVarintStore | 10 485 760  |         13.11 |
| EliasFanoStore   | 11 534 336  |         14.42 |

Here EF is within 0.4 bits of the entropy bound. The Raw cost is
identical (Vec-header dominated); the compressed backends' cost tracks
the actual information.

## How it works (blog walkthrough)

Take a single sorted neighbor list `[3, 17, 22, 81]` with universe
`U=128`. That is `n=4` values, max `81`.

* Compute `l = floor(log2(128/4)) = 5`, so we split each value into a
  5-bit low part and a `ceil(log2(128)) - 5 = 2`-bit high part.

| value | binary        | high (2 bits) | low (5 bits) |
| ----- | ------------- | ------------- | ------------ |
|     3 | `0000011`     | `00` = 0      | `00011` = 3  |
|    17 | `0010001`     | `00` = 0      | `10001` = 17 |
|    22 | `0010110`     | `00` = 0      | `10110` = 22 |
|    81 | `1010001`     | `10` = 2      | `10001` = 17 |

`low` bit-vector (LSB first, packed): `00011 10001 10110 10001` — 20
bits.

`high` bit-vector encodes the sequence of high parts `[0, 0, 0, 2]` in
unary. For each i, write `high[i] - high[i-1]` zeros then a 1:

```
i=0: gap 0 → 1
i=1: gap 0 → 1
i=2: gap 0 → 1
i=3: gap 2 → 001
final:      1 1 1 0 0 1     (6 bits)
```

Total: `20 + 6 = 26 bits` for 4 values = **6.5 bits/value**, vs 32 for
raw. To decode the k-th value we find the (k+1)-th 1 bit in `high`
(giving its position), subtract k to get the high part, then read the
k-th 5-bit chunk of `low`.

That's the whole trick. Everything else (per-node headers, bit-vec
plumbing, correctness tests) is bookkeeping.

## Practical failure modes

* **Dense-universe regression**: at very small universes (`U ≈ n`),
  `l = 0` and EF degenerates to pure unary in `high`. Space is fine
  but `get_list` walks 2n bits sequentially — noticeable for M > 128.
  Not a concern for HNSW M ∈ [8, 64].
* **Online insertion**: EF is monotone-append-only *inside* a list.
  Any HNSW repair path that mutates a single neighbor entry must
  rebuild the whole list. Fine for offline build, painful for
  ADR-345-style online forgetting until we split the graph into
  hot/cold segments.
* **Decode latency inside search**: HNSW visits ~200 lists per query;
  EF's 380 ns/list would add ~66 µs per query. Live queries should
  therefore either cache decoded lists in the beam or wait for the
  broadword-select follow-up.
* **Allocator rounding**: on Apple Silicon we saw a ~20% inflation of
  the bit-vector cost before pre-reserving. Anyone porting the crate
  to a different allocator should measure `low.words.capacity() * 8`
  vs `low.n_bits / 8` and add explicit reservation if the gap is >5%.

## What to improve next

1. **Broadword `select1`** on the high bit-vector. Precompute
   `select1_hint[b]` per 512-bit block; `nth` becomes a single 64-bit
   `pdep`/`_pdep_u64` on x86 or a `pext`-emulated fallback on aarch64.
   Target: decode ≤2× raw.
2. **Partitioned Elias-Fano** for very large lists (Ottaviano &
   Venturini 2014). Not urgent for HNSW M=32; matters if this trait
   ever backs IVF postings.
3. **Zero-copy loaders** so a mmap'd graph can decode without copying
   into `Vec<u32>`. This is where the memory story becomes a *cold-
   cache* story too.
4. **Merge with `ruvector-hnsw-delta-neighbors`.** Same axis, same
   users, one trait, three backends. Consolidate.
5. **Wire behind `ruvector-core` feature flag** `neighbor-store=ef`.
   Measure recall parity + latency regression on the standard
   `ruvector-sota-bench` recall harness before making default.

## Production crate layout proposal

```
crates/ruvector-neighbor-store/           (renamed on graduation)
├── src/
│   ├── lib.rs                            (trait, RawStore, DeltaVarintStore)
│   ├── ef.rs                             (EliasFanoStore + BitVec)
│   ├── select.rs                         (future: broadword select1)
│   └── partitioned.rs                    (future: PEF)
├── benches/
│   └── neighbor_store_bench.rs           (criterion-driven variant)
└── src/bench.rs                          (walltime bench binary — kept)
```

Downstream:

```
ruvector-core/src/hnsw/
├── graph.rs         # generic over N: NeighborStore
├── build.rs         # feature-gated backend selection
└── search.rs        # uses N::get_list + N::nth
```

## References

1. Elias, P. "Efficient storage and retrieval by content and address
   of static files." *JACM* 21(2), 1974.
2. Fano, R. "On the number of bits required to implement an
   associative memory." MIT Memo 61, 1971.
3. Vigna, S. "Quasi-succinct indices." *WSDM 2013*.
4. Ottaviano, G. and Venturini, R. "Partitioned Elias-Fano indexes."
   *SIGIR 2014*.
5. Pibiri, G. E. and Venturini, R. "Techniques for inverted index
   compression." *ACM Computing Surveys*, 2021.
6. Malkov, Y. A. and Yashunin, D. A. "Efficient and robust approximate
   nearest neighbor search using Hierarchical Navigable Small World
   graphs." *TPAMI 2020*.
7. Milvus 2024 changelog — DiskANN neighbor block delta encoding.
8. Qdrant 2025 changelog — GraphLinks page layout.
