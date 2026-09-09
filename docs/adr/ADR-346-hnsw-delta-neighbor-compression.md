# ADR-346: HNSW Delta-Encoded Neighbor-List Compression with Locality-Preserving ID Remapping

## Status

Accepted (experimental crate, `ruvector-hnsw-delta-neighbors`, feature-gate
candidate for existing HNSW crates once integration wiring lands). Not yet
integrated into `ruvector-coherence-hnsw`, `ruvector-graph-locality-hnsw`, or
`ruvector-cache-oblivious-hnsw`.

## Context

Once a vector index quantizes its vectors (PQ, RaBitQ, INT8, or the RaBitQ
+ residual fused path shipped in the 2026-04-23 nightly), the **graph**
becomes a first-order contributor to resident set size. An HNSW node with
`M = 32` neighbors stored as `Vec<u32>` costs 128 bytes per node on the base
layer alone, before any vector data. At 10M points this is 1.28 GB of
neighbor lists — often larger than the quantized vector payload itself.

Prior nightlies have optimized graph *quality* (`ruvector-graph-locality-hnsw`,
`ruvector-cache-oblivious-hnsw`, `ruvector-hnsw-rng-diverse`,
`ruvector-landmark-alt-hnsw`) and vector *compression* (`ruvector-rabitq`,
`ruvector-fused-rabitq-residual`, `ruvector-pq-search`, `ruvector-cascade-adc`)
but none have compressed the neighbor storage itself. Sorted neighbor lists
compress extremely well with the same techniques used in inverted-index
posting lists (varints, bit-packing, elias-fano), and HNSW's search algorithm
does not require neighbor order to be preserved — so lists can be stored
sorted-ascending without any change to correctness.

## Hypothesis

Given a k-NN graph on `n` nodes with degree `k` and neighbor IDs in `[0, n)`,
after applying a locality-preserving BFS remap of the IDs, sorted-delta
encoding with either LEB128 varints or per-list fixed bit-width packing
should reduce neighbor storage by ≥ 2× versus the flat `Vec<u32>` baseline,
with decode overhead low enough (< 200 ns per neighbor list on a modern ARM
laptop core) that the extra work per hop remains a small fraction of the
vector-distance work HNSW already performs.

## Decision

Add a new experimental crate, `ruvector-hnsw-delta-neighbors`, exposing a
`NeighborStore` trait with three implementations:

* `FlatU32Store` — baseline, 4 bytes per edge + offset table.
* `VarintDeltaStore` — sorted deltas encoded as LEB128 varints.
* `BitpackedDeltaStore` — sorted deltas packed at a per-list fixed bit width.

Plus a `locality_remap_bfs` helper that produces a BFS permutation and an
`apply_permutation` function that renumbers node IDs so neighbors land close
in the new ID space. All three stores decode into a caller-owned buffer
(`&mut Vec<u32>`) so the hot search path is allocation-free after warmup.

Downstream HNSW crates will opt in via a feature flag rather than a hard
swap: `NeighborStore` is a trait, and each backend is a drop-in for the
neighbor-arena today held as `Vec<Vec<u32>>` or `Vec<u32>` + offsets.

## Consequences

**Positive**

* Measured 2.29× – 2.87× reduction in neighbor-list bytes on brute k-NN
  graphs over random unit vectors (which is *harder to compress* than real
  HNSW graphs due to the absence of hierarchical-insertion locality). See
  benchmark section below.
* No change to graph semantics or search correctness (order-agnostic; sorted
  lists round-trip exactly, verified by `flat_roundtrip`, `varint_roundtrip`,
  `bitpacked_roundtrip` tests).
* Trait-based design lets future backends (Elias-Fano, PForDelta, `roaring`)
  slot in without breaking HNSW code.

**Negative / Trade-offs**

* Decoding is not free: 30–95 ns per neighbor list versus 1–2 ns for flat.
  This is an additive per-hop cost. For HNSW searches that visit ~200 nodes
  per query, that is ~15–20 µs extra per query on the measured M4 Max — a
  few percent of typical HNSW query latency on quantized data, but not
  negligible for latency-sensitive workloads.
* Insertion becomes more expensive: adding a neighbor requires re-encoding
  the list. For write-heavy workloads a hybrid "hot flat, cold encoded"
  layout is required (deferred to follow-up ADR).
* BFS remap requires an offline pass; incremental remap during insertion is
  out of scope here.
* This crate only measures the base-layer graph; upper layers are small
  and would compress with the same code but the absolute savings are
  dominated by the base layer.

## Alternatives Considered

1. **Elias-Fano coding.** Optimal information-theoretic bound for sorted
   monotone sequences and slightly tighter than bit-packing here, but
   markedly more complex decoder logic. Rejected for the first cut;
   revisit if bit-packing bottlenecks become the limiter.
2. **PForDelta / Simple16 / QMX.** Widely used in inverted-index systems.
   Better on lists with heterogeneous deltas but overkill for the narrow
   fixed-degree HNSW case. Adding as a fourth backend is a future option.
3. **Frame-of-reference on absolute IDs.** Same information density as
   sorted-delta after remap, no real advantage.
4. **`roaring` bitmap of neighbors.** Cache-inefficient at typical HNSW
   degrees (32–64); designed for much larger sets.
5. **Do nothing / bet on more RAM.** Rejected: neighbor bytes dominate on
   quantized workloads, and cache-line efficiency of encoded lists is
   competitive with flat u32 for the small `M` regime.

## Numeric Acceptance Test

Executed via `cargo run --release -p ruvector-hnsw-delta-neighbors
--bin delta-neighbors-bench` on Apple M4 Max, macOS 15.6, `rustc 1.x`
release profile.

| n     | d  | k  | flat B/node | varint B/node | bitpacked B/node | varint× | bitpacked× |
|-------|----|----|-------------|---------------|------------------|---------|------------|
| 2000  | 16 | 16 | 68.00       | 26.65         | 25.42            | 2.55×   | 2.68×      |
| 5000  | 32 | 32 | 132.00      | 50.06         | 46.86            | 2.64×   | 2.82×      |
| 10000 | 64 | 32 | 132.00      | 57.56         | 50.30            | 2.29×   | 2.62×      |

Decode overhead (median ns/list, M4 Max release build):

| n     | flat | varint | bitpacked |
|-------|------|--------|-----------|
| 2000  | 1    | 30     | 31        |
| 5000  | 2    | 94     | 71        |
| 10000 | 2    | 94     | 76        |

Both compression ratios exceed the 2× hypothesis threshold. Decode overhead
is under the 200 ns/list ceiling for all configurations tested.

## References

* Elias, P. (1974). *Efficient storage and retrieval by content and address
  of static files.*
* Cuthill, E., & McKee, J. (1969). *Reducing the bandwidth of sparse
  symmetric matrices.* — inspiration for locality-preserving remap.
* Malkov, Y.A., Yashunin, D.A. (2018). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.*
* Lemire, D., Boytsov, L. (2015). *Decoding billions of integers per second
  through vectorization.* — future SIMD-decode direction.
* Prior nightlies referenced: `ruvector-graph-locality-hnsw`,
  `ruvector-cache-oblivious-hnsw`, `ruvector-landmark-alt-hnsw`,
  `ruvector-fused-rabitq-residual`.
