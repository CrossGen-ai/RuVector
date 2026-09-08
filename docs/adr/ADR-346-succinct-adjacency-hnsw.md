# ADR-346: Succinct Adjacency for Proximity-Graph ANN — BFS-Reordered Delta + VarByte

## Status

Accepted as a nightly research artifact. Landed as
`crates/ruvector-succinct-hnsw` (behind no feature flag; standalone
crate, no consumers yet). Promotion into the main `ruvector-hnsw`
and `ruvector-diskann` graph builders is deferred until (a) an
`Adjacency` trait split lands in those crates and (b) the SIMD
VarByte decode follow-up (see "Consequences") is prototyped.

## Context

Every proximity-graph ANN index in the ruvector workspace stores
neighbours as `Vec<Vec<u32>>` (or a fixed-stride equivalent). For a
canonical HNSW at `N = 1e6, M = 32` this is ~150 MB before any
vector data — the single largest post-quantisation cost. None of
the surveyed open-source vector DBs (FAISS, HNSWlib, Milvus,
Qdrant, Weaviate, LanceDB, DiskANN, usearch) compress the graph
itself, despite the web-graph literature (Boldi & Vigna 2004+)
showing 3–10× reductions are routine on graphs with natural
locality via reorder + varint / Elias-Fano encoding.

Two things had to be measured before we could commit an approach:

1. Whether ANN kNN graphs have enough id-locality (or gain it
   cheaply from reordering) to benefit from delta encoding at all.
2. Whether the decode cost fits inside a beam-search hot loop
   without visibly moving latency.

The 2026-09-08 nightly (`docs/research/nightly/2026-09-08-succinct-adjacency-hnsw`)
answers both.

## Decision

Add a swappable `Adjacency` trait with three concrete backends
in a standalone `ruvector-succinct-hnsw` crate:

- `DenseAdj` — baseline `Vec<Vec<u32>>`, for A/B comparison.
- `DeltaVarByteAdj` — sorted-neighbour deltas encoded as unsigned
  LEB128 into a flat `Vec<u8>` blob with a `Vec<u32>` offset
  table.
- `ReorderedDeltaAdj` — build-time multi-source BFS produces a
  permutation `old_to_new`; adjacency is re-labelled into new-id
  space, delta+VarByte encoded, and translated back at read time.

The graph builder (single-layer symmetrised exact-kNN, purely to
isolate encoding effects) writes `Vec<Vec<NodeId>>` once; each
backend consumes the same lists. Beam search is
backend-agnostic and takes `&dyn Adjacency` via generics.

### Rationale for the specific choices

- **VarByte over Elias-Fano.** VarByte decodes with a trivial hot
  loop and no auxiliary `select` structures. Elias-Fano is
  denser at high `d̄` but the delta between the two is
  <25 % on graphs with `d̄ ≤ 32` — the regime that dominates in
  production HNSW.
- **BFS over LLP.** BFS is `O(N + E)`, no tuning. LLP is
  20–30 % better on web graphs but requires iterative label
  propagation with hyperparameters. Deferred.
- **Sorted deltas, not delta-of-deltas.** Single delta already
  gives 1–2 byte varints on locality-rich graphs; the second
  derivative is <5 % additional saving and adds decode branches.
- **Standalone crate first.** The current HNSW/DiskANN crates
  bake `Vec<Vec<u32>>` deep into their APIs; a clean trait
  split is a separate change and would inflate this ADR's blast
  radius.

## Consequences

### Positive (measured on 2026-09-08)

- **3.08× adjacency reduction** on n=20 000, dim=32, M=16
  clustered corpus (`2 235 372 B → 726 749 B`).
- **3.44× reduction** on n=5 000, M=32 (`984 424 B → 285 953 B`).
- **Zero recall change**: encoding is bit-lossless; identical
  neighbour sets → identical beam walks → identical top-k.
- **Latency neutral within ±3 %** on Apple Silicon release
  builds (reordered variant recovers most of the raw
  varint-decode overhead because smaller blobs mean fewer
  cache lines touched during the walk).

### Negative

- **Isotropic-corpus regression.** On the d32_isotropic scenario
  the reordered variant costs 0.40× (vs 0.35× for plain delta):
  no id-locality to exploit → permutation table is dead weight.
  Mitigation planned: build-time `min(bytes(delta), bytes(reord))`
  auto-selection; not yet implemented.
- **Batch-build only.** Both encoded backends are immutable once
  built. Streaming ingest would require a segment-based
  copy-on-write layout — out of scope for this ADR.
- **Adds a `dyn`-ish seam.** Callers of `neighbors_into` must
  supply a scratch `Vec<NodeId>`. Fine for the search hot loop
  (one buffer per query) but slightly changes the ergonomics
  vs. `&[NodeId]`-returning APIs.

### Follow-ups (tracked, not blockers)

- SIMD (masked-VByte / group-varint) decode.
- LLP reordering as an opt-in alternative to BFS.
- Elias-Fano payload for `d̄ ≥ 32`.
- `mmap`-backed variant (blob is already flat).
- Auto-select delta vs. reordered by `bytes()` at build time.
- Integrate as the default `Adjacency` in `ruvector-hnsw` once
  the trait split lands.

## Alternatives considered

1. **Elias-Fano throughout.** Denser on very dense graphs, but
   adds a `select` structure and its own code path. Roughly
   15–25 % additional saving at `d̄ = 32`. Chose VarByte for
   PoC simplicity; EF is a drop-in follow-up.
2. **Delta + fixed-width bit packing** (à la FastPFor blocks).
   Would give 2–3× reduction at higher decode throughput than
   VarByte. Rejected for the PoC because tuning block size and
   exception storage adds moving parts that aren't the point
   of this measurement.
3. **Snappy / zstd on the whole blob.** Tested informally: 2×
   smaller than dense on the same data (comparable to VarByte
   *before* reordering) but decode is 5–10× slower and does
   not preserve random access. Wrong shape for a search hot
   loop.
4. **Do nothing; rely on RAM getting cheaper.** RAM per-GB has
   been flat since 2022; a 3× reduction on the largest
   post-quantisation cost is worth ~$1–3 per index-GB per
   month on cloud rates.
