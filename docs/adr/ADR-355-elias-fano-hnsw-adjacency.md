# ADR-355: Elias-Fano Compressed HNSW Adjacency Lists

- **Status**: Proposed (research PoC)
- **Date**: 2026-10-26
- **Owner**: ruvector nightly research
- **Related**: ADR-354 (spherical-kmeans IVF), prior nightly `hnsw-freq-reorder`, `hnsw-delta-neighbors`

## Context

HNSW graphs dominate ruvector's dense retrieval surface. For a layer with `N`
nodes and `M` neighbors per node, the adjacency alone costs `4 * N * M` bytes
when stored as `Vec<u32>` ids. On a billion-vector shard with `M = 32` that is
~128 GiB *before* vectors — adjacency memory is now a first-order budget line.

Two cheap wins already exist in literature:

1. **Fixed-width bit packing.** For `N` nodes an id fits in `ceil(log2 N)` bits
   (17 bits at `N = 100k`, 30 bits at `N = 1B`), giving a 32 → `W` bit ratio.
2. **Elias-Fano (EF).** Sorting the per-node list and encoding with EF uses
   `~m * (2 + log2(u/m))` bits, which is strictly less than packed when the
   universe `u` is much larger than `m` — exactly the HNSW regime.

Previous nightly research covered neighbor **ordering** (frequency reorder) and
neighbor **delta** encoding for incremental writes. Neither tackled *static*
compressed adjacency with random `select(i)` access during search.

## Decision

Ship a vendor-free `ruvector-ef-hnsw-adjacency` crate exposing an `Adjacency`
trait with three static backends:

- `PlainAdjacency` — flat CSR `Vec<u32>` baseline.
- `PackedAdjacency` — single shared bitstream, `ceil(log2 N)` bits per id.
- `EFAdjacency`     — per-node Elias-Fano (low bits + differential-unary high
  bits) sharing two global bitvectors with per-node offsets.

All three implement the same iteration contract so an HNSW search loop can
swap backends without touching the planner. The PoC is a standalone crate so
it does not perturb the workspace build.

## Consequences

### Positive

- Opens a knob — "memory per edge" — that is independent of recall tuning.
- Measured 1.4×–2.1× byte reduction over the plain baseline at `N = 10k–100k`
  with `M = 16–32` (benchmark below).
- Packed-bits backend lands within ~5 % of EF bytes with better decode speed —
  a strong default for shards where memory is bounded but not tight.
- Clean trait boundary enables mixing: compressed layer 0, plain upper layers.

### Negative

- Decode is slower than plain: ~3–4× ns/traversal in the PoC, driven by the
  unary scan over the high bitstream. Mitigation is a precomputed `select1`
  index per node, which raises bytes-per-edge by ~0.1–0.3 and recovers near-
  plain speed; left for a follow-up.
- Compression wins shrink as `M / N` grows (dense layers near 1 B). At the
  100 k × 32 point measured, EF only beats plain by 1.76× — packed matches.

### Alternatives considered

- **Partitioned Elias-Fano (Vigna, 2014).** Would improve wins on skewed
  degree distributions but adds a partition descriptor; deferred.
- **Delta + v-byte / PFOR.** Rejected: the gap distribution for HNSW neighbor
  ids is close to uniform within `[0, N)`, where EF's `~log2(u/m)` term is
  already near-optimal.
- **Dense bitmap per node.** Only cheaper for `M >= N/32`, which never
  happens in practice.

## Follow-up roadmap

1. Add a `select1` directory (every k-th 1 bit) to the EF high stream — target
   is <1.5× plain decode cost at the same byte count.
2. Thread the trait through `ruvector-hnsw-*` so an existing index can be
   rewritten to EF offline, no re-index required.
3. Measure on real SIFT1M / GIST1M embeddings for recall-preserving numbers.
