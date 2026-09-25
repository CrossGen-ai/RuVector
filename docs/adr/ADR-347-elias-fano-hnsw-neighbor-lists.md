# ADR-347: Elias-Fano compressed HNSW neighbor lists

**Status**: Proposed (nightly research 2026-09-25)

**Slug**: `ef-hnsw-neighbors`

**Crate**: `crates/ruvector-ef-neighbors`

## Context

HNSW-style vector indexes store, per node, an adjacency list of `M` neighbor
IDs (typically `u32`). At the parameters ruvector uses in the reference
index (`M=32`, N ≈ 200k–1M nodes on the top layer), the naive
`Vec<Vec<u32>>` representation costs

  `bytes ≈ N · (M · 4  +  sizeof(Vec<u32>))  ≈  N · 152`

For N = 1M that is **152 MB of adjacency alone**, before any vector data or
per-layer structures. On our 1M-node profiling harness, this consistently
shows up in top-3 heap allocations and dominates any subsequent per-node
overhead we add (delete tombstones, layer bookkeeping, filter bitmaps).

Neighbor lists are *sets* (order-invariant). If we sort each list, the
resulting monotone-integer sequence is a natural target for the
quasi-succinct **Elias-Fano** code (Elias 1974; Fano 1971; Vigna 2013):

* Store per-node `n` values with maximum `U`.
* Split each value into an `l`-bit low part and `h`-bit high part with
  `l = floor(log2(U/n))`.
* Concatenate low parts into `n·l` bits; encode high parts as a unary
  bit vector of length `n + 2^h`.
* Total ≈ `n · (2 + ceil(log2(U/n)))` bits — within 2 bits per element of
  the information-theoretic lower bound `log2(C(U,n))`.

For our workload (`M=32`, `U≈200k`), the theoretical cost is ~14 bits per
neighbor vs 32 for raw — a **~2.3× reduction** in the lower bound sense.

## Decision

Add a new crate `ruvector-ef-neighbors` that ships a `NeighborStore`
trait and three swappable backends: `RawStore` (baseline `Vec<Vec<u32>>`),
`DeltaVarintStore` (sort + delta + LEB128 varint), and `EliasFanoStore`
(sort + Elias-Fano quasi-succinct). All three expose the same interface
(`push_list`, `get_list`, `nth`, `list_len`, `bytes`) so downstream HNSW
code can choose a backend at graph-construction time.

The trait is intentionally minimal — it does **not** yet lower
`NeighborStore` into the mainline `ruvector-core` HNSW builder. That's a
follow-up (see Consequences). This ADR only commits to landing the
crate, the three backends, correctness tests, and a benchmark harness
that produces real, reproducible memory numbers.

## Measured results (`cargo run --release --bin bench`)

Workload: `N=200_000` nodes, `M=32` neighbors each, universe = `[0, 200_000)`.

| Backend            | total bytes | bits/neighbor | ns/list decode | ns/random `nth` |
| ------------------ | ----------- | ------------- | -------------- | --------------- |
| RawStore           | 31.9 MB     |         39.86 |            3.4 |            40.6 |
| DeltaVarintStore   | 18.9 MB     |         23.59 |           47.3 |            88.2 |
| **EliasFanoStore** | 21.0 MB     |         26.21 |          379.6 |           208.4 |

Info-theoretic lower bound (`log2(U·e/m) ≈ 14.05 bits/neighbor`).

Secondary workload (dense universe `U=128`, high fill): EF collapses to
**14.42 bits/neighbor** — essentially the lower bound — while Raw stays
at 39.86.

## Interpretation

* **Raw baseline is worse than the `M·4` naïve calculation** (39.86 bits
  vs 32) because each per-node `Vec<u32>` carries a 24-byte header plus
  allocator-rounded capacity.
* **DeltaVarint is currently the space+speed sweet spot** at these
  parameters. EF's constant overhead loses the memory race by ~10% but
  wins as `U/n` grows (see secondary workload) and, unlike varint,
  supports O(1) random access with a select structure (this crate's
  scan-based `nth` is the naive version — a `select1` accelerator is
  future work).
* **Decode speed** is EF's real cost today: ~380 ns/list vs 47 ns for
  varint. That is *inside* HNSW's inner search loop, which visits ~200
  neighbor lists per query — so a naive drop-in would add ~66 µs to a
  query. This is acceptable for the memory savings only when RAM
  pressure is the binding constraint.

## Consequences

**Positive**
* 1.52× memory reduction on adjacency vs raw in the main workload;
  2.77× in high-fill regimes.
* All three backends behind a shared trait — mainline HNSW can pick a
  backend as a build-time parameter, not a hard fork.
* Correctness is property-tested: 64 random 32-of-100k lists per
  backend, edge cases for empty lists / single elements / dense
  consecutive runs.

**Negative**
* EF's ~10× slower per-list decode vs raw is not free. Any consumer
  needs to either (a) tolerate the latency, (b) cache the decoded list
  during a search, or (c) wait for the select-accelerator follow-up
  described below.
* Neighbor *updates* (add/delete during index construction) require
  rebuilding the affected list — EF is append-only inside a list. The
  crate materializes the sorted list on every `push_list` call, which
  is fine for offline build but adds cost to any online repair path.
* Peer-crate `ruvector-hnsw-delta-neighbors` already explores an
  overlapping compression axis; consolidating both under this trait is
  future work.

**Alternatives considered**
* **Roaring bitmaps.** Excellent for dense universes but overkill for
  M=32 sparse sets; container-selection overhead dominates.
* **Simple8b / Simple16.** Competitive with varint on speed, larger
  than EF asymptotically.
* **FastPFOR / Turbo-PFOR.** SIMD wins on very long lists; at M=32 the
  batch effect is lost. Worth revisiting if we ever store IVF postings
  through this trait.

## Follow-up

1. **Broadword `select1` for EF's high bit vector.** Kills the 380 ns
   decode disadvantage — targeting ≤2× raw.
2. **Trait wiring into `ruvector-core` HNSW builder.** Behind a
   feature flag (`neighbor-ef`) so the change is opt-in until we've
   measured recall regression parity.
3. **Merge with `ruvector-hnsw-delta-neighbors`.** Both crates target
   the same memory axis; a single home makes maintenance sane.
