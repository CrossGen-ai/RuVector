# ADR-272 — AISAQ: All-in-Storage ANNS with Quantization

**Status:** Proposed (nightly research, 2026-07-09)

**Related work:** ADR-268 (SPANN partition-spill), prior nightly
`2026-06-20-pq-adc-search` (in-memory PQ+ADC), prior nightly
`2026-06-19-lsm-ann` (LSM layout for ANN), `crates/ruvector-diskann/`
(Vamana on-disk graph).

## Context

DiskANN-style graph search has become the default recipe for billion-scale
ANN on a single node: raw vectors on SSD, PQ codes in RAM for fast
candidate scoring, graph in RAM for navigation. At scale the PQ codes
themselves become the memory bottleneck — 32 GB at 10⁹ points with
M=32 codes. Kioxia's AISAQ paper (arXiv:2404.06004, 2024) proposes
moving the PQ codes to SSD too, memory-mapped, and lets the OS page
cache serve the hot working set. Only the navigation graph is kept
resident. Recent 2024–2025 releases of Milvus and Qdrant have shipped
similar "on-disk PQ" flags, so this is now mainline SOTA — but ruvector
has no equivalent.

The design decision is not "should we do AISAQ" (the storage math is
compelling) but "**where in ruvector's crate graph does the storage
tier belong, and what is the stability boundary**". Today, our PQ
implementations (`ruvector-pq-search`, `ruvector-anisotropic-pq`,
`ruvector-rabitq`) each own their own storage layout, making it hard
to swap RAM for SSD.

## Decision

Introduce a new crate `ruvector-aisaq` whose sole export is the
`DistanceBackend` trait plus three reference implementations:

* `FlatF32Ram` — raw f32 vectors in RAM (recall-perfect baseline)
* `PqRam` — PQ codes in RAM with in-memory ADC
* `PqDisk` — PQ codes mmap'd from disk (AISAQ)

All backends plug into the same beam search over the same graph. The
trait defines the exact stability boundary between "algorithm" and
"storage" for future ruvector ANN work. New quantisers (RaBitQ, LVQ,
learned) become new `DistanceBackend` impls; new storage tiers (direct-
IO, S3, tiered NVMe) also become new impls. Nothing above the trait
needs to change.

We deliberately do **not** couple this to `crates/ruvector-diskann`
in v0.1 — that crate's Vamana build has ownership assumptions we don't
want to force onto every backend. The PoC ships with a brute-force k-NN
graph so the storage question can be measured in isolation.

## Consequences

**Positive**

* AISAQ heap footprint measured at 147 KB vs 10.24 MB for the flat
  baseline on N=20 000, D=128 — a **69×** shrink of the resident set.
* Identical top-k output between `PqRam` and `PqDisk` on 200 queries,
  so the storage decision is provably algorithm-neutral.
* The `DistanceBackend` trait becomes the natural extension point for
  future storage tiers without churning the graph or algorithm code.
* Aligns ruvector with what Milvus 2.4 and Qdrant 1.10 already ship.

**Negative**

* +40% per-query latency in the hot-page-cache regime at N=20 K
  (85 µs vs 61 µs). At small N this is a bad tradeoff — the PoC
  documents this as a config decision, not a default.
* AISAQ is read-mostly: streaming inserts need a two-tier design
  (mutable RAM head + immutable mmap tail), not implemented in v0.1.
* PoC uses brute-force graph build (O(N² D)) which is fine at 20 K
  but does not scale — production must swap in Vamana from
  `ruvector-diskann`.

**Alternatives considered**

1. **Do nothing, keep PQ in RAM.** Rejected: at billion scale the codes
   alone blow past commodity RAM budgets; this is now competitor-table
   stakes.
2. **Extend `ruvector-pq-search` with a "storage" enum.** Rejected: it
   couples storage to that specific PQ implementation; RaBitQ and
   LVQ would need parallel enums.
3. **Fold AISAQ into `ruvector-diskann`.** Rejected: DiskANN's Vamana
   build is opinionated about ownership of the vector buffer; forcing
   every future backend through Vamana's graph builder would fight
   the trait boundary we want.
4. **Direct-IO backend from the start.** Deferred: correctness first,
   then bypass the page cache in a follow-up crate
   (`ruvector-aisaq-directio`). See research doc roadmap.

## Follow-ups

* Wire `DistanceBackend` into `ruvector-diskann`'s Vamana search loop.
* Add a `RerankTier` that fetches raw vectors from a second mmap file
  for the top-k' before returning top-k — this recovers most PQ recall
  loss and is the canonical DiskANN move.
* Publish a `crates/ruvector-aisaq-bench` scaling study at
  N ∈ {10⁵, 10⁶, 10⁷} once SIFT/DEEP data ingest is wired in.
