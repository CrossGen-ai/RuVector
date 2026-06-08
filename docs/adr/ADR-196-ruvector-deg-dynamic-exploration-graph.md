# ADR-196: ruvector-deg — Dynamic Exploration Graph (DEG) index

- **Status:** Proposed (2026-06-08, nightly research branch)
- **Authors:** nightly-researcher agent
- **Supersedes:** none
- **Superseded by:** none
- **Related:** ADR-193 (RAIRS IVF), ADR-194 (ONNX embedder), ADR-195 (embedder unification).

## Context

The ruvector workspace already ships a strong family of graph indexes
(`ruvector-roargraph`, `ruvector-diskann`, `ruvector-hyperbolic-hnsw`,
`ruvector-acorn`) and quantizer-backed graph variants. All of them target
the *build-once / query-many* workload pattern that HNSW was designed
for. Three operational pain points still surface for ruvector users:

1. **HNSW under churn degrades.** When a steady stream of inserts and
   deletes is applied, HNSW's layered structure accumulates orphans and
   the upper layers stop being representative. Re-indexing is expensive
   and disruptive.
2. **Deletes are tombstone-only.** Existing ruvector graph backends treat
   delete as a soft mark, which inflates memory and degrades recall over
   time until a compaction pass runs.
3. **No "edit the graph in place" primitive.** Downstream agents
   (`agentic-robotics-rt`, `ruvector-rulake`) want to slot a vector in or
   out at sub-millisecond cost without rebuilding.

The **Dynamic Exploration Graph (DEG)** family (Hezel et al., 2023–2024)
is the SOTA answer to (1)–(3): a single-layer regular graph with a
back-edge-optimisation step on insert and a swap-out delete that needs
no compaction. Reported results on SIFT1M and DEEP10M match or beat
HNSW after ≥ 50 % churn while preserving build throughput.

No crate in the workspace currently implements DEG.

## Decision

Adopt DEG as a first-class index family inside ruvector, shipped as a
new crate `crates/ruvector-deg`. Scope of the first cut:

- **Single-layer regular graph** with fixed out-degree `M` (default 24).
- **RNG-pruning back-edge optimisation** on insert — the textbook
  Relative Neighbourhood Graph occlusion test applied to the candidate
  set returned by greedy search.
- **Swap-out delete** that moves the last node into the deleted slot
  and re-stitches the donor's neighbours via a fresh greedy search.
  O(1) memory reclaim, no tombstone field.
- **Pluggable `Metric` trait** so cosine / inner-product / hamming
  backends can be added without forking the graph code.
- **No-`unsafe` lib** (`#![forbid(unsafe_code)]`).

The crate is feature-gated for an optional `parallel` Rayon build path
but ships a deterministic, single-threaded default so that small-edge
deployments (`ruos-thermal`, MCU shadows) can pick it up unchanged.

## Consequences

### Positive

- Closes the dynamic-update gap in the ruvector graph family.
- Provides a swap-removal delete primitive that the lake-of-vectors
  (`ruvector-rulake`) can call directly on tenant evictions.
- Composable with existing quantizers — `ruvector-rabitq` / `-lvq` /
  `-anisotropic-pq` already implement the right traits to slot a
  re-rank pass underneath DEG.
- Independent of ANN-Benchmark code paths, so it does not regress
  recall numbers in `ruvector-bench`.

### Negative

- Adds a fourth graph index to maintain (alongside HNSW-likes, ROARgraph,
  DiskANN). Documentation cost only — they share no code.
- The naive swap-delete is O(n·M) for the relabel scan. Acceptable at
  PoC scale (< 1 M vectors); production deploys will need a reverse
  adjacency or chunked relabel.
- Insert latency grows with `eps_insert`; the default (80) trades ~12 %
  build throughput for ~11 pp of recall@10 versus the baseline variant.

### Alternatives considered

| Alternative              | Why rejected                                            |
|--------------------------|---------------------------------------------------------|
| Extend `ruvector-acorn`  | ACORN is a *filtered*-search index — different problem. |
| Patch HNSW with deletes  | Tombstones + re-link only delay rebuild, not eliminate. |
| FreshDiskANN port        | SSD-tier, not memory-tier. Future complementary crate.  |
| SymphonyQG (SIGMOD '25)  | Strong but couples quantization choice into the index.  |
| Rebuild on a timer       | Latency spikes are exactly what we want to remove.      |

## Acceptance gates

Hit in this branch (numbers from `cargo run --release -p ruvector-deg
--example sweep` on the build host, see research doc for hardware):

- `cargo build --release -p ruvector-deg` — clean.
- `cargo test --release -p ruvector-deg` — 4 / 4 passing.
- Recall@10 ≥ 0.97 on n=2000, d=64 random uniform with the "recall"
  variant (M=32, eps_insert=160).
- Delete latency < 2 ms/op end-to-end at n=2000, including the donor
  re-stitch search.

## Future work

- Reverse-adjacency table so swap-delete is O(M²) rather than O(n·M).
- `RoarBitmap`-backed visited set for memory-light search at high
  `eps`.
- Out-of-core mode reusing the `ruvector-diskann` page cache.
- Quantizer integration: store `RaBitQ`-encoded vectors and re-rank in
  the last hop of greedy search.
