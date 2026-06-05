---
adr: 196
title: "AIVF — Adaptive IVF with Online Partition Split/Merge"
status: accepted
date: 2026-06-05
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ivf, ann, vector-search, streaming, adaptive, nightly-research]
---

# ADR-196 — AIVF: Adaptive IVF with Online Partition Split/Merge

> **Provenance note.** The Quake / SPFresh / FreshDiskANN / SOAR references
> used in framing AIVF are real lines of literature but specific citations
> have *not* been independently re-verified in this run. The contribution
> here is the implementation in `crates/ruvector-aivf` and the reproducible
> numbers in
> `docs/research/nightly/2026-06-05-aivf-adaptive-split/README.md`. Treat
> AIVF as an original implementation that draws on those ideas, not a port
> of any named paper.

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-06-05-aivf-adaptive-split` as `crates/ruvector-aivf`.
`cargo build --release -p ruvector-aivf` is green, `cargo test --release -p
ruvector-aivf` is green (2 passing tests including a drift-driven split
test), and `cargo run --release -p ruvector-aivf --bin aivf-demo` produces
real benchmark numbers.

## Context

ruvector now has multiple IVF-adjacent index families:

| Crate                 | Role                                              |
|-----------------------|---------------------------------------------------|
| `ruvector-rairs`      | IVF with redundant primary+secondary assignment (ADR-193) |
| `ruvector-betivf`     | β-tunable IVF variant                              |
| `ruvector-anisotropic-pq` | Anisotropic loss PQ codes              |
| `ruvector-rabitq`     | 1-bit ScaNN-style quantisation                    |

**None of these adapt their partition shape online.** Centroids are built
once from a bootstrap sample (typically via k-means) and then frozen. Under
streaming workloads with distribution drift — recommendation tail-shift,
seasonality, fresh content batches — fixed centroids produce two failure
modes:

1. **Hot lists**: a small number of lists absorb most new inserts, blowing
   per-probe latency.
2. **Cold lists**: lists that no longer match the live distribution sit
   empty, wasting probes.

The industry workaround is an offline retrain on a schedule. That's
operationally heavy and leaves a stale window between retrains.

## Decision

Introduce `ruvector-aivf` — an IVF that performs **amortised online
maintenance** of its partition shape:

* Each list carries online Welford statistics (`count`, `sum`, `sse`).
* Every `rebalance_every` inserts (default 1 024), a maintenance pass:
  * **Splits** lists whose size > `split_size` or whose mean radius >
    `split_radius_factor × global_mean_radius` using a local 2-means with
    deterministic furthest-point seeding.
  * **Merges** lists whose size < `merge_size` into their nearest sibling.
* The backing vector store is behind a `Quantizer` trait, so a flat / PQ /
  RaBitQ backend can plug in without touching the index logic.
* Splits never invoke an RNG, so two builds with the same insertion order
  are bit-identical (important for snapshot tests and reproducible
  benchmarks).

The reference workload bundled as `aivf-demo` (drifting 30 000-vector
synthetic Gaussian-cluster workload, 64-d, `nprobe = 3`) produces:

| variant            | lists | search µs/q | recall@10 |
|--------------------|-------|-------------|-----------|
| static IVF         |  64   |    80.6     |  0.990    |
| aivf split-only    |  79   |    48.9     |  0.998    |
| aivf split+merge   |  36   |    80.1     |  0.999    |

Split-only is the new sweet spot here: **+0.008 recall lift, –40 % search
latency** vs the static baseline. Split+merge climbs the recall further
but the merge over-collapses lists with these defaults; merge needs a
`min_lists` floor before it's a sensible production default.

## Consequences

### Positive

* **Streaming-stable recall.** Recall doesn't decay as workload drifts.
* **Lower search latency in split-only mode.** Finer-grained lists make
  each probe cheaper without changing `nprobe`.
* **No global rebuild.** Maintenance is fully online, amortised, and
  bounded by `max_lists`.
* **Pluggable backend.** Composes with existing quantisers via the
  `Quantizer` trait; no need to fork ruvector-rabitq / -anisotropic-pq.
* **Deterministic.** No RNG inside the index → reproducible snapshots,
  easier regression tests.

### Negative

* **Insert cost up ~20 %** for the maintenance amortisation (measured;
  see the research doc).
* **Merge bookkeeping is approximate.** Merged-list SSE is an
  over-estimate; over many merges the radius signal degrades. Fix
  documented in the "What to improve next" section.
* **No delete path yet.** Insert-only; deletes need tombstones + a
  compaction pass (SPFresh-style) before this is a drop-in replacement
  for the static IVF backends.
* **No multi-threading.** Per-list locking and a stop-the-world rebalance
  are the obvious next steps.

### Neutral

* AIVF and RAIRS (ADR-193) are orthogonal — one adapts the *shape* of
  lists, the other adapts the *assignment* per vector. They should
  compose, but the composition is future work.

## Alternatives considered

1. **Offline periodic retrain.** Status-quo in Milvus / Qdrant. Operationally
   heavier; leaves stale windows; loses the "no rebuild" guarantee.
2. **Quake-style query-cost-driven rebalance.** Strictly better as a final
   design but requires per-list query counters and a cost model. AIVF's
   data-distribution triggers are the cheaper first step; query-cost
   triggers are listed as the next iteration in the research doc.
3. **LSM-IVF.** The sibling upstream branch
   `research/nightly/2026-06-05-lsm-vector-index` explores layering IVF
   under LSM compaction. AIVF is *intra-level* adaptation — it can run
   inside one LSM level. The two ideas are complementary, not competing.
4. **Switch to a graph-based index (HNSW/DiskANN) and forget IVF.** Loses
   IVF's compactness and predictable memory layout, which the upcoming
   `ruvector-server` storage tier depends on.

## Reproducing

```
git checkout research/nightly/2026-06-05-aivf-adaptive-split
cargo build  --release -p ruvector-aivf
cargo test   --release -p ruvector-aivf
cargo run    --release -p ruvector-aivf --bin aivf-demo
```

All numbers in this ADR come from the last command. Fixed seed
`0xA1ABF107`, single-thread, Darwin arm64.
