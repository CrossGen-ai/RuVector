---
adr: 196
title: "DEDRIFT — Incremental IVF Rebalancing Under Content Drift"
status: proposed
date: 2026-06-07
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-194]
tags: [ivf, ann, vector-search, drift, dedrift, rebalance, nightly-research]
---

# ADR-196 — DEDRIFT: ruvector's Incremental IVF Rebalancer

## Status

**Proposed.** Implemented on branch
`research/nightly/2026-06-07-dedrift-ivf-rebalance` as
`crates/ruvector-dedrift`. All unit tests pass; `cargo build --release -p
ruvector-dedrift` and `cargo test -p ruvector-dedrift --release` are green;
end-to-end benchmark numbers reproduced byte-for-byte on Apple M4 Max,
rustc 1.89.0.

## Context

ruvector now ships a real IVF family (`ruvector-rairs`, ADR-193) and a
production-grade quantizer story (`ruvector-anisotropic-pq`, ADR-194). The
remaining gap is **online maintenance**: every IVF in the literature
degrades under content drift, and ruvector has no incremental
rebuild path. Today the only options are:

1. Keep the stale index — quality drifts with the data.
2. Schedule a periodic full retrain — expensive and disruptive.

Neither is acceptable for streaming workloads (LAION-scale moderation,
RAG over evolving corpora, recommendation systems with daily content
turnover). The literature has a clean answer: **DEDRIFT** (Baranchuk
et al., ICCV 2023, arXiv:2308.02752) shows that incremental list-level
maintenance recovers the full-rebuild recall at 1–2 orders of magnitude
lower cost.

This ADR introduces a from-scratch Rust implementation of DEDRIFT's three
core policies (`Split`, `Lazy`, `Hybrid`) plus a `FullRebuild` baseline,
exercised on a deterministic Gaussian-mixture drift simulator with real
measured numbers.

## Decision

**Adopt DEDRIFT-style incremental rebalancing as ruvector's standard IVF
maintenance story.** Concretely:

1. Land `crates/ruvector-dedrift` as a *research-grade* crate under the
   existing `nightly` umbrella. It contains a plain-IVF impl, all three
   policies, a `FullRebuild` baseline, and a reproducible benchmark
   harness.
2. Make **Lazy** the recommended default policy for online IVF
   maintenance. Measured recall@10 matches FullRebuild (0.990 vs 0.989)
   at **23× lower maintenance cost** (1.4 ms vs 32.2 ms over 10 drift
   steps, see research doc for the full table).
3. Promote `ruvector-dedrift` to a mainline crate that hooks into
   `ruvector-rairs` storage **after** items 1–5 in the research doc's
   "What to improve next" section land (Merge policy, token-bucket
   scheduler, PQ-compressed lists, real-data benchmarks).

### Public API (frozen for this PoC)

```rust
use ruvector_dedrift::{Ivf, dedrift::{Policy, PolicyConfig, apply}};

let mut ivf = Ivf::new(dim, n_lists);
ivf.train(&training_vectors, /*iters*/ 8, /*seed*/ 0);
for v in stream {
    ivf.add(&v);
    if step % 100 == 0 {
        apply(&mut ivf, Policy::Lazy, &PolicyConfig::default());
    }
}
```

## Consequences

### Positive

* **Bounded online maintenance.** Lazy is O(lists × dim + drifted_lists ×
  members) per step. For the demo workload (16 lists × 1500 inserts) that
  came in under 0.2 ms per step.
* **No service downtime.** All policies are in-place mutations of the
  centroid array. The vector slab is never copied.
* **Clear cost / recall knobs.** `split_threshold` and `lazy_threshold`
  are scalar floats; production users can sweep them on their own data.
* **Plays well with RAIRS (ADR-193) and anisotropic PQ (ADR-194).** RAIRS
  provides the storage layout DEDRIFT operates on; anisotropic PQ shrinks
  the per-vector cost of Lazy's member-sum pass by 32×.

### Negative / risks

* **No `Merge` policy yet.** Repeated `Split` runs monotonically grow
  `n_lists`. For long-running streams (months, not hours) we need a
  symmetric merge step — listed under "What to improve next" in the
  research doc.
* **Median-based Lazy threshold is noisy for tiny indexes (<8 lists).**
  Production needs a fixed-cutoff fallback for cold-start.
* **Synthetic drift only.** The current benchmark uses a Gaussian-mixture
  simulator. Until SIFT1M-drift and CLIP-on-LAION traces land, the
  measured numbers should be read as *qualitative* evidence that the
  policy works, not as a quantitative production target.
* **No SIMD / no PQ.** The PoC is a scalar f32 reference impl. Production
  needs the items in the research doc's roadmap.

### Operational

* **Default policy:** `Policy::Lazy` with `lazy_threshold = 1.5`.
* **Trigger:** every N inserts where N is `~5%` of the index size, or
  whenever `ivf.drift_score()` exceeds a service-level threshold.
* **Observability:** `PolicyReport { centroids_after, splits_applied,
  lazy_recenters_applied, elapsed_ms }` is returned by every `apply`
  call and should be wired to Prometheus.

## Alternatives considered

1. **Status quo (periodic full rebuild).** Rejected: 23× cost gap at
   matched recall in the PoC, and full rebuilds force a service window.
2. **SOAR-style anti-correlated multi-assignment (Google, 2024).**
   Complementary, not a substitute. SOAR addresses *cell-boundary*
   misses; DEDRIFT addresses *cell-drift* misses. Future work to combine.
3. **Roll a custom "rolling k-means" inside `ruvector-rairs`.** Rejected:
   couples maintenance policy to a specific storage layout. The current
   design keeps `Policy` orthogonal to storage so future ruvector IVF
   variants (RAIRS, IVF-PQ, BetIVF) can all reuse it.
4. **Switch to HNSW for drifting workloads.** Considered. HNSW handles
   inserts gracefully but degrades on *high-prevalence-drift* workloads
   where one cluster grows 10×; HNSW also costs more memory per vector
   than IVF-PQ. DEDRIFT lets us keep IVF's memory profile and handle
   drift incrementally — strictly broader coverage.

## Implementation pointer

Crate:        `crates/ruvector-dedrift/`
Demo:         `cargo run --release -p ruvector-dedrift --bin dedrift-demo`
Bench:        `cargo bench -p ruvector-dedrift`
Research doc: `docs/research/nightly/2026-06-07-dedrift-ivf-rebalance/README.md`
