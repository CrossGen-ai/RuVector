# ADR-253: Tribase Triangle-Inequality Pivot Pruning for IVF Search

- **Status**: proposed
- **Date**: 2026-06-15
- **Deciders**: nightly-research
- **Tags**: ann, ivf, pruning, triangle-inequality, exact, cpu

## Context

ruvector ships several IVF-style ANN backends (`ruvector-rabitq`,
`ruvector-rairs`, `ruvector-betivf`, `ruvector-acorn`) but none of them
exploits the triangle inequality inside a probed cell. Every probed
list pays one distance computation per resident vector, even though a
small precomputed scalar — the vector's distance to its own centroid —
gives a free, exact lower bound on `d(q, x)`.

Tribase (Liu et al., SIGMOD 2024) reports 1.4× – 3× wall-time speedups
on DEEP1B and SIFT1M by spending **4 bytes per indexed vector** on this
residual and using `|d(q,c) - r(x)|` as a pruning lower bound. The
pruning is *exact* — it skips only vectors that cannot possibly enter
top-k — so recall is bit-for-bit identical to standard IVF.

The technique is orthogonal to existing crates: it composes with
RaBitQ quantization (RaBitQ supplies a probabilistic LB; Tribase
supplies an exact LB; the maximum of the two prunes harder than
either), with ACORN (filters compose, geometry is unchanged), and
with HNSW-style graphs (apply inside a graph "page" of vectors).

## Decision

Add a new workspace crate `ruvector-tribase` that:

1. Implements three trait-comparable search backends — `FlatIndex`,
   `IvfIndex`, `TribaseIndex` — so the contribution can be benchmarked
   in isolation against the obvious baselines.
2. Stores residuals as `Vec<f32>` parallel to ids, sorted ascending
   per list, enabling bisect-at-`dqc` followed by two outward sweeps
   with optimal short-circuit.
3. Uses squared L2 internally (one `sqrt` per list, not per vector).
4. Is CPU-only Rust with two dependencies (`rand`, `rand_chacha`)
   and no `unsafe`.
5. Ships an `examples/bench_pruning.rs` reproducible harness emitting
   real `cargo run` numbers (QPS, average distance computations,
   recall@k, p99 latency) — no mocks.

The crate is **not** wired into `ruvector-core` or `ruvector-server`
in this ADR. It lands as an opt-in research crate so the design can
mature (multi-pivot, RaBitQ composition, SIMD, parallel probing)
before promotion.

## Consequences

Positive:

- ~4 B per vector buys 17 % – 51 % fewer distance computations and up
  to 2.44× QPS on clustered synthetic data, at recall 1.000.
- Pruning is exact (theorem-grade), not statistical — no recall tuning
  knob, no surprises.
- Trivially compositional: residuals are independent of the
  representation (quantized or not), independent of the index
  structure (IVF cell, HNSW page, RaBitQ cluster), and independent
  of the metric so long as it satisfies the triangle inequality.

Negative / accepted limits:

- L2 only; cosine workloads must normalise upstream.
- Pruning collapses on near-uniform high-dim data — observed in the
  bench at large `n` / large `nprobe`, where the per-vector
  bisect/sqrt overhead occasionally exceeds the work saved (0.93×
  in one configuration, while still doing 17 % fewer distance ops).
- Adds O(n log(n/k)) sort cost at build time.

## Alternatives Considered

- **Quantization-only pruning (status quo RaBitQ).** Cheaper per
  prune check but probabilistic; cannot give recall = 1.000
  guarantees and uses 16–32× the per-vector memory.
- **Apply triangle bound on the fly without sorting residuals.**
  Loses the early-break property; degenerates to a constant-factor
  overhead per visited vector with no skip.
- **Embed Tribase directly into `ruvector-rabitq`.** Tempting (the
  two compose) but conflates the contribution and makes A/B
  benchmarking impossible. Keep separate, then expose a
  `TribaseRabitqIndex` in a follow-up ADR once both are mature.
- **Multi-pivot Tribase from day one.** Better pruning, larger
  implementation surface, more places for bugs. Defer to the next
  iteration; ship single-pivot first to lock in the framework
  and the benchmark harness.
