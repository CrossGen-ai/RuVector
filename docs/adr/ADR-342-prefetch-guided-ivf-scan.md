# ADR-342: Prefetch-Guided IVF Scan — Portable Software Prefetch, Gated on Access Pattern

## Status

Proposed. Standalone experimental crate
(`crates/ruvector-prefetch-guided-ivf-scan`), not wired into any
production IVF path. Companion nightly research
`docs/research/nightly/2026-08-31-prefetch-guided-ivf-scan/` carries the
benchmark methodology and raw numbers.

## Context

RuVector's IVF-flat and IVF-PQ scan paths (`ruvector-turboquant`,
`ruvector-rabitq`, and the future `ruvector-ivf-flat` split) are
memory-bound at production dims. On contiguous row-major posting lists
the hardware stream prefetcher on modern cores (M-series, Zen 4,
Sapphire Rapids, Graviton3) hides DRAM latency well. On *strided* or
*gathered* access patterns — the realistic case for multi-cluster
fanout, IVF-PQ code lookup interleaved with residual scan, or
DiskANN's SSD-sector spill — the hardware prefetcher is blind and
every distance computation stalls on a cold cache line.

FAISS ships a hand-tuned `_mm_prefetch` at fixed lookahead 1 across
every scan (no aarch64, no gating). Qdrant 1.14 added `__builtin_prefetch`
in the aarch64 path that silently no-ops when the compiler can't lower
it. Milvus / Knowhere shipped a per-list prefetch tuning knob in 2026-Q1
with Xeon-only numbers. None of these ecosystems ships a
*portable-across-x86_64-and-aarch64* adaptive-lookahead prefetch
abstraction on stable Rust — which is the gap this ADR closes.

The write-up's headline finding is a negative-result-turned-guardrail:
software prefetch is neutral-to-harmful for contiguous scans on M4 Max
(≤ 18 % slowdown at dim=512) because the HW prefetcher already saturates
the L2→L1 pipe, but wins 1.36× – 3.90× on strided scans at dim ∈
{128, 256}. **Software prefetch belongs behind a `ScanMode` gate, not
compiled in unconditionally.**

## Hypothesis

```text
Given a contiguous posting list of `n` f32 vectors of dimension `dim`
scanned to produce a top-k in L2 distance,

when we issue one `PRFM PLDL1KEEP` (aarch64) / `PREFETCHT0` (x86_64)
per cache line of vector `i + K` before computing distance for
vector `i`, with `K` picked from vector byte size,

then on strided/gathered access patterns (where the hardware stream
prefetcher cannot help) we observe a monotonic speedup in ns/query
that saturates near the compute floor for K ≥ 4, while on
contiguous-linear scans (where the hardware prefetcher is already
saturating DRAM bandwidth) we observe a small regression in ns/query
proportional to the number of prefetches issued per vector.

The gate: production callers select `ScanMode::Contiguous` (SW
prefetch off) vs. `ScanMode::Strided` (SW prefetch on with adaptive
lookahead) based on their known access pattern, not compile-time
feature flags.
```

## Decision

Ship `crates/ruvector-prefetch-guided-ivf-scan` as a small stable-Rust
crate providing:

- `prefetch::prefetch_read<T>(ptr, locality)` — one call site, lowers
  to one machine instruction per architecture. `#[cfg]`-selected:
  `core::arch::x86_64::_mm_prefetch` on x86_64,
  `core::arch::asm!("prfm pldl1keep, [{p}]", ...)` on aarch64, no-op
  on other targets. Marked `nostack, preserves_flags, readonly` so
  the optimizer keeps its full freedom around the prefetch site.
- `prefetch::adaptive_lookahead(bytes_per_vec, budget_lines)` —
  computes lookahead from vector byte size, clamped to `[1, 32]`.
  Deterministic and unit-tested at the boundaries.
- `scan::scan_no_prefetch`, `scan_fixed_prefetch(la)`,
  `scan_adaptive_prefetch` — contiguous variants over
  `PostingList = (dim, Vec<f32>)`.
- `scan::scan_strided_no_prefetch(stride)`,
  `scan_strided_prefetch(stride, la)` — gather variants that simulate
  IVF-PQ cross-cluster fanout. Prime-stride coprime-to-`n` walks visit
  every element once with a hardware-prefetcher-hostile order.
- `scan::TopK { push, into_sorted }` — bounded linear-scan max-list
  shared by all variants; identical top-k across variants is a
  correctness invariant enforced in every integration test.

`PostingList` is a `(dim, Vec<f32>)` newtype so RuVector's existing
storage layers (turboquant, rabitq) can adopt these kernels without a
storage-layout refactor.

The crate is a peer of the receipt / rabitq / turboquant crates: no
runtime dependencies (not even `rand` — internal xorshift64 in the
bench), stable Rust only.

## Consequences

**Positive**

- **First stable-Rust portable prefetch abstraction across x86_64 and
  aarch64 in the RuVector workspace.** No nightly, no external crate.
- **Real-numbers verdict on a common perf-mythology question**: SW
  prefetch is not a free win. Contiguous scans on aggressive
  prefetchers get slower. The crate documents this and gates it.
- **Up to 3.9× measured speedup** on strided IVF-PQ-style scans at
  the sizes RuVector will hit in production (dim=128, n=200 000).
- **Correctness-preserving by construction**: prefetch is architecturally
  side-effect-free, and every scan variant produces byte-identical
  top-k on distinct-distance data (tested).
- **Trivially adoptable**: 3 function signatures, one type. IVF-flat
  and IVF-PQ crates can import and add a `ScanMode` field without
  refactoring their storage layout.

**Negative**

- **Cannot help contiguous scans on modern cores**; the ADR delivers
  a guardrail, not a universal speedup. Callers who blindly wire this
  everywhere will regress.
- **Fixed lookahead requires tuning per platform**. `adaptive_lookahead`
  is a reasonable default but leaves ~2 % on the table vs. hand-picked
  `la=8` at dim=128 strided.
- **Aarch64 inline asm is stable but architecture-locked**. A future
  RISC-V or Loongarch port needs its own `PRFM`-equivalent case (both
  targets have `PREFETCH` variants; adding them is one `cfg` arm).
- **Only measured on Apple M4 Max**. Xeon and Graviton behavior is
  expected to be qualitatively similar based on published prefetch-
  stream widths, but not yet confirmed.
- **Speculation-window sensitivity**: aggressive callers with long
  dependency chains in the loop body will not see the full speedup.

## Alternatives Considered

- **`std::intrinsics::prefetch_read_data` (nightly).** Rejected — the
  RuVector workspace pins stable, and forcing a nightly toolchain on
  every contributor for a one-line intrinsic is a bad trade.
- **`prefetch` crate.** Rejected — 4-year-old, aarch64 path relies on
  `__builtin_prefetch` via `cc`, which introduces a C compiler dep
  and can silently no-op depending on how `cc` was configured.
- **Compile-time `#[cfg(feature = "prefetch")]` gate.** Rejected —
  puts the decision at build time, but the "should I prefetch here"
  question is per-call-site, not per-binary.
- **HW-prefetcher-only, no SW prefetch.** Rejected — leaves the 3.9×
  strided speedup on the floor and closes the door on future gather-
  heavy paths (multi-cluster IVF, GNN rerank, graph-locality reads).
- **`madvise(MADV_WILLNEED)` at the mmap boundary.** Complementary,
  not a substitute: `madvise` operates on 4 kB / 16 kB pages, this
  ADR operates on 64 B cache lines. RuVector will use both — mmap
  tier for cold pages, this crate for warm-line scan.
- **Cache-oblivious van Emde Boas layout (VLDB 2025).** Complementary:
  a good layout still needs a prefetch hint when the next block
  crosses a boundary the HW prefetcher can't see. Future ADR.

## Follow-Ups

- Wire `ScanMode::{Off, Contiguous, Strided}` into
  `ruvector-turboquant`'s IVF-PQ scan; auto-select `Strided` when the
  scan crosses a posting-list boundary.
- Re-run benchmarks on Xeon Gold and Graviton3 and land a
  hardware-detection table so `ScanMode::Contiguous` can flip to
  "SW-prefetch-on" for cores with less aggressive stream prefetchers.
- Explore combining with NEON-explicit distance kernels; the
  autovectorized `l2_sq` leaves ~20 % on the table at dim=128.
