# ADR-199 — PDX Vertical Block Layout for SIMD-Accelerated Scan

**Status:** Proposed — PoC merged on research branch `research/nightly/2026-06-11-pdx-vertical-layout`. Not yet wired into `ruvector-core`.

**Date:** 2026-06-11

**Crate:** `crates/ruvector-pdx`

## Context

Every IVF, brute-force, and re-rank scan in ruvector ultimately reduces to
"compute distance(query, candidate) for N candidates". Today every such scan
uses the canonical row-major layout: candidates stored as `N` contiguous
`D`-element f32 rows, with an inner loop over the `D` dims of one row before
moving to the next row.

That layout is convenient but has two well-known costs:

1. **SIMD width is wasted on low/mid dimensions.** With D=128 each query
   handles one row's worth of data per SIMD register-pass — eight `f32`
   lanes process eight dims of one row, then we move on. The compiler can
   autovectorise, but the *outer* loop (over candidates) can't be widened.
2. **Pruning is per-candidate, not per-batch.** L2 partial sums are
   monotone; we *could* short-circuit a candidate once its partial sum
   exceeds the worst hit in the heap, but in row-major layout the
   short-circuit lives inside the hot inner SIMD path and tends to defeat
   autovectorisation.

The CWI paper *PDX: A Data Layout for Vector Similarity Search* (Kuffo,
Krimpen-Stoop, Tang, Manegold; SIGMOD 2025) proposes a block-transposed
layout: vectors are grouped into fixed-size blocks (commonly 64 or 128
candidates per block) and stored as `D` consecutive *stripes*, each stripe
holding one dimension across every vector in the block. This flips the
inner loop so that one SIMD-wide step now processes one dim of `BLOCK`
candidates at once — the SIMD width is spent across candidates, not within
a single vector. Pruning then becomes a clean per-block bookkeeping step
("PDX-BOND"): after every `PROBE_STEP` dimensions, sweep the per-lane
partial sums, mark dead lanes whose partial sum already exceeds the heap
threshold, and skip the entire block once every lane is dead.

The published wins on FVS / SIFT1M / DEEP1B are 2–8× over horizontal
brute-force at the same recall, with the BOND pruning variant adding a
further 1.5–3× on clustered (real-world embedding) data. The relevant
question for ruvector is whether the layout itself buys us anything on
Apple Silicon (where NEON has only 4 f32 lanes) and on synthetic data
(where pruning is pessimistic).

## Decision

Land a self-contained `ruvector-pdx` crate that:

* Defines a `Scanner` trait with `search(query, k) -> Vec<Hit>` plus an
  `last_ops()` counter for honest accounting.
* Provides three implementations: `Horizontal` (row-major baseline),
  `PdxVertical::new(rows, prune=false)` (layout-only PDX), and
  `PdxVertical::new(rows, prune=true)` (layout + BOND-style early
  termination).
* Uses fixed `BLOCK = 64`, `PROBE_STEP = 16`. Both are crate-level
  constants and can be tuned later without changing the API.
* Has zero external dependencies — no `rand`, no `criterion`, no `simd`
  crates. The PRNG, the benchmark harness, and the top-k heap are all
  in-tree.
* Ships a `pdx_demo` example and a `pdx_bench` benchmark binary that
  produce comparable numbers without Criterion's bootstrap noise.

Do **not** yet wire PDX into `ruvector-core` IVF / brute-force paths. The
purpose of this ADR is to establish the baseline and let the next nightly
decide whether to (a) integrate as the default flat-scan layout, (b) keep
as an opt-in `Storage::Pdx` for high-dim corpora only, or (c) extend with
RaBitQ-style 1-bit pre-filtering on top of PDX stripes.

## Consequences

**Positive**

* +2.4×–8.8× scan throughput across the three measured configs on an
  Apple M4 Max — the layout win persists even with NEON's narrow SIMD.
* Pure-safe Rust, zero unsafe blocks, zero external deps. Drops cleanly
  into existing workspace.
* The `Scanner` trait is small and storage-agnostic — future variants
  (RaBitQ-on-PDX, IVF-cluster-PDX) implement the same surface.
* The `last_ops()` counter makes pruning savings audit-able: the
  benchmark output reports both throughput and inner-loop MUL count, so
  regressions can be attributed to either the layout or the pruning
  heuristic.

**Negative / risks**

* Pruning is currently *slower* than no-pruning on synthetic
  uniform-random data (~0.6×–0.7×). The bookkeeping overhead dominates
  on data with no cluster structure. We must benchmark on real
  embeddings (GIST/SIFT/MSMARCO) before flipping the default.
* Build cost: PDX layout requires a full copy from row-major input. For
  ingest-heavy workloads we either build PDX lazily or stream-write
  stripes directly. Out of scope for this PoC.
* `BLOCK = 64` is hard-coded. NEON wants 4-lane sweeps; AVX-512 wants
  16-lane sweeps. A future tune-pass should make `BLOCK` a const-generic
  with platform-specific defaults.

## Alternatives Considered

* **PDX without pruning** (layout-only). This is what we report as
  `pdx-vertical` and is the safer default; we keep it as a first-class
  variant rather than a private knob.
* **Brute-force quantization (RaBitQ / LeanVec)** — already covered by
  ADR-193 and the existing `ruvector-rabitq` crate. PDX is orthogonal:
  RaBitQ can in principle sit on top of PDX stripes for a multiplicative
  win. Tracked as a follow-up.
* **GPU-side reorder (CAGRA-Q)** — out of scope for an x86/ARM-CPU
  nightly. Future work once the FPGA/Hailo paths stabilise.
* **Keep row-major; switch to manual `core::simd`.** Doesn't recover the
  outer-loop SIMD width and adds platform conditional code.

## References

* Kuffo, Krimpen-Stoop, Tang, Manegold. *PDX: A Data Layout for Vector
  Similarity Search.* SIGMOD 2025.
* ADR-193 (RaBitQ) — orthogonal compression strategy we may stack on PDX.
* ADR-194 (ONNX embedder throughput) — provides the upstream `D=768`
  workload that motivates the high-dim measurements here.
