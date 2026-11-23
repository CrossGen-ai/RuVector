# ADR-361: Bit-Plane Progressive SQ8 (BPP-SQ)

**Status:** Proposed (PoC landed on `research/nightly/2026-11-23-bit-plane-progressive-sq`).
**Date:** 2026-11-23
**Deciders:** nightly research routine, to be reviewed by core-index owners.
**Supersedes:** none.
**Supersedes:** none.
**Related:** ADR-level prior art in `ruvector-pdx-vertical` (vertical SQ layout),
`ruvector-sq-cascade-int4` (two-code cascade), `ruvector-bit-sliced-hamming`
(plane-slicing for Hamming only).

## Context

ruvector has accumulated eight nightlies' worth of scalar-quantization
improvements (LVQ, Lloyd-Max SQ4, mu-law SQ, variance-adaptive SQ, Hadamard
SQ, per-cluster SQ, SQ-cascade-int4, Gray-code SQ) and five nightlies' worth
of PQ-adjacent bit-plane work (bit-sliced Hamming, Polysemous PQ, etc.).

The common pattern across all SQ variants is: **one byte per code, one scan
precision**. Query time has no handle to spend less bandwidth when recall
headroom is available, or more bandwidth when the query lands in a confused
neighborhood.

Simultaneously, memory bandwidth — not SIMD throughput — is the dominant
cost of IVF/PQ scans at modern code sizes. Any mechanism that scans *fewer
bytes* for most vectors and spends the full budget only on shortlist
candidates is a strict pareto move.

The design-space gap: **one SQ8 code per vector, physically re-laid as
bit-planes so the scan can pick its own precision**.

## Decision

Introduce an experimental crate `ruvector-bit-plane-sq` that:

1. Fits the usual per-dim min/scale SQ8 codebook.
2. Transposes the resulting u8 code array into 8 packed-bitmap planes
   (MSB-first), one `u64` word per 64 dims per vector per plane.
3. Exposes three search backends behind an `L1Search` trait:
   - `FullSq8` — baseline row-major L1 scan.
   - `TopMApprox(M)` — reconstruct an approximate u8 code from the top-M
     planes with midpoint bias; scan with the baseline kernel.
   - `Progressive(M, r)` — shortlist with `TopMApprox`, refine the top `k·r`
     with `FullSq8`.

The crate is tiny (single `src/lib.rs`, one example, three tests), has
zero external deps, and registers in the workspace as a leaf crate.

### Why this design

- **Correctness-first PoC.** The u8 reconstruction path keeps the search
  kernel identical to baseline, which is how recall was verified at parity
  on the synthetic set (progressive(M≥3, r≥2) matches baseline within
  measurement noise).
- **Trait boundary for future SIMD.** Replacing `TopMApprox` with a
  bitmap-direct `xor`+`popcount`+weight kernel is additive: it implements
  the same trait. The baseline kernel is unchanged.
- **Memory layout is the primary win.** Even without the SIMD kernel,
  storing only M bitmaps (at M=3: 37.5 % of the SQ8 bytes) recovers full
  recall; the refine cost is `k·r` vectors, which is a rounding error.

## Consequences

**Positive.**

- Code footprint at parity recall drops to 25–50 % of baseline SQ8 for
  typical (k=10, r∈[2,8], M∈[2,4]) settings.
- Trait-based search surface slots cleanly into the existing search layer;
  no breaking change to callers.
- The planes layout is also a natural match for streaming / mmap: hot
  planes resident, cold planes paged on demand.

**Negative / costs.**

- Current PoC *does not* lower latency because it reconstructs to u8 before
  scanning. The latency win depends on landing the bitmap-direct kernel.
- Per-query LUT setup (for the midpoint bias) is trivial for L1 but will
  grow for L2 if we extend. L2 extension is not in this ADR.
- Another byte-layout variant adds to the maintenance surface. The counter
  is that it's isolated to its own crate and feature-gated at call sites.

**Neutral.**

- Memory footprint for the full 8-plane store equals row-major SQ8 bytes
  (both are `n·D` bytes). The savings come from choosing M<8.

## Alternatives Considered

1. **Narrower uniform codes (SQ4 / SQ2).** Simpler, cheaper, but applies
   the same precision to every query. BPP-SQ gets the SQ4 footprint for
   non-shortlist work while preserving SQ8 precision on the shortlist.
2. **Two-code cascade (SQ-cascade-int4-residual, already in tree).** Pays
   2× code memory for a two-stage scan. BPP-SQ pays 1× and gets a
   query-time precision dial for free.
3. **Vertical / PDX layout (already in tree).** Orthogonal to BPP-SQ — PDX
   speeds SIMD fusion across dims; BPP-SQ reduces the bytes loaded in
   total. They compose: PDX-of-bit-planes is a valid future layout.
4. **RaBitQ.** Binary codes with an elegant error bound. BPP-SQ is strictly
   coarser (bitmaps of an existing SQ), but gives up nothing because the
   refine step is exact SQ8.
5. **Do nothing.** The main-line SQ8 path is adequate for today's
   benchmarks. BPP-SQ is a bet on bandwidth savings that compound once the
   bitmap kernel lands; without that, it's a no-op except for storage.

## Acceptance

- [x] `cargo build --release -p ruvector-bit-plane-sq` — green.
- [x] `cargo test --release -p ruvector-bit-plane-sq` — 3 tests green.
- [x] Benchmark binary emits real numbers (see research doc for the table).
- [x] At least 3 measured variants (baseline + top-m + progressive × M's).
- [ ] Bitmap-direct SIMD kernel — follow-up ADR + crate change.
- [ ] IVF integration (per-posting planes) — follow-up ADR.
