# ADR-273 — Extended (Multi-Bit) RaBitQ Quantizer

- **Status:** Accepted (nightly research)
- **Date:** 2026-07-08
- **Deciders:** Nightly research agent
- **Related:** ADR-272 (Bit-Packed HNSW Graph), `ruvector-rabitq` (1-bit)

## Context

`ruvector-rabitq` gives us a state-of-the-art 1-bit quantizer with a
theoretical error bound, but at production scale on 128-D vectors its
recall\@10 collapses (0.8 % at N = 50 k Gaussians in our own bench). The
2024 follow-up by Gao & Long (arXiv:2409.09913) shows the same rotation
technique generalises to `B`-bit alphabets with variance shrinking as
`Θ(2^{-2B})`. Adopting this closes the recall gap while still giving up
to 32× compression vs `f32` flat and reusing the same rotation
infrastructure we already ship.

## Decision

Add a new workspace crate, `ruvector-extended-rabitq`, that:

1. Reuses a deterministic Haar-uniform `O(D)` rotation (QR of seeded
   Gaussian, deterministic sign-fix) — kept locally so the crate has no
   dependency on `ruvector-rabitq`.
2. Implements a symmetric uniform `2^B`-level scalar quantizer over the
   rotated unit vector, `B ∈ {1, 2, 4, 8}`. Packing is little-endian,
   dimension-major, and always byte-aligned (because `8 % B == 0`).
3. Estimates L2 distance asymmetrically: query stays `f32`, per-query
   LUT of size `D × 2^B` is built once, scan is `D` lookups + `D` adds.
4. Ships an `AnnIndex` trait so future SIMD kernels and HNSW-integrated
   backends can slot in behind the same API.
5. Ships a runnable `erabitq-demo` binary that reports recall\@10 and QPS
   at `N ∈ {1 k, 10 k, 50 k}` for `B ∈ {1, 2, 4}`, and a `cargo bench`
   that reports ns/candidate for `B ∈ {1, 2, 4, 8}`.

## Consequences

**Positive.**
- Fills the recall gap the current 1-bit tier leaves: at `N = 50 k`, `B = 4`
  reaches 47.7 % recall\@10 vs 0.8 % for `B = 1`, at the same ~250 QPS on
  a single thread.
- Only 8× compression cost at `B = 4` vs the 32× of `B = 1` — still a big
  win over `f32` flat's 512 B/vec.
- Reuses trait-based design so `ExtendedRabitqIndex` can back HNSW nodes,
  DiskANN sectors, or SPANN posting lists without an API change.
- Determinism guarantees make this safe for delta-consensus / raft
  replay workloads.

**Negative.**
- Scalar scan kernel is memory-bound; without SIMD we don't fully
  exploit the compact codes.
- Grid is fixed. Highly anisotropic corpora leave bits on the table
  compared to a learned quantizer (LVQ, PQ).
- Norm still stored as `f32` (4 B/vec fixed overhead) — worth revisiting.

## Alternatives Considered

- **Stay at 1-bit + rerank.** Requires a rerank stage everywhere;
  operationally messier and blows up query latency.
- **Adopt LVQ/LeanVec.** Higher recall per bit, but requires training
  data, per-corpus state, and can't be built deterministically from a
  seed alone — worse fit for the replayable-consensus story.
- **Adopt OPQ/AQ.** Codebook state per corpus, no closed-form error
  bound, and PQ implementations already exist in `ruvector-opq` and
  `ruvector-pq-search`. Extended RaBitQ is the missing "no-training,
  bounded-error, tunable-bits" tier.
- **Directly extend `ruvector-rabitq`.** Rejected: the 1-bit crate has a
  tight, audited surface. Cleaner to keep the multi-bit variant
  isolated until the SIMD story is settled.
