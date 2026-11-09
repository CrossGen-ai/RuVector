# ADR-358: Cascaded int8 + int4-Residual Scalar Quantization

- **Status:** Proposed
- **Date:** 2026-11-09
- **Scope:** Storage-side vector compression for IVF posting payloads, HNSW node vectors, and rerank buffers.

## Context

ruvector already carries several scalar- and bit-level quantizers:
int8 SQ, Lloyd-Max SQ4, μ-law companded SQ, variance-adaptive SQ, Hadamard
SQ, and bit-sliced Hamming. Integer-level int8 is the workhorse because it
gives 4× memory reduction vs fp32 with modest recall loss (~2 pp at k=10 on
Gaussian-like data). The next rung — int16 — doubles storage but recovers
nearly all the recall. There is no built-in codec that sits at the 1.5× int8
storage point with roughly int12 precision, which is exactly where "fast
rerank buffer" workloads want to live: the extra 50 % over int8 is cheap
RAM, the latency is dominated by RAM-bandwidth already, and the recall gap
to fp32 vanishes.

## Decision

Add `CascadeInt8Int4` as a new `Codec` variant: an int8 per-dim SQ base
plus a packed signed int4 per-dim quantization of the residual. Storage is
exactly `D + ceil(D/2)` bytes per vector. Training is a two-pass scan over
the dataset with no codebook learning. Trait surface: `Quantizer::{train,
encode, decode, bytes_per_vector}`.

Reference crate: `crates/ruvector-sq-cascade-int4` (standalone workspace;
production integration into `ruvector-edge-quant` deferred to a follow-up
PR that threads the `Codec` enum through IVF posting and HNSW payload paths).

## Consequences

**Positive:**
- Pareto point between int8 and int16 the repo currently lacks.
- On Gaussian D=128 N=50 k, recall@10 goes from 0.977 (int8) to 0.998 at 1.50×
  int8 storage and 196× lower MSE/dim. Full numbers in the research doc.
- Zero training machinery: two linear scans. Safe to recompute online under
  mild drift.
- Trait design is shared with existing SQ work, so this slots into
  `ruvector-edge-quant` as another variant rather than a new abstraction.

**Negative / costs:**
- Encode cost ~7× int8-only because encoding redecodes the base to compute
  residuals. Decode ~4× int8-only. Both are still sub-microsecond.
- No SIMD in the reference impl; production will want NEON/AVX2 inner loops.
- Storage is 1.5× int8; not appropriate for memory-pinched edge use where
  int8 is already the ceiling.

## Alternatives considered

1. **Int16 scalar quantization.** Simplest path to recall. Rejected as the
   default because it doubles storage for less than 5 % extra recall over
   cascade on Gaussian data.
2. **Residual VQ (RVQ) with a 16-entry codebook per stage.** Similar effective
   precision but requires k-means training and incurs a lookup per dim at
   decode time. Heavier for marginal gain over scalar cascade on
   near-Gaussian data; retained as a separate codec option for
   heavy-tailed distributions (tracked under `ruvector-rvq-cascade`).
3. **LVQ-style per-vector adaptive int8.** One scale per vector rather than
   per dim. Faster to train, worse per-dim precision on anisotropic data.
   Orthogonal; could be combined with cascade (per-vector base scale, per-dim
   residual scale).
4. **Do nothing.** We already have variance-adaptive SQ and μ-law SQ. Both
   improve over flat int8 but neither reaches cascade's recall at <2×
   storage.

## Notes

- The nightly research doc with full benchmark methodology, failure modes,
  and a production integration proposal lives at
  `docs/research/nightly/2026-11-09-sq-cascade-int4-residual/README.md`.
- No workspace Cargo.toml changes: crate is a detached workspace, matching
  the exclude-list convention already used by `sonic-ct`, `ruos-thermal`, and
  others in this repo.
