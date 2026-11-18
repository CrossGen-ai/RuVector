# ADR-361: NF4 (NormalFloat-4) quantization for ANN database vectors

- **Status**: Proposed; trait-based PoC landed in `crates/nf4-quantization`
- **Date**: 2026-11-18
- **Related**: ADR-282 (nightly research quality gate), ADR-359
  (shared PQ/IVF clustering), nightly research doc
  `docs/research/nightly/2026-11-18-nf4-quantization/README.md`

## Context

The project ships many quantizers — RaBitQ, PQ-FastScan, LVQ, SQ4/8,
cascade-int4, Hadamard SQ — but no quantile-shaped scalar quantizer.
Uniform SQ-4 is the default 4-bit scheme and treats the data as if
each code point were equally probable, which is not true for the
near-Gaussian embedding vectors produced by modern encoders. QLoRA
introduced NF4 for *LLM weight* quantization in 2023; nothing in the
tree applies it to *database vectors* for ANN search, even though the
derivation carries over directly. Prior nightly topics covered many
PQ/IVF/HNSW/RaBitQ variants (60+ gists in the parent nightly folder)
but none ported NF4 to the retrieval path.

## Decision

Add `crates/nf4-quantization` with a `Quantizer` trait and three
production-shaped swappable variants:

| Variant          | Codebook                              | Per-vector header | Use case                                   |
|------------------|---------------------------------------|-------------------|--------------------------------------------|
| `FixedNf4`       | 16 quantiles of N(0,1), rescaled      | one global scale  | memory-tight, dataset ≈ Gaussian           |
| `AsymmetricNf4`  | QLoRA's original NF4 (appendix E.2)   | max-abs scale     | shifted / sparse-ish embeddings            |
| `LearnedNf4`     | 1-D Lloyd-Max fit on a sample         | min+max → scale,bias | non-Gaussian data, best recall / MSE    |

The trait fixes asymmetric L2 distance so downstream indexes (IVF, HNSW,
SPANN) consume all three identically. Packed `Code4` storage holds two
4-bit nibbles per byte, so a 128-D vector encodes to 64 B + 8 B header
= 72 B, matching uniform SQ-4 byte-for-byte.

## Consequences

**Positive**

- Drop-in replacement for uniform SQ-4 at the exact same compression
  ratio, with measurably better recall and MSE on near-Gaussian data
  (benchmark: +0.75 pp recall@10, −29 % MSE, 1.49× faster scan —
  see nightly research doc).
- Trait-based interface composes with existing PQ, IVF, and SOAR
  pipelines without touching those crates.
- `LearnedNf4` removes the "must be Gaussian" assumption entirely,
  so operators can run it on any distribution (post-ReLU features,
  bimodal L2-normalized shells).
- Learned codebook adds four `f32` lookups per step in a 64-byte LUT
  that fits in one L1 cache line.

**Negative**

- Encode throughput is ~10× slower than uniform SQ-4 because each
  scalar needs a 4-step binary search in `levels[]` (one-time ingest
  cost; query path is unaffected).
- `FixedNf4` requires dataset-level statistics (global σ) that must be
  estimated once up front; stale on heavy drift.
- `LearnedNf4` needs a representative training sample; streaming
  deployments must schedule periodic refits.
- Not competitive below 2 bits — RaBitQ and its extended variants
  remain the right choice at 1–1.5 bits.

## Alternatives considered

- **LVQ-4 (Aguerrebere et al., VLDB 2024).** Uses 4-bit uniform codes
  with per-vector scale+bias. Same bytes, different codebook shape.
  LVQ-4 is strictly a subset of `LearnedNf4` with the codebook pinned
  to uniform; we retain LVQ's scale+bias layout but replace its grid
  with a fitted one.
- **Cascade int4 residual (gist-sq-cascade-int4-residual-2026).**
  Already shipped; targets a different point on the recall/bytes
  curve (two-tier int4). Orthogonal to NF4 and composable: a future
  variant could use NF4 for the base level and uniform for the
  residual.
- **Extended RaBitQ (gist-extended-rabitq-multibit-2026).** Multi-bit
  randomized quantizer with analytic error bounds. Trades recall
  guarantees for distribution-specific optimality; NF4-learned
  optimizes empirically for the actual data at hand.
- **8-bit SQ.** Doubles memory for ~1 pp recall gain, so not a
  substitute at the 4-bit operating point.

## Implementation notes

- Files under 300 LOC, no `unsafe`, no FFI, pure Rust.
- `Codebook16::fit_lloyd_max` is exposed for operators who want to
  pre-compute codebooks offline and freeze them.
- `Code4::storage_bytes()` reports the exact on-disk cost so IVF /
  SPANN write paths can size posting lists deterministically.
- All three variants pass round-trip MSE bounds in unit tests; the
  asymmetric-distance kernel is cross-checked against a decode-then-
  measure reference so downstream indexes can trust it.
