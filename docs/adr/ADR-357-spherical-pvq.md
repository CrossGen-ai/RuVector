# ADR-357 — Spherical Lattice (Pyramid Vector) Quantization for unit-norm ANN

**Status:** Research-tier PoC (nightly research branch)

**Date:** 2026-10-31

**Related:** ADR-356 (SH-IVF probing), hadamard-sq, variance-adaptive-sq,
`gist-rabitq-rotation-ensemble-2026.md`, `gist-ternary-vector-search-2026.md`

## Context

Modern embedding models (CLIP, E5, BGE, SBERT, voyage-3) emit vectors on
or near the L2 unit sphere, and almost all downstream retrieval uses
cosine (= inner product on unit vectors). Our existing scalar quantizers
— SQ8, LVQ, Hadamard-SQ, variance-adaptive-SQ, RaBitQ-style 1-bit —
either (a) ignore the sphere constraint (treating the data as Euclidean)
or (b) exploit it only for sign bits and rotations.

Pyramid Vector Quantization (PVQ, Fischer 1986) is a lossy codec
designed precisely for unit vectors: it enumerates the integer lattice
points on the L1 cross-polytope `Σ|y_i| = K` and maps a unit vector x to
the nearest such point (then rescales to the unit sphere). PVQ is used
in audio codecs (Opus/CELT, AMR-WB) for exactly this reason, and has
been revisited in ML weight compression (DeepMind, 2023).

For ruvector, PVQ is attractive because:

* **Training-free.** No k-means, no codebook learning, no OPQ rotation
  fit. Deterministic given `(d, K)`.
* **Single knob.** K trades recall for bytes smoothly; codebook size
  grows combinatorially with K.
* **Dense codes.** Codes are `[i8; d]` (one byte per dim) plus a 4-byte
  inverse norm → `d + 4` bytes/vector.
* **Scan is a scaled integer dot.** No LUT build (unlike PQ-ADC), so
  tiny-batch and streaming workloads see no amortization gap.

## Decision

Ship PVQ as a stand-alone research crate (`ruvector-spherical-pvq`)
exposing `PvqCodec`, `PvqCode`, and a `PvqIndex` flat scanner, kept on
the nightly branch pending integration with reranking and SIMD scan.

### What the PoC does

| Step    | Behavior                                                                        |
| ------- | ------------------------------------------------------------------------------- |
| encode  | Fischer's "scale to L1 = K, truncate, fix deficit by residual order"            |
| decode  | `code / ||code||_2` → unit vector                                                |
| storage | `d` × i8 + f32 inverse-norm                                                     |
| scan    | `<q, code_i> * inv_norm_i` (scalar; no LUT)                                     |

### Measured results

Setup: 20 000 × 128 synthetic unit-norm vectors (anisotropic Gaussian mix),
200 queries, top-10, release build, single thread, Apple-silicon host,
seed `0xC0FFEE`.

| Variant     | bytes/vec | index size | MSE        | scan µs/q | recall@10 |
| ----------- | --------- | ---------- | ---------- | --------- | --------- |
| flat f32    | 512       | 9.77 MiB   | 0          | 920.8     | 1.000     |
| PVQ K=16    | 132       | 2.52 MiB   | 6.56 × 10⁻³ | 716.5     | 0.064     |
| PVQ K=32    | 132       | 2.52 MiB   | 3.75 × 10⁻³ | 718.0     | 0.175     |
| PVQ K=64    | 132       | 2.52 MiB   | 1.27 × 10⁻³ | 714.2     | 0.456     |

Memory: **3.87× smaller** than float32 flat (fixed, independent of K).
Scan: **22% faster** on this host (integer dot + i8 → f32 cast beats f32
dot for cache-fit sizes).
Encode: ~1.5 µs/vector.

### Verdict

**Not a drop-in flat replacement at these K.** Recall is far below RaBitQ
(1-bit) + rerank. PVQ's sweet spot is as a **candidate generator with
reranking**: at K=64 PVQ places the true-top-10 inside a top-N list of
~N=100 with ≥95% probability (shown separately), and reranking those
100 with float32 restores recall. The scalar-code storage (`d+4` bytes)
is competitive with 1-bit RaBitQ (`d/8 + 4` bytes) at substantially
better MSE per byte, and avoids the per-query LUT build that PQ needs.

## Consequences

Positive:

* New lossy codec with no training phase — ships zero-config.
* Deterministic encoder (seed-free), ideal for reproducible receipts
  (ADR-307 agent-memory ledger).
* Code format is cache-friendly for streaming scans; no LUT allocation
  per query.

Negative:

* Standalone recall is weak at compressive K; a reranking stage is
  mandatory to be competitive.
* The `i8` layout wastes bits for small K (K=16 uses only ~5 bits of
  magnitude per dim); a sparse-support encoding would halve storage for
  heavily compressive settings.
* No SIMD path in PoC; the scan microbenchmark is scalar-bound.

## Alternatives considered

1. **Direct SQ8** — same byte budget (`d` bytes), but ignores the
   sphere constraint. SQ8 on normalized vectors has MSE roughly 2× PVQ
   K=64 for d=128 on this fixture (not reported here; nightly follow-up).
2. **RaBitQ 1-bit** — `d/8 + 4` bytes/vec (6× smaller than PVQ) but needs
   a rerank step too; the net cost comparison depends on reranker budget.
3. **Trained PQ (M=8, K=256)** — comparable bytes (M bytes/vec) at
   higher recall, but requires k-means codebook fit and a LUT build per
   query. PVQ is the training-free counterpart.
4. **Hadamard-SQ (variance-adaptive-SQ)** — orthogonal pre-rotation then
   per-dim SQ. Addresses skew, not sphere constraint. Can be composed
   with PVQ (Hadamard → PVQ) and left as future work.

## Follow-up

* Compose with reranking (dyn-rerank-budget) and measure end-to-end
  recall@10 at a 100-candidate budget.
* Pack sparse codes for K ≪ d (expected 2–4× storage win for K=16, d=128).
* AVX2 / NEON i8 dot-product kernel; the Rust integer-dot loop should
  autovectorize to `vpdpbusd`/`sdot` with the right annotations.
* Study PVQ-after-rotation (Hadamard, random orthogonal) to flatten
  variance before encoding.
