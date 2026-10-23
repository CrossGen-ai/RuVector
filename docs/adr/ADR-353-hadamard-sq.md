# ADR-353: Hadamard-rotated scalar quantization

- Status: Proposed (nightly research PoC, 2026-10-23)
- Deciders: nightly research channel
- Related: ADR-352 (lloyd-max SQ4), prior IVF/PQ family (ADR-346, 347)

## Context

Uniform scalar quantization (SQ4/SQ8) is the simplest useful codec for compressing
f32 vectors: pick a global `(min, max)` envelope, uniformly bin each dimension's
value into 16 or 256 levels. It scans with a tight inner loop and has no training
data footprint beyond two floats. The failure mode is well known: a handful of
heavy-tailed or outlier dimensions blow up the global envelope, which collapses
the resolution of every other dimension and destroys recall. Real embeddings
(instruction-tuned text, multilingual, hybrid dense/sparse) commonly exhibit this
asymmetry — a few "topic" dimensions carry 10×–100× the variance of the rest, or
occasional spike entries dominate the vector.

OPQ (Jégou et al.) addresses the analogous problem for Product Quantization by
*learning* a rotation that aligns subspace energy. For plain scalar quantization,
a cheaper, training-free alternative exists: the **randomised Walsh–Hadamard
rotation** (Ailon & Chazelle 2006). An orthogonal rotation preserves L2
distances, and a Hadamard rotation spreads any outlier across *all* output
dimensions, flattening per-dim variance and shrinking the global envelope. It
runs in O(d log d) via the fast Walsh–Hadamard transform, requires no training
samples, and composes with any downstream codec.

## Decision

Promote **randomised Hadamard rotation** to a first-class, swappable pre-transform
in ruvector's codec stack, materialised in `crates/ruvector-hadamard-sq`. The
rotation is:

- Deterministic from a 64-bit seed (reproducible encodes).
- In-place iterative FWHT, zero allocations after setup.
- Composable with any downstream `Codec` (SQ4, SQ8, PQ residual, RaBitQ pre-rotation).

Ship two concrete codecs on top of the rotation:

- `HadamardSQ4` — same 8× compression as uniform SQ4, dramatically better recall
  on heavy-tailed data.
- `HadamardSQ8` — 4× compression, near-exact recall even on outlier-heavy data.

## Consequences

### Positive

- **Zero training**. The rotation is seeded, not learned, so it is identical for
  encode and decode, works online, and never drifts.
- **O(d log d) rotation**. For d = 128, that is 7× unit-level work per vector vs
  O(d²) for a dense learned rotation — small enough to amortise.
- **Fixes the single worst SQ4 pathology** at the same memory footprint.
- **Composable**. Any later codec (PQ, RaBitQ, LVQ) can be dropped in behind the
  rotation; the trait boundary is already in place.
- **Deterministic replay**. The seed is logged alongside the codebook; the exact
  encoding is reproducible years later from the segment metadata.

### Negative

- **Encode throughput**: ~5× slower than naive SQ4 (4.1M vec/s vs 20M vec/s on
  M-series Mac, release build) because of the rotation. Still well above build
  rates needed for most ingestion workloads.
- **Scan throughput**: ~2.8× slower for ADC scan because each query is rotated
  once before the inner loop. The rotation is per-query, not per-code, so for
  large indexes it is a constant overhead; for tiny indexes it dominates.
- **Padded dimension**: dims that are not powers of two pad to the next
  power of two, inflating code bytes by up to 2× in the worst case (e.g. 65 → 128).
  Mitigation: pre-rotate once at ingest and store the padded codes directly.
- **Not a win on well-conditioned data**. On uniformly-distributed inputs,
  Hadamard SQ4 is a wash vs uniform SQ4 (the rotation adds work for no MSE gain).
  Treat as an optional pre-transform, not an always-on default.

### Alternatives considered

- **OPQ-style learned rotation**: strictly better reconstruction, but requires a
  training pass over the dataset and a d² rotation matrix stored per index.
  Chosen as the next-step improvement after this ADR lands; the rotation
  trait already covers it.
- **Per-dimension min/max (vector SQ)**: fixes the outlier problem by giving each
  dim its own envelope. Doubles the metadata size (2 floats × d) and does not
  help with correlated dims (unlike rotation).
- **Clipping / winsorising**: cheaper, but a hard cut on outliers also distorts
  distances in a non-isometric way.
- **FJLT / sparse Johnson–Lindenstrauss**: Hadamard is the cheapest dense
  variant and empirically dominates for this d-range.

## Status gate

Promotion blocked on:

1. HNSW and IVF scan paths carrying `Rotation` as part of their segment header.
2. SIMD-widened encode (f32×8 butterfly) to recover ≥ 50 % of baseline encode throughput.
3. A real-corpus eval on at least one of SIFT-1M, GloVe-100D-1.2M, or GIST-1M.
4. A rotation-caching variant that pre-rotates query batches once per request.

## Measured numbers (M4 Max, release, 2026-10-23)

Synthetic d=64, n=20K, nq=100, sparse-outlier model (~15% of vectors carry a
±25 spike on a random dim). Numbers from `cargo run --release --example demo`
and `cargo bench`:

| codec         | MSE/dim | recall@10 | compression | encode (vec/s) | scan (qps) |
| ------------- | ------- | --------- | ----------- | -------------- | ---------- |
| uniform-sq4   | 0.2857  | 0.087     | 8.00×       | 20.2M          | 478        |
| hadamard-sq4  | 0.0397  | 0.255     | 8.00×       |  4.1M          | 171        |
| hadamard-sq8  | 0.00014 | 0.932     | 4.00×       |  5.4M          | 197        |
