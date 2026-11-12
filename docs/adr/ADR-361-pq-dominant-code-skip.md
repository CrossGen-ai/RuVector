# ADR-361: PQ Dominant-Code Skip Codec (nightly)

**Status**: Proposed (nightly research — not production)
**Date**: 2026-11-12
**Research**: [docs/research/nightly/2026-11-12-pq-dominant-code-skip](../research/nightly/2026-11-12-pq-dominant-code-skip/README.md)
**Crate**: `crates/ruvector-pq-dominant-code-skip`

## Context

ruvector maintains a growing portfolio of PQ-family codecs
(`ruvector-pq-search`, `ruvector-pq-adc-early-abandon`,
`ruvector-pq-prefix-pruning`, `ruvector-rabitq`,
`ruvector-variance-adaptive-sq`, …). All current codecs store
PQ codes as a dense `N × M` byte matrix regardless of code value
distribution. Natural clustered data — the regime that matters for ANN —
produces strongly skewed per-subspace centroid frequencies after k-means
training. We want to know whether exploiting that skew yields a usable
exact codec, and under what regime.

## Decision

Introduce an exact-ADC codec, **Dominant-Code Skip PQ**, in the nightly
crate `crates/ruvector-pq-dominant-code-skip`:

1. Train vanilla PQ codebooks (k-means++, Lloyd).
2. For each subspace, permute centroids so the mode lives at index 0.
3. Store codes as `(bitmap of non-dominant positions) + (compact
   exception codes) + (offsets)`.
4. At query time precompute `C0 = Σ_m LUT[m][0]`; scan only exception
   positions; add `LUT[m][exc] − LUT[m][0]`.

The scan is bitwise exact w.r.t. flat PQ ADC — zero recall loss. The
codec is implemented as a standalone crate (its own `[workspace]`) with
zero dependencies and `#![forbid(unsafe_code)]`.

## Consequences

**Positive**

* **Exactness.** Across 1 400 measured queries (seven configs × 200
  queries) the top-10 set agreed with baseline PQ in every case.
* **Memory savings.** At `K ≤ 4` the codec reduces code footprint by
  14–42% including the `u32` offsets overhead. This is a real,
  reproducible win for memory-bound deployments.
* **Clean trait integration.** Both `BaselinePq` and `DominantSkipPq`
  implement the same `Searcher` trait, so downstream crates can swap
  backends without API change.

**Negative**

* **Slower scan on CPU.** Measured scan is 1.6–2.7× slower than flat PQ
  across all tested configs on Apple Silicon. The bit-walk loop
  (`trailing_zeros` + exception indirection) adds per-vector latency
  that outweighs the saved LUT touches.
* **Breaks SIMD-packed ADC** (PQFastScan-style). Any future SIMD port
  would need block-column layout (see "What to improve next" in the
  research doc).
* **Mutation cost.** Insertions must append to `exc_codes` and extend
  the `offsets` array; in-place updates are expensive.

**Neutral**

* Standalone nightly crate with its own `[workspace]` — zero risk to
  `cargo build --workspace` at the ruvector root.

## Alternatives considered

1. **Delta-LUT rebase without bitmap** — rewrite `LUT[m][k] -= LUT[m][0]`
   at query time so `code == 0` contributes 0. Saves no memory, saves no
   LUT loads (`lut[0]` is still read), only trims an add per vector.
   Rejected: no memory benefit, marginal scan benefit.
2. **Run-length coding of codes** — groups runs of identical codes
   across *adjacent subspaces of the same vector*. Rejected: subspaces
   are independent after PQ training, so run lengths are tiny.
3. **Nibble-packed PQ (K=16)** — the current industry default (Milvus,
   Weaviate, LanceDB). Orthogonal: our skip codec can be composed with
   nibble packing in a future iteration.
4. **ScaNN anisotropic loss** — adjusts the distance metric, not the
   storage. Orthogonal.
5. **Do nothing.** Rejected: we wanted a measured answer to the
   question "does codebook skew buy anything?". The answer is "yes, on
   footprint; no, on CPU scan speed".

## Deployment path

Promotion into `ruvector-pq-codec` (see production layout in the
research doc) is **gated on a SIMD block-column implementation** that
removes the per-vector bitmap-walk overhead. Until that lands, the
codec stays nightly and is documented as a memory-compact exact codec
for cold-tier IVF postings and disk-resident ANN, not a scan-speed
replacement for `pq_baseline`.

## References

* Jégou, Douze, Schmid (2011). *Product Quantization for Nearest
  Neighbor Search*. IEEE TPAMI.
* Ge et al. (2013). *Optimized Product Quantization*. CVPR.
* André et al. (2015). *PQ fast scan*. SIGMOD.
* Guo et al. (2020). *ScaNN*. ICML.
* Gao, Long (2024). *RaBitQ*. SIGMOD.
* This nightly: `crates/ruvector-pq-dominant-code-skip`.
