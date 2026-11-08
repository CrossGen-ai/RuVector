# ADR-357: Radial Head-Tail Quantization for IVF Cells

**Status:** Proposed
**Date:** 2026-11-08
**Nightly research:** `docs/research/nightly/2026-11-08-ivf-radial-head-tail/`
**Crate:** `crates/ruvector-ivf-radial-head-tail`

## Context

IVF posting lists store residuals `r_i = x_i - c` of each cell. Current
RuVector IVF backends (`per-cluster-sq-ivf`, `ivf-pfor-postings`, uniform
SQ8) use a single, cell-wide quantizer for every vector in the cell.

But residual magnitude is very non-uniform inside a cell: vectors close to
the centroid have small residuals where 8-bit quantization wastes bits,
while vectors far from the centroid dominate the distance-error budget
and need the full 8-bit precision. A uniform policy either over-provisions
the head or under-provisions the tail.

## Decision

Add a new IVF backend, **IVF-RHT**, that quantizes each cell as two
segments split by radial distance from the centroid:

- **Head** (fraction `h`, default 0.5, closest to `c`): packed **SQ4**
  (4 bits/dim), fit per-segment min/max.
- **Tail** (fraction `1-h`, farthest from `c`): **SQ8** (8 bits/dim), fit
  per-segment min/max.

Query time supports three scan modes:
1. `Full` — scan head + tail.
2. `HeadOnly` — scan head only (fastest, lowest recall).
3. `Adaptive` — if `‖q−c‖ > 0.5 · head_radius`, scan both; otherwise head
   only. The idea: queries sitting close to a centroid are unlikely to
   have NNs in that cell's tail.

Memory per cell for dimension `D` and `N_cell` vectors:
`SQ8-uniform: N_cell · D bytes` vs
`RHT: h·N_cell · ⌈D/2⌉ + (1−h)·N_cell · D bytes`
→ at `h=0.5`, `(0.5·D/2 + 0.5·D)/D = 0.75·D`, i.e. **25% theoretical
reduction**. Measured reduction with per-segment params overhead: 17.5%.

## Consequences

**Pros**
- 17.5% RSS reduction over uniform SQ8 (20K × 64, 128 cells, measured).
- `Adaptive` mode gives 1.19× speedup over `Full` at **no measured recall
  loss** on the mixture-of-gaussians benchmark.
- Trait-based (`Quantizer`) — Sq4/Sq8 can be swapped for product or
  anisotropic quantizers later.
- Interoperable with other IVF probing heuristics (ADR-355 weaver fabric).

**Cons**
- Full-mode QPS is lower than uniform SQ8 (12.9k vs 26.0k) because Sq4
  decode has a nibble-unpack branch per dimension. A SIMD packed decoder
  (AVX2 pshufb or NEON tbl) closes most of this gap but is deferred.
- Recall@10 drops 3.9 pp vs uniform SQ8 at the measured configuration —
  the Sq4 head is the error source. Mitigations:
  - Shrink `head_fraction` so only very-near residuals get 4-bit.
  - Add a float32 rerank of the top-K·α shortlist (cheap).
- Per-segment scale/offset doubles the fit metadata per cell (~1 KB extra
  at D=64, 128 cells). Negligible vs code bytes.

## Alternatives considered

1. **Uniform SQ8** (status quo): higher recall, 17.5% more memory.
2. **LVQ-style single-byte residual quantization** (gist-lvq-2026):
   compresses to 1 byte/dim uniformly, but still cell-wide.
3. **Per-cluster-sq-ivf** (existing crate): fits SQ per cell but uniform
   bit depth within cell. RHT is strictly a refinement.
4. **4-bit PQ with LUT16** (gist-pq-fastscan-lut16-2026): deeper
   compression but higher build cost and more quantization error; a
   different axis.

RHT differs from all of the above by making bit depth **intra-cell
heterogeneous along the radial axis** — a dimension none of the prior
backends exploit.

## Open questions / follow-up

- SIMD Sq4 unpack (AVX2 `pshufb` + `pmullw`, or NEON `tbl`).
- Learned `head_fraction` per cell from radial distribution histogram.
- Combine with ADC rerank: Sq4 head as pre-filter, Sq8 tail as exact.
- Compose with `ivf-radius-early-terminate` (gist-ivf-radius-…-2026) so
  that `Adaptive` skips entire cells not just tails.
