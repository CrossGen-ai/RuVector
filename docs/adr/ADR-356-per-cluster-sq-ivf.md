# ADR-356: Per-Cluster 4-bit Scalar Quantization for IVF Residuals

**Status:** Proposed
**Date:** 2026-10-27
**Related:** `research/nightly/2026-10-27-per-cluster-sq-ivf`

## Context

RuVector's IVF variants currently quantize residuals with a *global* scalar
quantizer (SQ): one `(min, max)` range per dimension, fitted across the
whole training set. This is the Faiss / Milvus / Qdrant default. In an
IVF index, however, residuals are centered around each cell's centroid
and span a tighter range than the global dataset. Fitting a per-cell
range recovers that extra precision for free at code-time; the only cost
is a one-time `O(k · d)` table and a per-probe table reload.

Prior art spans the extremes:
* **Global SQ** — one range for the whole dataset. Simple, cheap table.
* **LVQ (VLDB '23)** — one range per *vector*. Highest precision, highest
  overhead (`O(n · d)` metadata).

Per-cluster SQ sits in between: `O(k · d)` overhead, which is amortized
to near-zero at scale.

## Decision

Introduce `PerClusterSq4Backend` as a selectable quantizer for IVF
residuals, behind a new `IvfBackend` trait. The IVF driver (coarse
quantizer, probe selection, id lists) is unchanged; only the per-vector
decode and the quantizer-fitting phase are new.

Interface:
```rust
pub trait IvfBackend {
    fn train(&mut self, data: &[f32], d: usize, assignments: &[u32],
             centroids: &[f32], k: usize);
    fn score(&self, cid: u32, local_idx: usize, query_residual: &[f32]) -> f32;
    fn storage_bytes_per_vec(&self, d: usize) -> f64;
    fn name(&self) -> &'static str;
}
```

Shipped backends:
* `Float32Backend` — baseline, no quantization.
* `GlobalSq4Backend` — current default.
* `PerClusterSq4Backend` — new.

## Consequences

Positive:
* +3.15 pp recall@10 and 21.6 % lower residual reconstruction error vs
  global SQ4 at the same 32 B/vec code budget, measured on a 20 K
  Gaussian-mixture dataset (d=64, k=64, nprobe=8).
* No change to IVF driver, so safe to A/B behind a feature flag.
* Natural upgrade path to Lloyd-Max per cluster (combining ADR-N
  `lloyd-max-sq4` work with this).

Negative / risks:
* Table overhead `k · d · 2 · 4` B must be considered when choosing `k`.
  At `k = 65 536, d = 128` this is 64 MB — fits in RAM but not L2; a
  per-probe prefetch of just the `nprobe` tables we need is the right
  mitigation.
* Online insertion of a point whose dimension falls outside its cell's
  `(lo, hi)` requires either value clamping (lossy) or a cluster-level
  re-encode. Mitigation documented in the research doc.
* Latency gap of ~17 % vs global SQ4 at `n = 20 K` from table reload.
  Expected to vanish once decode is SIMD-vectorized (next ADR).

## Alternatives Considered

1. **Full LVQ (per-vector ranges).** Highest precision but adds `2 · d · 4`
   bytes per *vector* — at d=128 that's a 1024-byte overhead, often
   larger than the code itself. Not justifiable for IVF residuals, which
   are already pre-clustered.
2. **Per-cluster PQ codebooks.** Equivalent idea but on PQ. We separate
   this from PCSQ because SQ has independent production value and far
   simpler ops/debugging (no codebook training per cell).
3. **Global SQ + Lloyd-Max level placement.** Already covered in a prior
   nightly (`lloyd-max-sq4`). Combinable with PCSQ, not a replacement.
4. **8-bit per-cluster SQ.** Easy extension; left to a follow-up because
   4-bit is the storage-interesting regime.

## Rollout

1. Land `crates/per-cluster-sq-ivf` as an experiment crate (this ADR).
2. If reproducing on real embedding benchmarks (SIFT1M, GIST1M, LAION-5B
   subset) confirms the recall gain, promote to `ruvector-sq` crate and
   expose a `QuantizerBackend::PerClusterSq4` variant on the main IVF
   index.
3. SIMD decode (AVX2 + NEON) in a follow-up ADR.
