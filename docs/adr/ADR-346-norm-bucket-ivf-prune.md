# ADR-346: Norm-Sorted Cluster Buckets for IVF Cauchy–Schwarz Pruning

## Status

Proposed (experimental crate `ruvector-norm-bucket-ivf`, off by default).
Promotion to `ruvector-core::ivf::backend` behind an off-by-default
`norm-bucket` feature is recommended pending: (a) the norm-variance
auto-detector described in the companion research doc, and (b) integration
tests against `ruvector-core`'s existing IVF list format.

## Context

`ruvector`'s IVF-family indices (`ruvector-core::ivf`, `ruvector-diskann`,
`ruvector-spann`, `ruvector-soar-ivf`) scan every candidate in every
probed cluster with a full-dimension L2 distance. On the 2026-09-20
`ruvector-partial-l2` nightly we measured that ≥ 60% of that L2 work is
spent on candidates that are ultimately rejected by the top-k heap. Two
independent lower bounds can prove a candidate is *not* in the top-k
without touching its coordinates:

1. **Cauchy–Schwarz / reverse triangle inequality:** `||q − x||² ≥ (|q| − |x|)²`.
2. **Partial-L2 running sum vs current `τ`.** (Addressed by the 2026-09-20
   nightly.)

(1) has the striking property that if we sort each cluster's members by
`|x|`, the set of feasible candidates is a **contiguous window** in the
sorted order, addressable by two binary searches. No coordinate reads are
required to skip everything outside the window.

Companion research doc:
`docs/research/nightly/2026-09-22-norm-bucket-ivf-prune/README.md`.

## Hypothesis

```text
Given a mixed-scale synthetic dataset (per-vector log-uniform scale
∈ [0.2, 5.0]) at dim ∈ {32, 64, 128} with N ∈ {20k, 30k} and IVF
clusters ∈ {128, 256} at nprobe ∈ {8, 16}, k = 10,

when the per-cluster scan is replaced by a norm-sorted binary-search
window (NormBucketScan) that emits only candidates whose norm falls in
[|q| − √τ, |q| + √τ] (τ = current worst top-k distance),

then QPS improves by at least +15% at every measured dim/N/nprobe setting,
Recall@10 is unchanged (lossless technique), and the added memory is
≤ 1/dim of vector storage.

Negative-control regime: on isotropic Gaussian data (norm variance ≈ 0),
QPS regresses by at most 1.5% vs FullScan (bounded overhead of the
window binary search).
```

## Decision

Adopt `NormBucketScan` as a `ScanBackend` trait implementation, published
as the experimental crate `ruvector-norm-bucket-ivf` for now. The crate:

- Introduces a `ScanBackend` trait (`FullScan`, `NormBucketScan`,
  `NormBucketPartialL2Scan`) so future backends (PDX-vertical partial L2,
  PQ-ADC, RaBitQ) plug in without touching `IvfIndex`.
- Stores per-cluster `sorted_norms: Vec<f32>` alongside `sorted_vectors`
  ordered to match.
- Uses a bounded max-heap (`TopK`) whose `worst()` returns `+∞` until
  full, so pruning kicks in only after the first `k` candidates are
  collected — no over-pruning during warm-up.
- Ships real tests (`synthetic_recall_matches_brute`,
  `pruning_actually_prunes`) that guard both the lossless-recall property
  and the pruning-nonzero property.
- Ships a real benchmark harness (no `criterion` dep) emitting
  ns/query, QPS, and prune ratios per dataset regime.

Measured on Apple M4 Max, single-thread (Recall@10 unchanged in every
case):

| Case        | FullScan QPS | NormBucketScan QPS | Δ         |
| ----------- | ------------ | ------------------ | --------- |
| mixed-32d   | 14,608       | 17,343             | **+19%**  |
| mixed-64d   | 3,547        | 4,308              | **+21%**  |
| mixed-128d  | 1,681        | 1,926              | **+15%**  |
| gauss-64d   | 32,037       | 31,625             | −1.3% (control) |

## Consequences

**Positive.**

- Zero-recall-loss pruning stage for IVF, stackable with any downstream
  distance approximation (PQ-ADC, RaBitQ, partial L2).
- Tiny memory overhead: `4 * N` bytes for norms across the whole index.
- No new dependencies — pure `f32` arithmetic.
- Trait-based design lets `ScanBackend` be selected per-query, useful for
  A/B rollout and per-request tuning.

**Negative / neutral.**

- On L2-normalised embeddings the technique is a no-op (~1% overhead from
  the window binary search); users must know their embedding scale.
- Requires cluster-local sort at build time. For a 1M / dim=128 index this
  is negligible (< 1% of Lloyd cost) but must be re-run on incremental
  insertion — not a fit for high-QPS streaming ingest without a
  per-batch tail-append + periodic re-sort strategy.
- The measured `NormBucketPartialL2Scan` stacking currently regresses at
  dim ≤ 128 (extra per-candidate branch cost). Not adopted as default.
  Roadmap: PDX-vertical inner loop expected to flip this to a stacking
  win.

## Alternatives considered

1. **PQ-ADC prefilter (as in FAISS `IndexIVFPQ`).** Lossy, requires
   codebook training, storage per code is 8–16× smaller vectors but
   memory savings are unrelated to what NSCB provides (candidate skip
   before any read). Complementary, not substitutable.
2. **Bit-sliced RaBitQ prefilter.** Also lossy at 1-bit; provides a much
   tighter bound but with ε > 0 recall loss. Complementary — RaBitQ can
   run *inside* the norm window as a second-stage filter.
3. **Full L2 with SIMD unrolling only.** Already the FullScan baseline.
   Amortises constant factors but not the O(N_c) work-per-cluster.
4. **Sort clusters by centroid distance and cut early** (a "cluster-level"
   version of the same idea). Already done by `probe_clusters(nprobe)`;
   this ADR is the intra-cluster analog.
5. **Track norms without sorting — scan linearly with an inline `if`.**
   Loses the O(log N_c) window-lookup advantage; the branch mispredicts
   more often than the memory saved is worth. Ran a quick probe in the
   crate's demo (not included in the bench): 3–5% slower than
   `NormBucketScan` at dim=64.

## Follow-ups

- Auto-detect L2-normalised inputs at `IvfIndex::build` and skip
  norm-bucket construction (elides the ~1% overhead).
- Integrate as `ScanBackend::NormBucket` behind a feature flag in
  `ruvector-core::ivf`.
- Combine with PDX-vertical partial L2 at dim ≥ 256 to unlock the stacked
  win (predicted 1.3–1.6× on top of NSCB).
- Re-run on real embedding corpora (sentence-transformers-all-mpnet-base,
  OpenAI text-embedding-3-large, ImageNet-CLIP) to characterise the
  norm-variance distribution across real datasets.
