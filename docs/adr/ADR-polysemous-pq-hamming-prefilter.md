# ADR: Polysemous PQ Codes with Hamming Prefilter + ADC Rerank

**Status:** Proposed (nightly research)
**Date:** 2026-11-21
**Branch:** `research/nightly/2026-11-21-polysemous-pq-hamming-prefilter`
**Crate:** `crates/ruvector-polysemous-pq-hamming-prefilter`

## Context

RuVector ships several Product Quantization variants
(`ruvector-pq-search`, `ruvector-pq-adc-early-abandon`,
`ruvector-pq-dominant-code-skip`, `ruvector-pq-prefix-pruning`). All of
them score every stored PQ code with Asymmetric Distance Computation
(ADC): one f32 lookup per subspace, summed across `m` subspaces.

The ADC loop is already SIMD-friendly, but every code still passes
through it. For large flat PQ scans (and for IVF postings with long
lists) ADC-per-code is the bottleneck. A cheap integer **prefilter**
that drops obviously-far codes before ADC would multiply existing gains.

PQ codes themselves are integers, but their bit patterns are chosen by
k-means and carry no geometric meaning: two codes with Hamming distance
2 may index centroids far apart in ℝ⁸, and vice versa. Hamming is
therefore useless as a prefilter on vanilla PQ.

**Polysemous codes** (Douze, Jégou, Schmid; ECCV 2016) fix this by
reordering the ksub centroid *labels* within each subquantizer so that
Hamming distance between labels correlates with L2 distance between the
underlying centroids. The reorder is fit once offline by simulated
annealing; retrieval becomes cheap (XOR + popcount) before ADC.

No in-tree crate currently applies this reorder. This ADR proposes
adding a dedicated `ruvector-polysemous` crate that composes with — or
sits beside — existing PQ crates.

## Decision

1. Introduce a new standalone crate
   `crates/ruvector-polysemous-pq-hamming-prefilter` prototyping the
   training, encoding, and two-stage search pipeline in dependency-light
   pure Rust.
2. Fit the centroid permutation per subquantizer with **simulated
   annealing** against the cost
   `Σ (hamming(π(i), π(j)) − α · L2(c_i, c_j))²`, α fit analytically.
3. Store a forward permutation (`perm[s][old] = new`) and encode once
   into remapped codes at ingest time. The ADC LUT is built so that
   `lut[s * ksub + new_code]` holds `L2(q_sub, centroid_old)`; no runtime
   permutation lookups.
4. Expose a two-stage search: Hamming prefilter by threshold `ht` →
   ADC rerank on survivors → top-k heap. Threshold is a tunable knob;
   `ht = max_bits / 2` with `ksub = 256, m = 8` admits ≈ 55 % of codes at
   the current correlation.
5. Keep the crate out of the top-level workspace (`[workspace]` stub
   in its Cargo.toml) so nightly research never touches the main
   members list. Promotion to workspace membership happens with the
   production-layout migration listed below.
6. Treat the crate as additive: no public API of `ruvector-pq-search`
   or siblings changes.

## Consequences

**Positive**

* On 32×-compressed 50 000 × 64-dim synthetic data, the PoC delivers
  **2.03× throughput at 82 % of ADC’s recall@10** at `ht = 28`, and
  **4.15× throughput at ~46 %** at `ht = 24` — a well-behaved recall /
  latency Pareto, matching the shape reported in Douze et al.
* The prefilter is composable: it stacks cleanly with early-abandonment
  ADC (`ruvector-pq-adc-early-abandon`), with FastScan LUT16, and with
  IVF coarse pruning.
* Training is fast: 30 s of SA on an 8 × 256 codebook on M4 Max is a
  one-time offline cost per index.

**Negative**

* Thresholds above `max_bits / 2` admit nearly everything, so the
  popcount pass becomes pure overhead (~0.9× regression at `ht = 32`+
  in this bench). Production use needs either a conservative default
  (`ht ≈ max_bits * 0.44`) or a query-adaptive threshold.
* SA is probabilistic and not reproducible across machines without
  fixed RNG; this is seeded in the PoC, but for distributed training
  we would need a canonical cost function and a tie-breaker.
* The PoC is scalar; a SIMD rewrite (NEON `vcnt`, AVX2
  `_mm256_sad_epu8`) is required for production win margins at IVF-list
  sizes.

**Neutral**

* Behaviour for `ksub = 16` (4-bit codes) is enabled in the PoC but
  less explored; Hamming–L2 correlation ceiling is lower, so expect
  smaller Pareto gains.

## Alternatives considered

* **Keep plain ADC + rely on FastScan** — FastScan already ships elsewhere
  in the ecosystem. It accelerates ADC but does not reduce the number
  of codes scanned. Polysemous reduces N; FastScan reduces cost per code.
  Different axes; we should have both. This ADR adds the former.
* **Binary rerank on top of PQ (dense bit sketch)** — sibling
  `ruvector-rabitq` already provides a scalar-quantised sketch; its
  rerank sketch is logically similar but requires an additional
  separate sketch table. Polysemous is zero-overhead at storage time
  (same code bytes, reordered) and fits the existing PQ storage layout.
* **Skip PQ entirely for scalar quantization (SQ-cascade)** — covered by
  `ruvector-sq-cascade-int4`. SQ has worse compression per recall than
  PQ at these ratios, so not a substitute.
* **Learned-to-prune (predictor over ADC-top-k)** — more powerful but
  requires training data and a neural runtime. Polysemous is fully
  unsupervised and works offline on the codebooks alone.

## Follow-ups

* FastScan-aware SIMD inner loop (NEON / AVX2) in the production crate.
* Query-adaptive threshold tied to top-k ADC boundary.
* OPQ rotation before PQ training — expect Hamming–L2 correlation to
  improve by 10–15 % on natural embeddings.
* Integration behind the existing `PqIndex` trait so IVF postings can
  call a polysemous variant with no caller changes.
