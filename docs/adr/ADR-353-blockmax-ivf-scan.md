# ADR-353: Block-Max Scanning for IVF-PQ Posting Lists

**Status:** Rejected (as straight port). Follow-up work tracked.
**Date:** 2026-10-02
**Author:** nightly research
**Related:** [research doc](../research/nightly/2026-10-02-blockmax-ivf-scan/README.md), [ADR-352 PQ Prefix Pruning](ADR-352-pq-prefix-pruning.md)

## Context

The Block-Max WAND (BMW) technique from text IR skips entire blocks of a
posting list when a precomputed per-block upper bound on score cannot
improve the current top-K. We investigated porting it to IVF-PQ: split
each posting list into fixed-size blocks and attach a per-block *lower
bound on ADC distance* so the scanner can skip blocks it cannot beat.

Three bound variants were implemented and benchmarked
(`crates/ruvector-blockmax-ivf`):

- **Heuristic** — precompute per-subspace minimum code value at build time.
- **Tight per-query** — scan the block once per query, take elementwise
  min over the ADC LUT.
- **Tight bitmask** — build-time per-subspace code bitmasks; iterate only
  unique codes per query.

## Decision

**Reject** the straight BMW port at the current parameter regime.

Reasoning, grounded in measurements:

1. The heuristic bound is **unsound.** PQ centroid IDs are not
   distance-ordered, so the elementwise-min-code bound skips 90–98 % of
   blocks and drops recall@10 to 15–50 % of baseline. There is no fix
   without recomputing per query.
2. The tight bounds are **sound** (byte-identical top-K to sequential
   scan, verified in `tests/correctness.rs`) but their per-block
   evaluation cost rivals the full scan they displace.
3. Across four benchmark shapes (random and clustered, nprobe 16–32), the
   tight scanners are 7–11 % **slower** than the sequential IVF-PQ
   baseline despite pruning up to 23.9 % of blocks on realistic clustered
   data.
4. The bitmask optimisation does not pay off at `block_size=128` with
   `ksub=256` because almost every centroid is used in each block.

## Consequences

- `ruvector-blockmax-ivf` ships as a research crate under
  `crates/ruvector-blockmax-ivf` with the full PoC, tests, and
  benchmarks preserved for reproducibility.
- We do NOT promote this to a workspace production crate.
- We do NOT wire block-max into `ruvector-pq-search`.
- **Follow-up work is scheduled** (see roadmap in the research doc):
  * Sub-block bitmask variant (`B` split into 4 sub-blocks).
  * Low-ksub residual codebook variant (`ksub=16`, 16-bit masks).
  * SIMD min-reduce of per-query tight bound (AVX2/NEON).
  * Combination with existing partial-L2 early-abandon
    (`gist-partial-l2-early-abandon-2026.md`).
- The negative result is itself informative: BMW's success in IR rests on
  per-document score ordering within a posting block, an invariant PQ does
  not preserve. Future ANN research into block-level skipping should
  consider reordered or hierarchically-quantized codes instead.

## Alternatives considered

1. **Per-block FastScan (PDX / LUT-16).** Already covered by
   `ruvector-pq-fastscan-lut16` (ADR tracked separately). Orthogonal,
   not a substitute.
2. **Vector-level partial-L2 early-abandon.** Already shipped
   (`ruvector-partial-l2`). Operates at a finer granularity; proven
   positive on these parameter regimes.
3. **Sort codes within a posting list by first-subspace code** so blocks
   have narrow unique-code distributions. Prerequisite for the sub-block
   roadmap item; not implemented in this PoC.

## Measurement summary

| Shape | Scanner | QPS | Recall@10 | Blocks skipped |
|---|---|---|---|---|
| medium-clustered-nprobe32 | sequential | 1415.7 | 0.430 | 0% |
| medium-clustered-nprobe32 | blockmax-heur | 1616.8 | 0.126 | 98.6% |
| medium-clustered-nprobe32 | blockmax-tight | 1272.0 | 0.430 | 23.9% |
| medium-clustered-nprobe32 | blockmax-bitmask | 1231.3 | 0.430 | 23.9% |

Full table in the research doc.
