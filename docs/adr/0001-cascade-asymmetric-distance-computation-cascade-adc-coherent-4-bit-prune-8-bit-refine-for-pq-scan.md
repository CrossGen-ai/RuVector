<!-- One decision, stated in the filename (ADR-0021). -->

# ADR-0001: Cascade Asymmetric Distance Computation (Cascade-ADC): coherent 4-bit prune + 8-bit refine for PQ scan

> Decision date: 2026-09-05
> Status: Proposed
> Scope: crates/ruvector-cascade-adc, downstream PQ/IVF scan paths (ruvector-pq-search, ruvector-rabitq, ruvector-fused-rabitq-residual)
> Drivers: nightly research agent

## Context

Product-quantised (PQ) asymmetric-distance-computation (ADC) scans
dominate the cost of flat-PQ and IVF+PQ retrieval paths in ruvector.
Today every candidate is scored at a single precision — 8-bit PQ — so a
full sub-quantiser LUT lookup runs for every subspace of every vector in
the candidate list. This is wasteful when only a small fraction of the
corpus is competitive for the top-K.

Prior nightly work explored coarser codings (RaBitQ 1-bit,
fused-residual 4-bit tail) but always as *replacements* for the primary
codec, not as *pre-filters* sharing a codebook family with the primary
codec. Without a coherent coarsening, pruning at low precision can drop
true neighbours because the coarse and fine geometries disagree.

The forces:
- Memory is cheap on M-series and modern x86, but LUT bandwidth is
  precious inside the scan hot loop.
- Recall degradation is a hard failure — anything below the 8-bit
  baseline breaks downstream RAG quality gates.
- The pipeline must accept new backends (SIMD, GPU, disk-tiered)
  without a rewrite of the scan API.

## Decision

1. Introduce crate `ruvector-cascade-adc` with:
   - A coherent two-level PQ codebook: standard 8-bit codebook (K8 =
     256) plus a companion 4-bit codebook (K4 = 16) derived by a
     second k-means pass over the fine centroids. A `parent[j][fine]`
     table maps every fine code to its coarse partner so both codes
     describe the same physical vector region.
   - A `Scanner` trait with three implementations:
     `FullEightBitScanner`, `FullFourBitScanner`, and
     `CascadeScanner { rho }` (4-bit sweep over all N, 8-bit
     refinement over the top ρ·N survivors).
   - Stage-1 selection uses `select_nth_unstable_by` (O(N) partial
     partition), not a size-T heap — heap maintenance dominates when
     T = ρ·N is large.
   - A runnable benchmark binary producing real qps + recall numbers.
2. Downstream scan paths adopt the `Scanner` trait so backends can be
   swapped per posting-list at query time.

## Alternatives Considered

- RaBitQ 1-bit pre-filter (already in-tree): stronger compression, but
  the 1-bit and 8-bit codes do not share a codebook family, so
  Stage-1 ranks correlate less with Stage-2 ranks. Cascade's coherent
  coarsening avoids that mismatch.
- PQ4-only (full-4bit): halves memory but drops measured recall
  catastrophically (46.25 % → 16.85 % on the benchmark). Rejected.
- PQFastScan-style SIMD LUT16 (André, Kermarrec, Le Scouarnec, 2015):
  hand-tuned SIMD 4-bit scan. Orthogonal to cascade — can replace the
  scalar Stage-1 loop in a follow-up.
- AnisoPQ / OPQ rotation: improves absolute recall but does not
  address per-candidate cost, which is what cascade targets.

## Consequences

Positive:
- Recall preservation: cascade recall matches the 8-bit ceiling across
  ρ ∈ {0.05, 0.10, 0.20} on the Gaussian benchmark (46.25 % vs
  46.25 %; ρ=0.05 within 0.3 pp).
- Competitive throughput: cascade ρ=0.05 measured 464 qps vs 412 qps
  for full 8-bit (Apple M4 Max, scalar Rust, n=200k, d=128, m=32).
- Composable: `Scanner` trait lets IVF, RaBitQ, and FRR paths pick the
  scanner per posting-list at query time.
- Recall floor guarantee: `CascadeScanner::with_floor(rho, top_t)`
  never falls below `top_t` survivors even for small ρ.

Negative:
- Memory overhead: cascade holds both code layouts — `m + ⌈m/2⌉`
  bytes per vector, ~1.5× vs plain 8-bit PQ.
- Extra training cost: coarse k-means adds one Lloyd pass per
  subspace (< 5 % of total training time on the benchmark).
- No speed-up when the 8-bit LUT fits comfortably in L1 (small `m`);
  cascade shines when `m ≥ 32` and Stage 2 does non-trivial
  per-candidate work (residual re-rank, full-float fetch, disk seek).

## Testable Criteria

| ID | Criterion | How verified |
|----|-----------|--------------|
| TC-1 | `CascadeScanner` returns exactly `k` results with distances in ascending order for a small deterministic index. | `cargo test --release -p ruvector-cascade-adc` — test `scanners_return_topk` |
| TC-2 | Cascade recall ≥ `FullFourBitScanner` recall and within 5 pp of `FullEightBitScanner` recall on the deterministic mixture. | `cargo test --release -p ruvector-cascade-adc` — test `cascade_recall_beats_or_matches_4bit` |
| TC-3 | Benchmark binary `cascade-bench` runs and prints qps / recall for full-8bit, full-4bit, and cascade ρ∈{0.05,0.10,0.20} without panic. | `cargo run --release -p ruvector-cascade-adc --bin cascade-bench` |
| TC-4 | Cascade ρ=0.05 recall stays within 0.5 pp of full 8-bit recall on the benchmark distribution. | Recorded in `docs/research/nightly/2026-09-05-cascade-adc-scan/raw-runs.txt` (0.4595 vs 0.4625). |

## References

- crate: `crates/ruvector-cascade-adc/`
- research: `docs/research/nightly/2026-09-05-cascade-adc-scan/README.md`
- prior nightly: `docs/research/nightly/2026-09-04-fused-rabitq-residual/`
- André, F.; Kermarrec, A.-M.; Le Scouarnec, N. — *Cache-Locality PQ Scan (PQFastScan)*, 2015.
- Jégou, H.; Douze, M.; Schmid, C. — *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011.
