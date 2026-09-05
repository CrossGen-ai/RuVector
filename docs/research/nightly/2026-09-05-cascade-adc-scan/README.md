# Cascade Asymmetric Distance Computation (Cascade-ADC)

**Nightly research — 2026-09-05**
**Crate**: `crates/ruvector-cascade-adc`
**ADR**: `docs/adr/0001-cascade-asymmetric-distance-computation-...md`

## Abstract

Product Quantisation (PQ) with 8-bit codebooks is the workhorse
compressed-distance scan in ruvector's flat-PQ and IVF+PQ paths. Every
candidate is scored at a single precision, which is wasteful when only a
small fraction of the corpus is competitive for the final top-K. We
introduce **Cascade-ADC**, a two-stage progressive-precision scanner
built on a *coherent* two-level codebook: a companion 4-bit codebook is
derived from the 8-bit codebook by k-means over its centroids, and the
fine-to-coarse mapping is stored explicitly. Stage-1 sweeps every code
at 4-bit precision using a tiny L1-resident LUT (m × 16 × 4 B). Stage-2
refines only the top ρ·N survivors at 8-bit precision. On a 200 k × 128
Gaussian benchmark with m = 32, cascade at ρ = 0.05 preserves 8-bit
recall (0.4595 vs 0.4625) while running **+13 %** faster than the plain
8-bit scan (464 vs 412 qps, Apple M4 Max, scalar Rust, single-thread).

## SOTA Survey

- **PQ** — Jégou et al. (TPAMI 2011): the ADC LUT trick still baselines
  every compressed-scan pipeline in production.
- **OPQ / AnisoPQ** — Ge et al. (2013); Guo et al. (2020): learned
  rotations improve absolute recall but do not change the per-candidate
  cost story.
- **PQFastScan / SCANN** — André, Kermarrec, Le Scouarnec (2015);
  Guo et al. (2020): SIMD-vectorised 4-bit LUT scans that saturate
  memory bandwidth. Orthogonal to cascade — a natural Stage-1 drop-in.
- **RaBitQ** — Gao et al. (SIGMOD 2024): 1-bit binary codes with a
  provable variance bound. Aggressive compression but the 1-bit and
  8-bit codes are independent codebook families, so 1-bit rankings
  correlate weakly with 8-bit rankings — pruning is unsafe without
  large survivor sets.
- **DiskANN + PQ re-rank** — Jayaram Subramanya et al. (NeurIPS 2019):
  scans PQ on RAM, re-scores top-K with full-precision vectors on SSD.
  Cascade generalises this pattern to *within* the PQ stack itself.
- **CAGRA** — NVIDIA (2024): GPU graph-ANN; irrelevant to a scalar CPU
  cascade but confirms the industry direction of stage-wise precision.
- Competitor changelogs: Milvus 2.4 added `PQFastScan`, Qdrant 1.13
  added multi-tier quantisation, LanceDB shipped PQ-with-refine, and
  FAISS re-scoring is standard. Cascade formalises the "coherent
  companion codebook" idea none of them ship today.

## Proposed Design

### Codebook coherence

Given a standard product quantiser with `m` subspaces of dimension
`dsub` and `K8 = 256` fine centroids per subspace, we compute a coarse
codebook by running a *second* Lloyd pass over the 256 fine centroids
of each subspace to produce `K4 = 16` coarse centroids and a mapping
`parent[j][fine] → coarse` (`m × 256` bytes total).

The coarse centroids are therefore Voronoi-cell-average summaries of
groups of fine centroids: geometrically consistent with the fine
codebook by construction. A candidate whose 4-bit distance is small
tends to have a small 8-bit distance too — that's what makes Stage-1
pruning safe.

### Storage

```
codes8 : n × m bytes
codes4 : n × ⌈m/2⌉ bytes   (two 4-bit codes packed per byte)
parent : m × 256 bytes
```

Bytes per vector: `m + ⌈m/2⌉` (~1.5× vs plain 8-bit PQ).

### Scan

Stage 1 — coarse sweep:
```
lut4 : m × 16 floats           # L1-resident for any m ≤ 1024
for id in 0..N:
    s = Σ_j lut4[j*16 + unpack(codes4[id], j)]
    dists.push((id, s))
select_nth_unstable_by(dists, T)  # partition into top-T (unordered)
```

Stage 2 — fine refine:
```
lut8 : m × 256 floats
for cand in stage1_top_T:
    s = Σ_j lut8[j*256 + codes8[cand][j]]
    topk_heap.push((cand.id, s))
```

Selection uses `select_nth_unstable_by` because a size-T heap has
`O(N log T)` push cost when `T = ρ·N` is large; a flat partition is
`O(N)` and — critically — keeps the hot loop branchless.

### Trait boundary

```rust
pub trait Scanner {
    fn search(&self, index: &PqIndex, query: &[f32],
              k: usize, out: &mut Vec<ScanResult>);
    fn name(&self) -> &'static str;
}
```

Three concrete implementations ship: `FullEightBitScanner`,
`FullFourBitScanner`, `CascadeScanner`. Downstream paths (`IVF+PQ`,
`RaBitQ`, `FRR`) can pick per posting-list at query time.

## Implementation Notes

- No `unsafe`, no SIMD intrinsics. The scan loops are shaped to
  auto-vectorise (`select_nth_unstable_by`, contiguous LUT indexing).
- Zero heap allocation in the inner scan aside from the `dists`
  Stage-1 buffer, which is reusable per query.
- 4-bit unpacking is unrolled two-per-iteration to eliminate the
  parity branch that showed up as a hotspot in the first draft.

## Benchmark Methodology

- Synthetic corpus: `n = 200 000` unit-variance i.i.d. Gaussian
  vectors, `d = 128`.
- Queries: `nq = 200`, each formed as `data[i] + 0.05 · N(0, I)` — the
  standard SIFT/GIST-style "query near a corpus vector" recipe. Yields
  meaningful ground-truth top-K.
- Training: independent 10 k-vector Gaussian sample, seeded.
- Ground truth: exhaustive brute-force top-10 (float L2).
- PQ params: `m = 32`, `dsub = 4`, `K8 = 256`, `K4 = 16`, 12 Lloyd
  iters fine, 8 Lloyd iters coarse.
- Metric: mean `recall@10` and single-thread queries-per-second.
- Warm-up: 4 queries per scanner, excluded from timing.

## Results

Hardware: Apple M4 Max, macOS 15.6, rustc 1.89.0, release profile.

```
n=200000 d=128 m=32 nq=200 k=10 clusters=64
index built in 2.00s
codes8 bytes=6 400 000  codes4 bytes=3 200 000
brute-force ground truth in 1.80s

  full-8bit         qps=411.9   recall@10=0.4625   32 B/vec
  full-4bit         qps=497.3   recall@10=0.1685   16 B/vec
  cascade-rho0.05   qps=464.3   recall@10=0.4595   48 B/vec
  cascade-rho0.10   qps=428.3   recall@10=0.4625   48 B/vec
  cascade-rho0.20   qps=398.5   recall@10=0.4625   48 B/vec
```

Raw run: `raw-runs.txt`.

**Key findings**

- Full 4-bit alone destroys recall (0.169) — cannot be used as a
  drop-in.
- Cascade ρ=0.05 lands within **0.3 pp** of full 8-bit recall while
  being **+13 % faster**.
- Cascade ρ ≥ 0.10 matches 8-bit recall exactly; the throughput
  advantage narrows because Stage-2 does more work.
- Memory cost is 1.5× vs plain 8-bit — acceptable when the alternative
  is either lower recall (4-bit) or worse latency (8-bit).

## How It Works (blog-readable walkthrough)

Imagine sorting a mountain of résumés down to a shortlist. Reading
every résumé end-to-end (the 8-bit scan) is exact but slow. Reading
only the first line of each (the 4-bit scan) is fast but you miss
context and pick the wrong candidates. **Cascade** does both: skim
every résumé's first line, keep the top 5 %, then read those 5 % in
full detail. Because the "first line" was written by summarising each
"full résumé" — not sampled independently — a good full résumé almost
always has a good first line. The two levels agree by construction.

That's the trick: don't invent a separate coarse codec. Coarsen the
one you already have, and remember the mapping.

## Practical Failure Modes

- **Very small `m`** (e.g. m=4 with d=32): the 8-bit LUT (m·256·4 =
  4 kB) already fits in L1; cascade adds overhead with no win.
- **Extremely low ρ with tight recall targets** (< 90 % of 8-bit):
  survivor set may not contain all fine-recall neighbours. Use
  `with_floor(rho, top_t)` to guarantee a minimum survivor count.
- **Highly clustered corpora with tiny within-cluster spread**: PQ
  quantisation noise dominates true within-cluster distance and both
  scanners struggle; cascade cannot fix upstream quantisation error.
- **Adversarial queries** far from the training distribution: coarse
  centroid coverage is poor, Stage-1 pruning drops true neighbours.
  Detectable at query time by monitoring Stage-1 residual variance.

## What to Improve Next (roadmap)

1. **SIMD LUT16 Stage-1** — port André et al.'s PQFastScan kernel as
   the Stage-1 backend (NEON on M-series, AVX2 on x86). 4–8× headroom.
2. **Adaptive ρ** — set `rho` per query from Stage-1 distance
   distribution (wide tail → small ρ; flat distribution → large ρ).
3. **3-level cascade** — 1-bit RaBitQ → 4-bit coarse → 8-bit fine.
   Only feasible if all three levels share a codebook family.
4. **Disk-tiered Stage-2** — for `ruvector-diskann`, fetch full-float
   vectors on-SSD only for the ρ·N survivors. Latency wins dominate.
5. **Learned rerank residual** — instead of 8-bit fine, use a small
   MLP that maps `(query, coarse_code, coarse_residual)` to distance.

## Production Crate Layout Proposal

```
crates/ruvector-cascade-adc/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs         # public API, Layout, bytes_per_vector
│   ├── codebook.rs    # Codebook8, Codebook4, TrainingConfig
│   ├── pq.rs          # PqIndex, PqParams, LUT construction
│   ├── scan.rs        # Scanner trait + 3 impls
│   └── main.rs        # cascade-bench binary
├── tests/
│   └── scanners.rs    # topk shape + recall preservation
└── benches/           # (empty; use `cargo run --release --bin cascade-bench`)
```

Public surface stays minimal: `PqIndex`, `PqParams`, `Scanner`,
`ScanResult`, three scanner structs, `bytes_per_vector`, and the two
codebook types for advanced users.

## References

- Jégou, Douze, Schmid — Product Quantization for Nearest Neighbor
  Search, IEEE TPAMI 2011.
- André, Kermarrec, Le Scouarnec — Cache-Locality PQ Scan
  (PQFastScan), VLDB 2015.
- Ge, He, Ke, Sun — Optimized Product Quantization, CVPR 2013.
- Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar — Accelerating
  Large-Scale Inference with Anisotropic Vector Quantization
  (ScaNN), ICML 2020.
- Gao, Long, et al. — RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound, SIGMOD 2024.
- Jayaram Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi,
  Kannan — DiskANN: Fast Accurate Billion-Point NN Search on a Single
  Node, NeurIPS 2019.
- Prior nightly research: `2026-09-04-fused-rabitq-residual`,
  `2026-06-20-pq-adc-search`, `2026-08-13-entropy-adaptive-ann`.
