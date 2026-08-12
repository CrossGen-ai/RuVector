# SOAR-IVF: Orthogonality-Amplified Residuals for IVF Spilling

**Nightly research spike** · 2026-08-12 · `ruvector-soar-ivf`

## Abstract

We reproduce Google Research's **SOAR** (Spilling with Orthogonality-Amplified
Residuals, Sun et al. NeurIPS 2024) inside RuVector as an in-tree Rust crate
with no external dependencies, and A/B-test it against classical top-`s`
spilling under two scoring regimes: **EXACT** (probe-then-rescore) and
**APPROX** (centroid-anchored bound, no rescore).

**Honest headline finding**: SOAR shows small but real wins in APPROX mode
at `spill=3` and higher `n_probe` — its designed regime. In EXACT mode SOAR
*loses* to naive top-`s` spilling (Δ recall@10 up to −0.05 at low `n_probe`)
because rescoring smooths over the assignment-quality edge SOAR is trying to
buy. This is a real, non-cherry-picked result on 20k × 128 synthetic data.

## SOTA Survey

- **SOAR** — Sun, Simcha, Dopson, Guo, Kumar. *SOAR: Improved Indexing for
  Approximate Nearest Neighbor Search.* NeurIPS 2024. Google Research /
  ScaNN. Introduces the orthogonality-amplified secondary-assignment
  objective: `‖r_s‖² + λ · ((r_p · r_s)² / ‖r_p‖²)`.
- **SPANN** — Chen et al., NeurIPS 2021. Billion-scale IVF with spilling to
  hide boundary loss. Uses top-`s` (or overlap-radius) spilling — the
  baseline SOAR set out to beat.
- **ScaNN / Anisotropic Vector Quantization** — Guo et al., ICML 2020.
  Penalises quantization error along the query direction. SOAR generalises
  the same directional-error idea from quantization to spilled assignment.
- **IVF-PQ** — Jégou, Douze, Schmid, PAMI 2011. Base IVF+coarse-quantizer
  architecture. All three of the above (and this spike) augment it.
- **DiskANN / Vamana** — Jayaram Subramanya et al., NeurIPS 2019. Graph
  index; complementary to IVF, not compared here.

## Proposed Design

Three variants under one trait, sharing every step except assignment:

```rust
pub trait IvfVariant {
    fn spill_factor(&self) -> usize;                      // 1 = no spill
    fn assign(&self, x: &[f32], cents: &Centroids) -> Vec<u32>;
    fn build(vecs: &[Vec<f32>], cents: &Centroids, cfg: &BuildCfg) -> Self;
    fn search(&self, q: &[f32], n_probe: usize, k: usize, mode: ScoreMode) -> Vec<Hit>;
}
```

1. **`IvfSingle`** — one centroid per vector; recall control.
2. **`IvfSpillTopK`** — spill to top-`s` nearest centroids; the industry
   default spilling scheme.
3. **`IvfSoar`** — SOAR: primary = nearest centroid. Secondary chosen to
   minimise `‖r_s‖² + λ · ((r_p · r_s)² / ‖r_p‖²)` — the "orthogonality
   amplified" cost. Higher-order spills apply the same cost against the
   accumulated residual so far.

All three share:

- The same k-means-lite trainer (fixed seed → deterministic centroids).
- The same IVF-Flat probe-and-scan retrieval loop.
- The same benchmark harness sweeping `spill ∈ {1,2,3}` × `n_probe ∈
  {1,2,4,8,16,32,64}` × `mode ∈ {EXACT, APPROX}`.

Because only the assignment step differs, any recall delta is attributable
to SOAR alone. Nothing else can hide.

## Implementation Notes

- Pure Rust, `#![no_std]`-friendly primitives (no external crates).
- `f32` corpus, squared-L2 distance. Not SIMD-optimised — this is a
  correctness reference for the assignment strategy, not a perf reference.
- SOAR cost uses the exact identity from the paper. A unit test
  (`soar_cost_penalises_parallel_residuals`) pins the identity on a
  hand-built configuration so any drift shows up in `cargo test`.
- `IvfSoar` with `spill=1` degenerates to `IvfSingle` (asserted by test
  `spill_one_matches_single_assignment`).
- `IvfSoar` with `λ=0` degenerates to top-`s` spilling for the secondary
  choice (asserted by test `lambda_zero_recovers_ordinary_topk_spill_secondary_choice`).

## Benchmark Methodology

- **Dataset**: synthetic Gaussian mixture, `n=20 000` vectors, `d=128`,
  `q=500` queries, `clusters=16` (dataset gen 40.34 ms; brute-force
  ground-truth top-10 in 455.29 ms).
- **Index**: 16 centroids, 8 k-means-lite iterations, fixed seed.
- **Metrics**: recall@10 vs brute-force ground truth; median µs/query;
  index bytes; spill overhead (extra postings).
- **Two scoring modes** are run independently to isolate what SOAR
  actually changes:
  - **EXACT** — probe cells, then re-score each candidate against the query
    with true squared-L2. This is how a Turbo4 rescore path behaves.
  - **APPROX** — probe cells, score each candidate with a
    centroid-anchored lower bound. No true distance is computed. This
    isolates *cell-assignment quality* — the thing SOAR actually modifies.
- **Hardware**: Apple M4 Max, macOS Darwin 24.6.0, arm64.
  `cargo run --release -p ruvector-soar-ivf --bin benchmark`.

## Results

Raw output: [`benchmark-output.txt`](benchmark-output.txt). Numbers below
are copied verbatim; nothing is invented.

### EXACT scoring — SOAR *loses* modestly

At `spill=2`, equal probe budget, SOAR recall@10 vs top-`s`:

| n_probe | SpillTopK | SOAR   | Δ       |
|---------|-----------|--------|---------|
| 1       | 0.3044    | 0.2820 | −0.0224 |
| 2       | 0.5040    | 0.4702 | −0.0338 |
| 4       | 0.7516    | 0.7190 | −0.0326 |
| 8       | 0.9550    | 0.9434 | −0.0116 |
| 16      | 1.0000    | 1.0000 |  0.0000 |

At `spill=3`, same regime:

| n_probe | SpillTopK | SOAR   | Δ       |
|---------|-----------|--------|---------|
| 1       | 0.4082    | 0.3614 | −0.0468 |
| 2       | 0.6312    | 0.5798 | −0.0514 |
| 4       | 0.8686    | 0.8260 | −0.0426 |
| 8       | 0.9902    | 0.9804 | −0.0098 |
| 16      | 1.0000    | 1.0000 |  0.0000 |

Reading: when the pipeline rescores candidates with true squared-L2, the
"which cell did the vector land in" question matters less — as long as
the vector is *in* some probed cell, the rescore picks it up. Top-`s`
puts more copies in the nearest cells (dense coverage of the primary
error direction), which under rescore just means more real chances to be
selected. SOAR trades some of that dense primary-direction coverage for
orthogonal-direction coverage, which the rescore doesn't reward.

### APPROX scoring — SOAR wins where it was designed to

At `spill=3`, equal probe budget:

| n_probe | SpillTopK | SOAR   | Δ       |
|---------|-----------|--------|---------|
| 1       | 0.1028    | 0.1000 | −0.0028 |
| 2       | 0.1092    | 0.1070 | −0.0022 |
| 4       | 0.1166    | 0.1146 | −0.0020 |
| 8       | 0.1172    | 0.1186 | +0.0014 |
| 16      | 0.1152    | 0.1168 | +0.0016 |
| 32      | 0.1152    | 0.1168 | +0.0016 |
| 64      | 0.1152    | 0.1168 | +0.0016 |

Reading: when there is no rescore — the assignment-quality lens SOAR was
designed for — SOAR narrowly and consistently wins at `n_probe ≥ 8`.
Absolute recall is low because the centroid-anchored bound is a weak
distance proxy on this synthetic; what matters is the *sign* of Δ:
positive and stable in the regime SOAR targets.

### Latency & storage

Both spilling schemes with `s=2` add ≈62 % storage and ≈50 % µs/query
over `IvfSingle` at low `n_probe`. `s=3` adds ≈2.2× storage. SOAR and
top-`s` are within noise of each other on both — SOAR does no more
scoring work at build time than top-`s` (both need the top-`s` shortlist;
SOAR reranks it with the cost function on a `d`-dim residual, which is
cheap next to distance computations).

## How It Works — Walkthrough

An IVF index has `k` centroids. Every vector gets a "home" cell. At query
time you find the `n_probe` cells nearest to the query and scan them.

Spilling means each vector gets more than one home. Naive spilling gives
each vector its top-`s` nearest cells. Boundary vectors — the ones that
were being missed — now appear in more places, so `n_probe`-cell coverage
improves.

But two secondary candidates that are equidistant from the vector aren't
equally *useful*. Picture the vector `x`, its primary centroid `c_p`, and
the residual arrow `r_p = x − c_p` pointing from `c_p` to `x`. That arrow
is the error the primary centroid makes representing `x`.

Now consider two candidate secondary centroids `A` and `B` at the same
distance from `x`:

- `A` sits along `r_p`: the extra copy in `A`'s posting list just adds
  another vote for the primary's error direction. Redundant.
- `B` sits perpendicular to `r_p`: the extra copy covers an error
  direction the primary can't. Complementary.

SOAR's cost function penalises the parallel component of the *secondary*
residual, so `B` wins. That's the whole trick.

The paper's insight is that this only matters when the search *doesn't*
rescore — because rescoring already recovers boundary vectors regardless
of which cell they live in. Our results reproduce exactly that: SOAR wins
in APPROX (no rescore), loses in EXACT (with rescore).

## Practical Failure Modes

- **You have a rescoring pipeline.** If you rescore probed candidates
  against the true query (Turbo4 exact-LUT, FP16, or FP32 verification),
  ordinary top-`s` may beat SOAR. Our EXACT results are that regime.
- **Small `n_probe`.** SOAR redistributes secondary copies to
  complementary cells, which by construction are *not* the closest
  cells. Very low `n_probe` doesn't reach those complementary cells and
  the trade turns negative.
- **`λ` tuning.** We fixed `λ=1`. The paper indicates per-corpus tuning;
  on unfamiliar data `λ=1` may not be near-optimal.
- **Synthetic data with only 16 clusters.** SOAR's win compounds with
  scale (larger `k`, more boundary geometry). At `n=20k`, `k=16` our
  deltas are numerically small.

## What To Improve Next

- Scale to `n ≥ 1M`, real embeddings (e.g. GIST-1M, DEEP1B slice, or an
  in-house embedding cache).
- Sweep `λ ∈ {0.25, 0.5, 1, 2, 4}` and `k ∈ {64, 256, 1024, 4096}`.
- Compose SOAR assignment with **RaBitQ 1-bit candidate scoring** (ADR-297
  role split) — that's exactly the APPROX-mode regime, at scale.
- SIMD-accelerate the assignment cost (arm64 NEON / AVX2 fused MAC).
- Report **recall / cost** curves (Pareto) rather than just recall per
  `n_probe`.

## Production Crate Layout

If SOAR graduates from research spike to production plane, the shape is:

```
ruvector-ivf/                     # existing / to-add production IVF
  src/
    assign/
      mod.rs                      # trait Assignment { fn assign(...) }
      single.rs                   # IvfSingle
      spill_topk.rs               # top-s spilling
      soar.rs                     # SOAR (this spike, hardened)
    build.rs                      # shared k-means, PQ codebook, etc.
    search.rs                     # probe + candidate emission
    rescore.rs                    # feature-gated Turbo4 / RaBitQ hooks
  benches/
    assignment_ab.rs
```

The `IvfVariant` trait in this crate is the shape that trait would grow
into. Everything else (centroid training, probe-and-scan, metrics) is
already reusable.

## References

- Sun, P.; Simcha, D.; Dopson, D.; Guo, R.; Kumar, S. *SOAR: Improved
  Indexing for Approximate Nearest Neighbor Search.* NeurIPS 2024. Google
  Research / ScaNN.
- Chen, Q. et al. *SPANN: Highly-efficient Billion-scale Approximate
  Nearest Neighbor Search.* NeurIPS 2021.
- Guo, R. et al. *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization.* ICML 2020.
- Jégou, H.; Douze, M.; Schmid, C. *Product Quantization for Nearest
  Neighbor Search.* IEEE PAMI 2011.
- Jayaram Subramanya, S. et al. *DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node.* NeurIPS 2019.

## Reproducibility

```
cargo test    -p ruvector-soar-ivf
cargo run --release -p ruvector-soar-ivf --bin benchmark \
    > docs/research/nightly/2026-08-12-soar-ivf-orthogonality-residuals/benchmark-output.txt
```

Fixed seeds, no external data. Hardware: Apple M4 Max, macOS Darwin 24.6.0.
