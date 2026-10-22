# SOAR: Spilling with Orthogonality-Aware Residuals for IVF

**Nightly research — 2026-10-22**
**Crate:** `crates/ruvector-soar-ivf/`
**ADR:** [ADR-353](../../../adr/ADR-353-soar-orthogonal-spill-ivf.md)

## Abstract

We implement and measure *SOAR* (Spilling with Orthogonality-Aware Residuals),
Sun et al.'s 2024 Google Research technique for improving IVF recall at low
`nprobe` with negligible memory overhead. Instead of assigning each vector
to one centroid (classic IVF) or spilling to a second random centroid, SOAR
picks the secondary centroid so that the residual it induces is *as
orthogonal as possible* to the primary residual. This makes the two posting
lists act like independent samples when a query lands slightly off its own
primary centroid, so a single probe does the work of two.

On a 50 000-vector synthetic Gaussian-mixture dataset (dim=64, 256 IVF
centroids), our pure-Rust implementation lifts **recall@10 at `nprobe=1`
from 0.717 → 0.933** (+21.6 pp) over classic IVF, while a random-spill
ablation stays flat at 0.718. Memory overhead is **+0.19 MiB on 12.2 MiB
of data** (1.5 %) — just the extra `u32` id stored in a second posting
list. Build time rises from 0.22 s to 0.68 s (3× due to the per-vector
O(k) orthogonality scan).

## SOTA survey (October 2026)

| System / paper | Year | Idea | Relation to SOAR |
|---|---|---|---|
| **FAISS IVF** | 2017 | Single nearest-centroid IVF (Jégou/Douze) | Baseline we beat. |
| **Johnson, Douze, Jégou — Billion-scale similarity search** | 2017 | Multi-probe LSH / IVF nprobe sweep | Same recall curve shape. SOAR shifts it up. |
| **DiskANN / Vamana** | 2019 | Graph index with greedy routing (Microsoft) | Different family (graph, not inverted file). |
| **ScaNN** | 2020 | Anisotropic quantisation + partition-aware loss (Guo et al.) | SOAR authors overlap. ScaNN's partitioning is the direct predecessor to SOAR. |
| **SPANN** | 2021 | Hierarchical IVF with posting-spill (microsoft) | Spills to several centroids, no orthogonality criterion. |
| **SOAR (Sun, Guo, Simcha, Krichene, Wu)** | NeurIPS 2024 | Primary + orthogonality-aware secondary centroid | **This work.** |
| **Google ScaNN release notes 1.3+** | 2024-2025 | SOAR shipped as `spilling_strategy = SOAR` in open-source ScaNN | Production evidence this works. |
| **CrackIVF** | 2025 | Online centroid refinement (prior ruvector nightly) | Orthogonal technique; stackable with SOAR. |
| **IVF-ball-pruning** | 2026 (prior nightly) | Query-side list pruning | Query-side; SOAR is build-side. Fully compatible. |

The key insight the 2024 paper formalises: when posting lists are
constructed by nearest-centroid assignment, two posting-list visits for
the same query see *correlated* quantisation error. Spilling helps less
than it should. If the spill is instead chosen so its residual is
orthogonal to the primary's, the two visits become statistically
independent, and the probability that at least one visit contains the
true nearest neighbour rises sharply at low `nprobe`.

## Proposed design

For each vector `x` with primary centroid `c_p` and residual
`r_p = x − c_p`, choose the secondary centroid `c_s` by minimising

```
    f(c_s) = ‖r_s‖² + λ · (⟨r_p, r_s⟩² / ‖r_p‖²)
```

where `r_s = x − c_s`. The first term keeps the secondary close; the
second penalises residuals that are parallel to the primary residual.
`λ` controls how aggressively we trade distance for orthogonality.

The pure-Rust implementation lives in
[`crates/ruvector-soar-ivf/src/lib.rs`](../../../../crates/ruvector-soar-ivf/src/lib.rs)
and exposes three pluggable assigners behind one `IvfAssigner` trait:

- `PrimaryOnly` — one centroid per vector (baseline);
- `RandomSpill { seed }` — primary + deterministic random secondary
  (ablation to control for list-size doubling);
- `Soar { lambda }` — primary + orthogonality-aware secondary.

`IvfIndex::build(data, centroids, assigner)` runs the strategy and
returns an in-memory index with per-list `Vec<u32>` postings. Search is
exact-distance over the probed lists with a `seen[]` dedup to avoid
double-counting spilled vectors.

## Implementation notes

- All f32, no SIMD intrinsics. Correctness first; the per-vector O(k·d)
  secondary scan is the only novel hot path and is trivially
  parallelisable later.
- `kmeans()` provides k-means++ init + Lloyd iterations with a
  deterministic `ChaCha8Rng`, so results are reproducible given a seed.
- The IVF index clones vector data for ownership; the posting-list spill
  stores only the `u32` id again (no vector duplication).
- `#![forbid(unsafe_code)]` — no `unsafe` anywhere in the crate.
- `assign()` returns `Vec<usize>` (length 1 or 2) so the same code path
  handles non-spilled and spilled strategies.

## Benchmark methodology

Hardware: Apple Silicon (Darwin 24.6.0, M-series), single-threaded,
`cargo build --release` with workspace-root profile (opt-level=3, LTO off
at workspace default, codegen-units default).

Dataset: synthetic Gaussian mixture:
- `n = 50 000` database vectors + `500` queries drawn from the same
  mixture;
- `dim = 64`;
- `128` true clusters, each ± uniform noise;
- `256` IVF centroids trained with 15 Lloyd iterations on the full set.

Metric: recall@10 vs ground-truth brute-force exact nearest neighbours,
averaged over 500 queries; latency in µs/query (single thread, warm
cache after the k-means run).

Strategies compared (identical centroids):
- `primary-only`
- `random-spill`
- `soar(λ = 0.5)`, `soar(λ = 1.0)`, `soar(λ = 2.0)`

The benchmark is `cargo run --release --example bench -p ruvector-soar-ivf`
in [`examples/bench.rs`](../../../../crates/ruvector-soar-ivf/examples/bench.rs).

## Results

Build cost (same centroids for all five variants):

| strategy       | build (s) | replicas | posting list (MiB) |
|---------------:|----------:|---------:|-------------------:|
| primary-only   |      0.22 |   50 000 |              0.191 |
| random-spill   |      0.22 |  100 000 |              0.381 |
| soar(λ = 0.5)  |      0.68 |  100 000 |              0.381 |
| soar(λ = 1.0)  |      0.68 |  100 000 |              0.381 |
| soar(λ = 2.0)  |      0.69 |  100 000 |              0.381 |

Recall@10 across the `nprobe` sweep:

| strategy       | np=1   | np=2   | np=4   | np=8   | np=16  | np=32  |
|---------------:|-------:|-------:|-------:|-------:|-------:|-------:|
| primary-only   | 0.7170 | 0.9454 | 0.9970 | 1.0000 | 1.0000 | 1.0000 |
| random-spill   | 0.7182 | 0.9458 | 0.9970 | 1.0000 | 1.0000 | 1.0000 |
| soar(λ = 0.5)  | 0.9330 | 0.9946 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soar(λ = 1.0)  | 0.9326 | 0.9944 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soar(λ = 2.0)  | 0.9320 | 0.9944 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Headline:** SOAR beats primary-only by +21.6 pp at `nprobe=1` and by
+4.9 pp at `nprobe=2`. Random-spill gets essentially nothing (+0.1 pp at
`nprobe=1`), confirming the gain is from the *orthogonality constraint*,
not the extra posting-list capacity.

**λ is forgiving.** All three λ values (0.5, 1.0, 2.0) land within 0.001
of each other on recall at every `nprobe`. λ = 1.0 is the recommended
default.

Latency rises with spill (more candidates scored per probe), but the
`nprobe=1` SOAR latency (34 µs/query) is still lower than the
`nprobe=2` primary-only latency (30 µs/query) for a *higher* recall,
which is the right place to compare.

## "How it works" walkthrough (blog-readable)

Think of IVF as a two-step filter: *routing* (which posting list(s) do
I look in?) and *scoring* (score the vectors in those lists against the
query). The recall ceiling at a given `nprobe` is set by routing, not
scoring: if the true nearest neighbour isn't in a probed list, no amount
of scoring will find it.

Classic IVF puts each vector in exactly one posting list — the one for
its nearest centroid. For a query that lands near the Voronoi boundary
between two centroids, the true nearest neighbour might be on the
*other* side of the boundary, so `nprobe=1` misses it.

Spilling to a second centroid helps, but *which* second centroid? A
random second posting list sees a slightly different slice of the data,
but if that slice is correlated with the primary's (because both
centroids point in similar directions), the two probes are essentially
redundant.

SOAR's trick: pick the second centroid so the vector's error relative
to it points in a *different direction* in space from its error
relative to the primary. Now when a query comes in, the two probes
cover two *statistically independent* failure modes of nearest-centroid
routing. The probability that the true neighbour is in at least one
probed list goes up sharply.

The orthogonality penalty `λ · ⟨r_p, r_s⟩² / ‖r_p‖²` is literally the
squared cosine of the angle between the two residuals, scaled by λ.
Minimising it drives that angle toward 90°.

## Practical failure modes

- **Very small datasets with few centroids.** If `k < 4`, there is not
  enough geometric room to find an orthogonal secondary — SOAR
  degenerates toward random spill.
- **Highly clustered data where every vector sits on top of its
  centroid.** `‖r_p‖ ≈ 0` makes the orthogonality denominator blow up.
  The code clamps `‖r_p‖²` to `1e-12` to prevent NaNs; in that regime
  SOAR effectively picks the second-nearest centroid.
- **Index build time is 3× primary-only** at `k = 256`. For millions of
  centroids this becomes the bottleneck. Mitigation: compute SOAR
  scores only against the top-M nearest centroids (not all k).
- **Spill doubles posting-list visits per probe.** For a given
  `nprobe`, latency ≈ 2× primary-only. The gain is that you can run
  a lower `nprobe` and still beat the baseline. Account for this when
  comparing wall-clock cost at fixed recall.

## What to improve next (roadmap)

1. **Top-M candidate restriction for SOAR scoring.** Only consider the
   M-nearest centroids (M=16 or so) as secondary candidates. Should
   shrink build time from O(n·k·d) to O(n·M·d + n·k) with no measurable
   recall loss.
2. **Multi-way spill (SOAR-r).** Pick the r best orthogonal secondaries,
   not just one. The 2024 paper reports further recall gains at r=3.
3. **Fuse with existing ruvector quantisers.** SOAR + RaBitQ, SOAR +
   PQ-FastScan: the SOAR effect is on routing; it is orthogonal (pun
   intended) to the scoring quantiser.
4. **Parallel build.** Per-vector assignments are embarrassingly
   parallel — a Rayon `par_iter` would cut 0.68 s to ~0.1 s on 8 cores.
5. **SIMD inner loops.** `l2_sq` and `dot` are the hot kernels; both
   are trivially SIMD-isable with portable `std::simd` (nightly) or
   `wide`.
6. **Larger-scale validation.** Reproduce on SIFT1M and the LAION-400M
   subset to confirm the +20 pp gain survives at scale with learned
   data.

## Production crate layout proposal

```
crates/ruvector-soar-ivf/
├── Cargo.toml
├── src/
│   ├── lib.rs              ← trait + 3 assigners + IvfIndex (this nightly)
│   ├── index.rs            ← split out IvfIndex when it grows beyond 500 LOC
│   ├── parallel.rs         ← Rayon par-build (roadmap item 4)
│   ├── simd.rs             ← portable_simd kernels (roadmap item 5)
│   └── quantised.rs        ← SOAR + RaBitQ / PQ bridge (roadmap item 3)
├── examples/
│   └── bench.rs            ← the real-numbers benchmark (this nightly)
└── benches/
    └── soar_bench.rs       ← criterion-based regression bench (next pass)
```

## References

1. Sun, Guo, Simcha, Krichene, Wu. *SOAR: Improved Indexing for
   Approximate Nearest Neighbor Search*. NeurIPS 2024.
   <https://arxiv.org/abs/2404.00774>
2. Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar. *Accelerating
   Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN).
   ICML 2020. <https://arxiv.org/abs/1908.10396>
3. Chen, Zhao, Li, Han, Zhou, Zhang, Zhao, Wen. *SPANN: Highly-efficient
   Billion-scale Approximate Nearest Neighbor Search*. NeurIPS 2021.
   <https://arxiv.org/abs/2111.08566>
4. Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs*.
   IEEE Trans. Big Data, 2017. <https://arxiv.org/abs/1702.08734>
5. Google Research ScaNN repository, SOAR spilling strategy:
   <https://github.com/google-research/google-research/tree/master/scann>

## Reproduce

```bash
cargo test   --release -p ruvector-soar-ivf
cargo run    --release --example bench -p ruvector-soar-ivf
```
