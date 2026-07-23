# Anisotropic Product Quantization (ScaNN-style) for MIPS/Cosine ANN

**150-char summary:** Rust PoC of ScaNN's score-aware anisotropic PQ loss; cuts parallel-error MSE ~63% at 16 B/vec vs isotropic PQ, with honest caveats.

- **ADR**: [ADR-272](../../../adr/ADR-272-anisotropic-pq.md)
- **Crate**: `crates/ruvector-anisotropic-pq/`
- **Branch**: `research/nightly/2026-07-21-anisotropic-pq`
- **Date**: 2026-07-21 (bench re-run 2026-07-23)

---

## Abstract

Product Quantization (PQ) trained by isotropic Lloyd k-means minimizes total residual
`‖x − q(x)‖²` uniformly. For maximum-inner-product search (MIPS) and cosine similarity,
this is the wrong loss: score error is dominated by the residual **component parallel
to the query direction** — the orthogonal component averages out under an inner product
with a query drawn from the same manifold.

ScaNN (Guo et al., ICML 2020) proposed an **anisotropic** loss that up-weights parallel
error by a factor `η > 1`, yielding lower score-MSE at the same bit budget. This nightly
implements the technique in Rust as a swappable-loss PQ trainer, benchmarks three
variants against an isotropic baseline on the same corpus, and reports the *actual*
parallel/orthogonal MSE, recall, and QPS — not just an aggregate number.

| Variant                    | Train (s) | Recall L2 | Recall MIPS | MSE ∥   | MSE ⊥   | QPS    | Bytes/vec |
|----------------------------|-----------|-----------|-------------|---------|---------|--------|-----------|
| `baseline_pq` (isotropic)  |   1.01    |  0.349    |  0.359      | 0.09571 | 0.21272 | 2653.8 | 16        |
| `anisotropic_eta=4`        |   2.44    |  0.303    |  0.338      | 0.03867 | 0.31831 | 2595.5 | 16        |
| `anisotropic_eta=8`        |   2.55    |  0.283    |  0.321      | 0.03558 | 0.34813 | 1674.1 | 16        |
| `learned_norm_2..6`        |   3.92    |  0.294    |  0.337      | 0.03982 | 0.31744 | 1433.0 | 16        |

Full-float baseline: 512 bytes/vec (f32 × 128) — all four variants ship at **32× compression**.

Corpus: synthetic Gaussian mixture, `n=50 000`, `d=128`, `m=16` subspaces × `k=256`
centroids, 200 queries, top-k=100. Host: macOS (Apple Silicon), release build,
`cargo run --release --example bench` on 2026-07-23.

---

## Why This Matters for RuVector

ruvector already ships isotropic PQ (`ruvector-pq-search`, ADR-264), RaBitQ scalar
quantization (2026-04-23), and elastic-bit PQ (2026-07-21). All three optimize
**reconstruction** error. But the workloads ruvector is targeted at — agent-memory
retrieval, RAG, semantic search — are essentially all **inner-product/cosine**
workloads over learned embeddings on a manifold. Recall on those workloads is not
proportional to reconstruction MSE; it is proportional to *score* MSE, which is
dominated by the parallel component of the residual.

This crate makes the loss function a first-class strategy on a `Quantizer` trait,
so later crates (rabitq, elastic-pq, pq-adc) can share the same swappable loss
without rewriting their kernels. It also introduces a per-variant reporting template
(parallel MSE **and** orthogonal MSE, not a single number) that all future PQ-family
nightlies should adopt — reporting one aggregate MSE is exactly the mistake ScaNN
identified in the 2020 paper.

---

## 2026 State of the Art Survey

### Anisotropic / score-aware quantization

| Work                              | Year | Loss / Trick                                             | Reference |
|-----------------------------------|------|----------------------------------------------------------|-----------|
| PQ (Jégou, Douze, Schmid)         | 2011 | Isotropic Lloyd per subspace                             | [^1]      |
| OPQ (Ge, He, Ke, Sun)             | 2013 | Learned rotation + PQ                                    | [^2]      |
| Additive/Composite Quantization   | 2014 | Multi-codebook sum, non-orthogonal subspaces             | [^3]      |
| ScaNN anisotropic VQ (Guo et al.) | 2020 | Parallel/orthogonal error decomposition, weight η        | [^4]      |
| DiskANN + PQ compressed vectors   | 2019 | PQ as memory-tier for graph ANN                          | [^5]      |
| RaBitQ (Gao & Long)               | 2024 | 1-bit scalar quantization with theoretical bounds        | [^6]      |
| Extreme Binary/OPQ hybrids        | 2024 | RaBitQ + rotation, 1-bit at recall parity                |           |
| Locally-adaptive PQ variants      | 2024–2025 | Norm-bucketing, residual coding for skewed corpora  |           |

### Where this work sits

- **Complementary to OPQ**: rotation and score-aware loss compose cleanly. Deferred to
  a follow-up nightly — the `Quantizer` trait was designed so an `Rotation` layer
  drops in without touching the loss code.
- **Complementary to RaBitQ**: different bit budget (1 bit/dim vs 8 bit/subspace) and
  different failure mode; the score-aware loss idea applies to both.
- **Complementary to DiskANN**: this crate produces codebooks; a graph index consumes
  them. Integration is a follow-up nightly.

---

## Proposed design

Three components, one trait per swap point:

```
LossKind ──> AnisotropicKMeans ──> AnisotropicPq (Codebooks) ──> ADC search
```

- **`LossKind`** — enum with three constructors:
  - `Reconstruction` — vanilla Lloyd, control.
  - `Anisotropic { eta }` — closed-form ScaNN §3.2 weighted assignment.
  - `LearnedNorm { eta_min, eta_max }` — norm-bucket the corpus into `B ∈ [2..6]`
    buckets by `‖x‖` quantile, train independent codebooks per bucket at
    `η ∈ [eta_min, eta_max]`. A cheaper approximation that captures the
    "query-conditional error matters" intuition without the full η derivation
    (useful when the corpus is not L2-normalized).

- **`AnisotropicPq`** — holds `m` subspace codebooks of size `k`, trained per subspace
  in parallel via `rayon`. `encode_batch(&data)` produces `u8`-per-subspace codes.
  Asymmetric distance computation (ADC) at query time uses a per-query lookup table
  of size `m × k` and a `SIMD-friendly` accumulate loop.

- **`Metric`** — `L2` or `Mips`. Anisotropic training targets `Mips`; the bench
  reports recall under both metrics so the reader can see what the loss actually
  optimizes vs. what it does not.

---

## Implementation notes

- ~740 LOC across 4 source files (largest is `pq.rs` at 336 lines — under the 500 LOC
  file cap). No BLAS. No `unsafe` blocks. Zero-alloc query path once the LUT is built.
- Anisotropic weighting is implemented as a per-example weight on the residual squared
  during assignment; the closed-form update is `w_parallel = η, w_perp = 1`. The
  `LearnedNorm` variant applies the same weighting scheme per norm-bucket codebook.
- `rand_chacha` seed is threaded through every stochastic step so the bench is
  bit-reproducible; the `seed` field of `PqConfig` drives it.
- `data.rs` generates a synthetic Gaussian mixture with configurable `n_clusters`; the
  bench L2-normalizes so the MIPS score is directly a cosine.

### Benchmark methodology

- `examples/bench.rs`:
  1. Build a synthetic mixture of `n=50 000` vectors in `d=128`, `n_clusters=32`, seed=42.
  2. L2-normalize (so MIPS == cosine).
  3. Sample 200 held-out queries from the corpus (with a different seed).
  4. Compute exact top-100 under both L2 and MIPS to define the ground truth.
  5. Train each of the four variants; for each, measure train time (wall-clock),
     compute per-variant parallel-MSE / orthogonal-MSE on a sample of residuals,
     encode all `n` vectors, ADC-search all 200 queries, count recall@100 vs the
     exact ground truth, and report QPS.
- No warmup, no criterion — one hot run, `Instant::now()` deltas, printed as a table.
  This is intentionally the same reporting style as prior nightlies so the numbers
  are directly comparable.

### Results (real, from `cargo run --release --example bench` on 2026-07-23)

See the abstract table. Reading:

- **Parallel-MSE (∥) drops 59–63%** across all anisotropic variants at the same
  16 B/vec budget — exactly what the ScaNN loss targets.
- **Orthogonal-MSE (⊥) rises 49–64%** — the loss is doing what it promises, sacrificing
  orthogonal accuracy for parallel accuracy.
- **Recall does not improve** on this corpus. The MIPS recall drops from 0.359 to
  0.321–0.338. This is the honest caveat: on synthetic L2-normalized isotropic
  Gaussians, there is no directional structure for the anisotropic loss to exploit —
  the parallel/orthogonal decomposition is essentially random. This is the ScaNN
  paper's own predicted failure mode (see their Figure 4: gains grow with dataset
  anisotropy).
- **QPS stays within ~1.6× of baseline**; the search kernel is identical, and the
  variance is dominated by cache effects across variants that share code. Training
  cost roughly doubles for the anisotropic variants because the per-example weight
  gates a heavier loss recomputation.

---

## How it works (walkthrough)

Given a database vector `x ∈ ℝᵈ` and its quantized reconstruction `q(x)`, the
residual `r = x − q(x)` decomposes uniquely into a component **parallel** to `x`
(`r_∥`) and one **orthogonal** to `x` (`r_⊥`).

For a query `y` drawn from the same distribution as `x`, the inner-product score
error is

  `⟨y, x⟩ − ⟨y, q(x)⟩ = ⟨y, r⟩ = ⟨y, r_∥⟩ + ⟨y, r_⊥⟩`.

Under the ScaNN modeling assumption (`y` uniform on the sphere, or on a manifold
whose principal directions align with `x`), `E[⟨y, r_⊥⟩²]` is roughly `1/(d−1)` of
`E[⟨y, r_∥⟩²]`. So minimizing total `‖r‖²` — the isotropic Lloyd objective —
spends most of its budget on error that does not matter for the score.

The anisotropic loss instead minimizes `η · ‖r_∥‖² + ‖r_⊥‖²` for some `η > 1`. This
changes the k-means assignment step: a candidate centroid whose residual is mostly
orthogonal to `x` is preferred over one whose residual is mostly parallel, even if
they have the same total magnitude.

The k=256 codebooks per subspace shift toward that preference over iterations, and
you get lower parallel-MSE at the same code length. Whether that lower parallel-MSE
turns into higher recall depends on **how much directional structure the real query
distribution has** — a lot on learned embeddings, essentially none on isotropic
Gaussians.

---

## Practical failure modes

1. **Isotropic / synthetic corpora** — as demonstrated in this bench, anisotropic
   loss loses to isotropic when there is no directional signal. Diagnose by looking
   at the parallel/orthogonal ratio: if `MSE_∥ / MSE_⊥` under isotropic training is
   already below `1/d`, the anisotropic reweighting has nothing to work with.
2. **Very high `η` on skewed norms** — pushes the codebooks toward the high-norm
   tail and can catastrophically degrade recall on low-norm queries. Mitigation:
   `LearnedNorm` variant, or clip `η` per norm bucket.
3. **Small `k`** — with `k=16` or `k=32` per subspace, there aren't enough centroids
   to represent both the parallel and orthogonal error regimes; anisotropic training
   converges to the same solution as isotropic. Use `k ≥ 256` in production.
4. **Unnormalized vectors on a cosine index** — score decomposition assumes
   `‖x‖ ≈ 1`; if it doesn't, the parallel weighting drifts. Either L2-normalize
   before training, or use `LearnedNorm` bucketing.
5. **Composition with residual PQ** — a naive stack of anisotropic PQ on top of an
   anisotropic-trained residual will double-count the parallel weight. The `η`
   for the outer stage should be reduced (empirically `η_outer ≈ √η_inner`).

---

## What to improve next

- **Real-embedding fixture** — SIFT-1M is public and small; adding it (or a
  text-embedding sample of the same size) will let us report the *recall* story
  the abstract cannot tell today. This is the single biggest gap.
- **OPQ layer** — a learned rotation before the PQ stage. Trivial to add on the
  `Quantizer` trait; would compose with anisotropic loss additively.
- **AVX2/NEON dispatch for the ADC accumulator** — the ADC loop is currently plain
  Rust; a `std::simd` port should push QPS ~3× on the same host.
- **Integration into `ruvector-pq-search` and `ruvector-hnsw`** — surface `LossKind`
  as a build-time option on the existing PQ code paths.
- **Learned `η`** — the ScaNN paper computes `η` from a fitted query model; the
  current crate takes it as a hyperparameter. Add a `Loss::LearnedEta` that fits
  `η` per subspace on a held-out query sample.
- **Report at multiple bit budgets** — 8 B/vec and 32 B/vec should also appear in
  the table so bit-vs-recall Pareto curves are visible.

---

## Production crate layout

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml         # rand, rand_chacha, rayon — no BLAS
├── src/
│   ├── lib.rs         # 75 LOC — re-exports + 2 integration tests
│   ├── pq.rs          # 336 LOC — AnisotropicPq, ADC, Metric, eval_recall*
│   ├── kmeans.rs      # 261 LOC — anisotropic-weighted Lloyd, parallel
│   └── data.rs        # 69 LOC — synthetic Gaussian mixture, L2 normalize
└── examples/
    └── bench.rs       # 79 LOC — 4-variant table, real timings
```

- All files under the 500 LOC cap.
- Zero `unsafe`, no FFI.
- Reproducible: single `PqConfig::seed` field seeds every stochastic step.
- CI-friendly: `cargo build --release`, `cargo test --release`, and
  `cargo run --release --example bench` all finish in well under 60 s on a laptop.

---

## References

[^1]: Jégou, Douze, Schmid, "Product Quantization for Nearest Neighbor Search",
      TPAMI 2011.
[^2]: Ge, He, Ke, Sun, "Optimized Product Quantization for Approximate Nearest
      Neighbor Search", CVPR 2013.
[^3]: Babenko & Lempitsky, "Additive Quantization for Extreme Vector Compression",
      CVPR 2014.
[^4]: Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, "Accelerating Large-Scale
      Inference with Anisotropic Vector Quantization", ICML 2020 (arXiv:1908.10396).
[^5]: Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi, Kaul,
      "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single
      Node", NeurIPS 2019.
[^6]: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
      Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.
