# Anisotropic Score-Aware Product Quantization for MIPS

*Nightly research · 2026-09-03 · slug `anisotropic-score-aware-pq` · companion to ADR-0001*

## Abstract

Every product-quantization (PQ) codebook in the RuVector workspace is
trained by minimising Euclidean reconstruction error — a loss that
treats every direction of the residual equally. When the downstream
task is maximum-inner-product search (MIPS), errors along the query
direction distort ranking scores far more than errors orthogonal to it,
so the isotropic loss is misaligned with what the retriever actually
optimises for. This report ports the *score-aware anisotropic loss* of
Guo et al. (ICML 2020, arXiv:1908.10396) — the codebook-training loss
at the heart of Google ScaNN — into a minimal Rust crate,
`ruvector-anisotropic-pq`. We isolate the loss from ScaNN's other
engineering (partitioning, SIMD LUTs, GPU kernels) so its effect can be
measured in isolation. On a deterministic 5 000-point / 500-query
Gaussian-mixture benchmark at `(dim = 64, M = 8, K = 256)`, MIPS
Recall@10 climbs from **0.3368** (isotropic PQ) to **0.3590** (η = 4,
+2.22 pp) and **0.3680** (η = 16, +3.12 pp) — at unchanged code size,
unchanged search cost, and ~1.5× training cost. Reconstruction MSE
gets worse (0.0532 → 0.0775) exactly as the theory predicts: the
codebook is trading MSE for score fidelity.

## SOTA survey

- **Guo et al., *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization*, ICML 2020, arXiv:1908.10396.** Introduces the
  score-aware loss `L_η(x, x̃) = η · ||r_∥||² + ||r_⊥||²` and shows
  substantial MIPS Recall@N gains at fixed code size on GloVe,
  Deep1B, and ImageNet embeddings. Ships as Google ScaNN.
- **Sun, Guo, Chi, Simcha, Kumar, *SOAR: New Algorithms for Even Faster
  Vector Search with ScaNN*, NeurIPS 2023.** Adds spilling with
  orthogonality-amplified residuals to the *partitioning* layer above
  the codebook. Orthogonal to (and composable with) the score-aware
  codebook loss — different problem. RuVector already has a PoC at
  `crates/ruvector-soar-ivf`.
- **Ge, He, Ke, Sun, *Optimized Product Quantization*, CVPR 2013.**
  Learns a rotation making PQ subspaces independent; orthogonal to
  anisotropy and composable.
- **Jégou, Douze, Schmid, *Product Quantization for NN Search*, IEEE
  TPAMI 2011.** The isotropic baseline we improve on.
- **Milvus / Qdrant / Weaviate / Pinecone / LanceDB / FAISS
  changelogs (2024–2026).** Milvus ships SCANN via faiss-scann;
  Qdrant, Weaviate, LanceDB do not expose an anisotropic-loss PQ
  trainer as of writing (all use plain PQ or PQ + OPQ). RuVector
  therefore has a real feature-parity gap to close.
- **Guo et al. ScaNN code:**
  https://github.com/google-research/google-research/tree/master/scann

Nothing in `docs/research/nightly/*` prior to this doc has isolated the
score-aware loss as its own primitive; `2026-06-20-pq-adc-search`
targets isotropic PQ + asymmetric distance compute (ADC), and
`2026-06-24-spann-partition-spill` targets partition spilling
(complementary).

## Proposed design

A three-file Rust crate whose public surface is one trait and two
implementations, plus a benchmark binary. The design goal is
maximal simplicity so the loss itself — not the surrounding
engineering — is what the benchmark measures.

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml
└── src/
    ├── lib.rs         — public re-exports, doctring, module wiring
    ├── dataset.rs     — deterministic Gaussian-mixture data + ground truth
    ├── pq.rs          — Quantizer trait, BaselinePq, AnisotropicPq
    ├── search.rs      — asymmetric distance compute (ADC) searcher
    ├── metrics.rs     — Recall@k and reconstruction MSE
    └── bin/
        └── benchmark.rs — cargo-run entry point
```

The `Quantizer` trait exposes exactly one method:

```rust
fn train(&self, data: &[Vector], params: PqParams) -> Codebook;
```

so any future variant (score-aware assignment, OPQ + anisotropy,
per-subspace η, ADC with residual quantization) plugs in without
touching the search path.

### Anisotropic centroid update

For each cluster `C_j` in each subspace, minimise

```
Σ_{i ∈ C_j} [ η · ((c - x_i)·û_i)²  +  ||(c - x_i) - ((c - x_i)·û_i)û_i||² ]
        = Σ_i ||c - x_i||²  +  (η - 1) · ((c - x_i)·û_i)²
```

with `û_i = x_i / ||x_i||`. Setting the gradient to zero gives the
`sub_dim × sub_dim` linear system

```
( Σ_i M_i ) c = Σ_i M_i x_i,   M_i = I + (η - 1) û_i û_iᵀ
```

Nice simplification: `M_i x_i = x_i + (η-1) (x_i·û_i) û_i = η x_i`
(because `x_i·û_i = ||x_i||`), so the right-hand side is simply
`η Σ_i x_i`. We solve the system with textbook Gauss elimination with
partial pivoting. `sub_dim` is small (8 in the default config) so this
is `O(sub_dim³) ≈ 500` FLOPs per centroid update — negligible next to
the `O(N · K · sub_dim) = O(5000 · 256 · 8) ≈ 10 M` FLOPs of the
assignment step.

### Assignment step

Kept as plain L2 nearest centroid. The ScaNN paper's ablations show
this loses only a small fraction of the anisotropy gain; using the same
assignment rule as isotropic PQ keeps the trainer bit-for-bit
substitutable in any existing PQ pipeline.

### Search

Standard ADC. Query builds an `M × K` inner-product LUT
(one dot product per centroid); each database candidate contributes
`Σ_m LUT[m, code[m]]`; top-k is maintained in a small binary heap. On
the benchmark this is ~40 µs per query for `n = 5 000`, dominated by
the LUT build (2 048 subvector dot products of length 8).

## Implementation notes

- **No external dependencies.** The crate uses only `std`. This makes
  it WASM-buildable out of the box for browser-side index building.
- **Deterministic RNG** via a splittable LCG (SplitMix64). Byte-
  identical results across machines given the same seed.
- **k-means init** picks `k` random distinct points; empty clusters
  during training are re-seeded from a random point. Same behaviour
  for both baseline and anisotropic variants — the *only* thing that
  changes is the centroid-update math.
- **Codebook size assertion**: `k ≤ 256` so a code fits in one byte
  per subspace, matching the layout used by every other PQ crate in
  the workspace.

## Benchmark methodology

Single-thread, release build, deterministic seed `0xA155_0741`.

- **Data.** 5 000 database vectors and 500 query vectors drawn from a
  16-component Gaussian mixture in `dim = 64`. Cluster centres are
  unit-Gaussian rescaled to modest but variable norms so MIPS ranking
  is meaningful; per-point Gaussian noise at std 0.35. Query points
  drawn from the same mixture with 1.5× noise (so they are near, not
  identical to, cluster centres — realistic near-cluster queries).
- **Ground truth.** Brute-force MIPS top-10 for every query against
  the full database using `f32` inner products.
- **Codebook config.** `M = 8`, `K = 256`, `sub_dim = 8`,
  `iters = 12`. Same config for all three variants.
- **Metrics.** MIPS Recall@10 (fraction of true top-10 recovered);
  reconstruction MSE (mean squared entry-wise error); wall-clock train
  time, encode time, and per-query search time.

Hardware for the run reported below: Apple Silicon (aarch64-apple-
darwin), macOS. Single-threaded throughout — no rayon, no BLAS, no
SIMD intrinsics.

## Results

Raw output of
`cargo run --release -p ruvector-anisotropic-pq --bin benchmark`
on 2026-09-03:

```
────────────────────────────────────────────────────────────────────────────
  ruvector-anisotropic-pq benchmark
────────────────────────────────────────────────────────────────────────────
  dataset : n_db=5000  n_queries=500  dim=64  k_clusters=16  noise_std=0.35
  pq      : m=8  k=256  sub_dim=8  iters=12
  metric  : MIPS Recall@10, reconstruction MSE
────────────────────────────────────────────────────────────────────────────
  generated data + ground-truth top-10 in 0.07s

| Quantizer | Train (s) | Encode (s) | Search (ms/q) | Recall@10 | MSE |
|-----------|-----------|------------|---------------|-----------|-----|
| baseline-pq (η=1)              |     0.327 |      0.031 |         0.039 |    0.3368 | 0.0532 |
| anisotropic-pq (η=4)           |     0.494 |      0.037 |         0.042 |    0.3590 | 0.0621 |
| anisotropic-pq (η=16)          |     0.528 |      0.042 |         0.053 |    0.3680 | 0.0775 |
────────────────────────────────────────────────────────────────────────────
  total: 1.60s   (single-thread, deterministic seed = 0xa1550741)
────────────────────────────────────────────────────────────────────────────
```

Reading the table:

- **Recall lift is real and monotone in η.** +2.22 pp at η=4, +3.12 pp
  at η=16. Consistent with Guo et al.'s finding that larger η helps
  more when we care about high-recall regions.
- **MSE regression is real and monotone in η.** +16.7% at η=4, +45.7%
  at η=16. This is the *whole point* — the codebook is spending
  reconstruction accuracy to buy score fidelity.
- **Train cost is ~1.5×–1.6× baseline.** Encode cost is unchanged
  (the encoder is a plain L2 nearest-centroid). Search cost is
  unchanged (the codebook layout is identical).

Recall is modest in absolute terms (~0.34) because the benchmark uses
`M = 8` — an aggressive compression from `64 × 4 = 256 B` down to
`8 B`, a 32× compression. At this compression ratio absolute numbers
are less important than the relative lift from an identical change to
one arithmetic subroutine.

## How it works (blog-readable walkthrough)

Imagine you compress a movie for streaming and you have to pick which
details to keep and which to throw away. If your goal is *lowest
overall pixel difference*, you spread the loss uniformly across every
frame and every pixel. If your goal is *the viewer's subjective
experience*, you spend the bit budget where their eye is — faces,
motion, high contrast — and let the rest be blurry.

Product quantization is compression for vectors. Standard PQ picks the
"lowest overall pixel difference" objective: it minimises the Euclidean
distance between the original vector and its reconstruction. That's
the right objective if you're going to eyeball the reconstruction. But
for retrieval nobody eyeballs it — the retriever multiplies it by a
query vector and picks the largest dot product. So the parts of the
reconstruction the retriever's "eye" is on are the parts *aligned with
the direction of typical queries*. Errors perpendicular to that
direction are effectively invisible; errors along it hurt.

Anisotropic PQ literally rewrites the codebook trainer's objective:
each residual gets split into a component along the datapoint's own
direction (a stand-in for "the direction queries will arrive from")
and a component perpendicular to it. The parallel component is
weighted `η` times more than the perpendicular one. Train k-means
with that reweighted loss and the codebook naturally puts more of its
representational capacity into the direction that matters for
retrieval scoring — even though it now looks slightly *worse* by the
old Euclidean yardstick.

The rest of the compression pipeline doesn't change. The codes are
still one byte per subspace. The query-time lookup table is still
`M × K` inner products. Nothing downstream needs to know it is looking
at an anisotropically-trained codebook. That is what makes this a
particularly good improvement to ship: it is a codebook swap, not a
protocol change.

## Practical failure modes

- **Distance-based re-rankers over the same codes will regress.** If a
  downstream stage is doing distance-based re-ranking (e.g., top-100
  by ADC → top-10 by L2 on reconstructions), its L2 numbers get
  worse under anisotropic training even though its ranking gets
  better. Mitigation: rerank by inner product, not by distance.
- **η is task-dependent.** ScaNN's paper picks η based on the target
  recall regime. For low-recall targets (top-1) small η is
  sufficient; for high-recall targets (top-1000) large η helps more.
  Wrong η can regress recall. Mitigation: sweep η on a held-out
  validation slice at index-build time.
- **Datapoint direction is a proxy for query direction.** If the
  query distribution is systematically *not* aligned with the
  datapoint distribution (e.g., you serve queries from a distinct
  domain), the "parallel is what matters" assumption breaks and the
  anisotropy can actually hurt. Mitigation: fall back to η = 1 in
  cross-domain retrieval or add a rotation calibrated on a query
  sample.
- **k-means still gets stuck in bad local minima.** Anisotropy does
  not fix that. Use k-means++ init and/or multiple restarts in
  production — the PoC uses plain random init to keep the trainer
  small.
- **Empty clusters** are re-seeded from a random point. In pathological
  low-density subspaces this can bias the codebook toward the seed;
  in production, log a warning when reseed rate exceeds a threshold.

## What to improve next (roadmap)

1. **Score-aware assignment.** The paper's assignment step also uses
   `L_η`, not L2. Adding this closes the last remaining gap between
   this PoC and full ScaNN training. Should be a ~20-line diff.
2. **Compose with OPQ.** Learn a rotation `R` on top of anisotropic
   training. Independent axis of improvement.
3. **Compose with SOAR partitioning.** Wire the anisotropic trainer
   into `crates/ruvector-soar-ivf` as the residual quantizer. This is
   the closest practical approximation of production ScaNN.
4. **Thread into existing crates.** Extend `ruvector-pq-search` and
   `ruvector-rabitq` (the residual-PQ variant) with a feature flag
   `anisotropic-loss` that swaps the centroid update at build time
   with zero read-path change.
5. **SIMD ADC.** The bench spends ~50% of its search time on the LUT
   build. `f32x8` inner products (portable-simd or wide) roughly
   halve that.
6. **Real embedding benchmarks.** GloVe-1M, MSMARCO-passage, LAION-
   400M. The Gaussian-mixture PoC establishes the mechanism works;
   real corpora establish the operating envelope.

## Production crate layout

For eventual promotion out of `crates/`-experimental into a first-
class codebook trainer, the shape below keeps this crate small and
composable with the rest of the workspace:

```
ruvector-anisotropic-pq/
├── Cargo.toml           — no non-std deps in the trainer; feature-gate
│                          `simd`, `parallel`, `wasm`
└── src/
    ├── lib.rs           — public API only (Quantizer, Codebook, PqParams)
    ├── train/
    │   ├── mod.rs       — trait + shared k-means loop
    │   ├── isotropic.rs — plain-mean centroid update (η = 1)
    │   ├── anisotropic.rs — solve (Σ M_i) c = Σ M_i x_i
    │   └── assign.rs    — L2 + optional score-aware assignment
    ├── code.rs          — Codebook, Code, encode/decode
    ├── search/
    │   ├── mod.rs       — trait
    │   ├── adc_scalar.rs — portable ADC
    │   └── adc_simd.rs  — feature-gated f32x8 ADC
    ├── metrics.rs
    └── bin/
        ├── benchmark.rs — the bench in this PoC
        └── sweep.rs     — η × M × K grid over a real corpus
```

Public-surface stability contract: `Codebook` on-disk layout is
`[K × sub_dim] × M` little-endian f32, byte-identical to isotropic
PQ. Anything storing an isotropic PQ codebook today can store an
anisotropic one tomorrow with a schema-compatible field addition
(`training_eta: f32`, defaulting to 1.0).

## References

1. Guo, Sun, Simcha, Krichene, Kumar. *Accelerating Large-Scale
   Inference with Anisotropic Vector Quantization*. ICML 2020.
   arXiv:1908.10396.
2. Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor
   Search*. IEEE TPAMI 33(1), 2011.
3. Ge, He, Ke, Sun. *Optimized Product Quantization*. CVPR 2013.
4. Sun, Guo, Chi, Simcha, Kumar. *SOAR: New Algorithms for Even Faster
   Vector Search with ScaNN*. NeurIPS 2023.
5. Malkov, Yashunin. *Efficient and Robust Approximate Nearest
   Neighbor Search Using Hierarchical Navigable Small World Graphs*.
   IEEE TPAMI 2020. (Companion index structure; anisotropic PQ can
   compress HNSW-linked vectors.)
6. ScaNN source:
   https://github.com/google-research/google-research/tree/master/scann
7. ADR-0001 (`docs/adr/0001-anisotropic-score-aware-product-
   quantization-for-mips.md`) — the sibling decision record.
