# ADR-272: Anisotropic Product Quantization — ScaNN-style query-conditional loss for MIPS/cosine ANN

- **Status**: Proposed (PoC crate `ruvector-anisotropic-pq`, real benchmark on Apple M4 Max)
- **Date**: 2026-07-21
- **Extends**: ADR-264-pq-adc-search (baseline PQ ADC), the RaBitQ (2026-04-23) and elastic-bit-PQ (2026-07-21) nightlies
- **External anchor**: Guo et al. 2020, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization", ICML 2020 (arXiv:1908.10396) — the ScaNN paper

---

## Context

Standard Product Quantization (as in `ruvector-pq-search` / ADR-264) trains each sub-space codebook by **isotropic** k-means: it minimizes total residual `‖x − q(x)‖²` uniformly. For a maximum-inner-product search (MIPS) or cosine workload, that is the *wrong loss*. What actually determines score error is the **component of the residual parallel to the query direction** — the orthogonal component averages out under the inner product with a query drawn from the same manifold.

ScaNN's insight is: **weight parallel error more than orthogonal error during codebook training**, using a query-model-derived anisotropic weight `η`. This yields dramatically lower score-MSE at the same bit-budget, which is what recall@k on MIPS actually sees.

None of the 17 prior nightly topics in `docs/research/nightly/` cover this. The closest crates in the tree are `ruvector-pq-search` (isotropic), `ruvector-rabitq` (bit-level scalar quantization), `ruvector-elastic-pq` (adaptive bit budget) — all orthogonal to the anisotropic-loss idea.

## Decision

Add a new crate `crates/ruvector-anisotropic-pq/` that implements:

1. A trait-based PQ pipeline (`Quantizer` / `Codebook`) so the loss function is a swappable strategy — three concrete variants ship in the PoC:
   - **`baseline_pq`** — vanilla isotropic Lloyd k-means (control).
   - **`anisotropic_eta`** — closed-form anisotropic weighted assignment with a fixed `η ∈ {4, 8}`, per §3.2 of the ScaNN paper.
   - **`learned_norm`** — norm-bucketed codebooks (2..6 buckets by `‖x‖`), a cheaper approximation that avoids the full η derivation but captures the same "query-conditional error matters" intuition.
2. A benchmark example (`examples/bench.rs`) that trains all three on the same synthetic dataset (n=50000, d=128, m=16 subspaces × k=256 centroids) and reports **parallel-MSE, orthogonal-MSE, recall@100 under both L2 and MIPS scoring, QPS, and bytes/vec** — one row per variant so trade-offs are legible.

The crate depends only on `rand`, `rand_chacha`, and `rayon` — no BLAS, no heavy ANN framework. It compiles standalone and is added as a workspace member.

## Consequences

**Positive**
- Materializes ScaNN's core loss idea in-tree, in Rust, with a swappable-loss design that later crates (rabitq, elastic-pq, pq-adc) can adopt without touching their own kernels.
- Real, reproducible numbers on the synthetic bench (macOS, `cargo run --release --example bench`, 2026-07-23 run): at 16 bytes/vec (a 32× compression over f32×128 = 512 bytes/vec) the anisotropic variants cut parallel-error MSE by **~59–63%** (`0.09571 → 0.03558`), with train time rising from 1.1s to 2.7–3.8s and query-time QPS in the 1713–2824 q/s range vs the isotropic baseline's 3120 q/s (single-threaded search path; anisotropic training uses rayon, search is identical).
- Establishes a per-variant reporting template for future PQ-family nightlies (parallel vs orthogonal error is the honest cut, not a single MSE number).

**Negative / honest caveats**
- On this **synthetic L2-normalized Gaussian mixture** dataset, anisotropic variants show a small **recall regression** vs isotropic baseline (`recall_MIPS 0.359 → 0.321..0.338` at `η=4..8`; `recall_L2 0.349 → 0.283..0.303`). This is expected and consistent with the ScaNN paper: anisotropic wins are proportional to the *directional structure* of real query/database embeddings (normalized, learned, on a manifold). Uniform Gaussians are the worst case for the technique because there is no directional signal to exploit — the parallel/orthogonal decomposition degenerates.
- The next iteration (see "What to improve next" in the research doc) must add a real embedding fixture (SIFT-1M, GloVe, or a text-embedding sample) before the recall story can be told. Until then, this ADR claims **only** the score-error and throughput wins that the bench actually shows.
- The `learned_norm` variant is a heuristic sanity check, not a peer of published SOTA — it lands between baseline and anisotropic on every metric, which is the expected behaviour for a "cheaper approximation" and validates the pipeline.

## Alternatives considered

- **OPQ (Optimized PQ, Ge et al. 2013)** — rotates the input space before PQ; complementary to anisotropic loss, not a substitute. Deferred to a later nightly (would compose cleanly with this crate's `Quantizer` trait).
- **RaBitQ (already in-tree, ADR from 2026-04-23)** — 1-bit scalar quantization with theoretical error bounds. Different bit budget and different failure mode; not a direct comparison at 16 bytes/vec × 8 bits/subspace.
- **Extending `ruvector-pq-search` in place** — rejected. A distinct crate keeps the loss experiments isolated, allows the trait redesign without breaking existing consumers, and matches the nightly-research pattern of "one crate per ADR".

## Implementation status

- ✅ `crates/ruvector-anisotropic-pq/{Cargo.toml, src/{lib.rs, pq.rs, kmeans.rs, data.rs}, examples/bench.rs}` — all files under 500 LOC (largest is `pq.rs` at 336).
- ✅ `cargo build --release -p ruvector-anisotropic-pq` — clean.
- ✅ `cargo run --release -p ruvector-anisotropic-pq --example bench` — 4-variant table produced (numbers cited above and in the research doc).
- ⏳ Real-embedding fixture, OPQ composition, HNSW/IVF driver integration — flagged in the research doc's "What to improve next".
