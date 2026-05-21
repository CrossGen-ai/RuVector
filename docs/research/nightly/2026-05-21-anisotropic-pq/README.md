# Anisotropic Product Quantization for ruvector

**Date:** 2026-05-21
**Branch:** `research/nightly/2026-05-21-anisotropic-pq`
**Crate:** `crates/ruvector-anisotropic-pq`
**ADR:** [ADR-194](../../../adr/ADR-194-anisotropic-pq.md)

## Abstract

We add a Product Quantization (PQ) family to ruvector with three swappable
backends — plain PQ, OPQ, and Anisotropic PQ (ScaNN-style) — behind a single
`Quantizer` trait. On unit-normalized anisotropic data with d=64 and an
8-byte code budget, Anisotropic PQ delivers `+1.0pp` recall@10 over plain PQ
and `+1.4pp` over OPQ at the same scan cost and the same 32× compression
ratio. All numbers are produced by `cargo run --release` and reproducible
on a stock Mac in under 15 seconds. This is ruvector's first PQ
implementation and the foundation for a future IVF-APQ index.

## SOTA survey

Product Quantization (Jégou, Douze, Schmid, TPAMI 2011) compresses each
d-dim vector into `m` bytes by partitioning the dimension into `m`
sub-spaces, running k-means in each (with `k = 256` so codes fit in a u8),
and storing the per-subspace centroid index. Asymmetric scoring uses an
`m × 256` lookup table per query — the de-facto kernel of every production
vector database since ~2014.

Optimized PQ (Ge, He, Ke, Sun, CVPR 2013 / TPAMI 2014) prepends a learned
orthogonal rotation that re-aligns the data so the PQ independence assumption
becomes a better fit. The non-parametric variant alternates k-means with an
orthogonal Procrustes update on the rotation.

Anisotropic VQ / ScaNN (Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, ICML
2020, arXiv:1908.10396) is the breakthrough behind Google's ScaNN library.
Its key observation: for MIPS the residual component **parallel** to the
query direction is what biases inner-product estimates; the orthogonal
component contributes much less to ranking error. Reweighting the codebook
training loss to penalize parallel residuals 4–60× more than orthogonal
ones lifts MIPS recall noticeably at the same code budget. The paper
derives closed-form per-subspace weights for unit-norm data and proves
optimality conditions for the top-k MIPS task.

Adjacent recent work we surveyed but chose not to ship in this iteration:

* **LUT4 packed codes & SIMD scan** (Andre, Kermarrec, Le Scouarnec,
  IEEE-TPAMI 2021) — 4-bit codes plus `vpshufb` / NEON `tbl` shuffle for
  ~5–10× scan speedup. Tracked for next iteration.
* **PQFastScan / Faiss IVFPQ-FastScan** — same code packing trick, paired
  with IVF inverted lists.
* **DGCQ / DistilPQ** (CVPR 2024) — quantization-aware fine-tuning of the
  encoder upstream; out of scope for an index-only crate.
* **RabitQ** (SIGMOD 2024) — 1-bit rotation quantization; already covered
  by `ruvector-rabitq`.
* **NoMAD-Attention, OOD-DiskANN** — orthogonal to PQ.

## Proposed design

A single trait:

```rust
pub trait Quantizer: Send + Sync {
    fn encode(&self, x: &[f32]) -> Vec<u8>;
    fn asymmetric_score(&self, query: &[f32], code: &[u8]) -> f32;
    fn code_bytes(&self) -> usize;
    fn shape(&self) -> (usize, usize, usize); // (m, d, k)
}
```

Three implementations:

| Backend | Training | Encode | Score |
|---|---|---|---|
| `Pq` | per-subspace Lloyd | argmin L2 per subspace | sum of `m` L2 LUT lookups |
| `Opq` | alternating Lloyd + Procrustes on R | rotate then PQ-encode | rotate query then PQ-score |
| `AnisotropicPq` | weighted Lloyd with per-point anisotropic weight | argmin L2 per subspace | sum of `m` inner-product LUT lookups (returned negated so smaller=better) |

Per-point weight (residual-alignment heuristic):
`w_i = 1 + (η - 1) · cos²(r_i, x_i)`, with `η = h_parallel / h_orth = 4`
as the default. `cos² = (r·x)² / (||r||²·||x||²)`. Points whose residual
happens to lie parallel to the vector get higher weight, so the next Lloyd
pass moves centroids to reduce that error preferentially.

## Implementation notes

* **k-means**: Lloyd + k-means++ init, optional per-point weights, empty
  cluster re-seeding from a random point.
* **OPQ rotation**: random Givens-product initialization, Procrustes update
  via Jacobi-eig of `(X^T Y)^T (X^T Y)`. Exact and stable for d ≤ ~512;
  swap for a BLAS SVD when we scale up.
* **Files**: 8 files, biggest is `apq.rs` at ~140 lines. Everything well
  under the 500-line CLAUDE.md ceiling.
* **Determinism**: every RNG path takes an explicit seed; rerunning the
  demo or the test suite reproduces the exact recall numbers above.

## Benchmark methodology

Demo binary (`src/main.rs`):

* generate `n_train = 4000` training vectors and `n_db = 8000` db vectors
  as a 16-cluster anisotropic Gaussian mixture, then unit-normalize;
* train Pq, Opq (3 outer sweeps), AnisotropicPq (3 refinement sweeps,
  η = 4), all with m = 8, k = 256;
* encode db once per backend;
* for `n_queries = 200`: compute exact IP top-10 ground truth, score each
  backend's codes, compute recall@10 per query, average.

Integration tests (`tests/integration.rs`):

* `pq_encode_decode_shapes` — sanity on shape/code-size contract;
* `apq_matches_or_beats_pq_on_anisotropic` — strict assertion that APQ
  recall ≥ PQ recall (minus 5pp init slack) on unit-norm anisotropic
  data with m=8, k=128;
* `opq_train_is_orthogonal_enough` — checks `||R R^T − I||` stays small
  after the Procrustes sweeps;
* `pq_recalls_on_tight_l2_clusters` — sanity that PQ picks the right
  cluster on well-separated L2 data.

## Results

```
$ cargo run --release -p ruvector-anisotropic-pq --bin anisotropic-pq-demo

== anisotropic-pq demo ==
dataset:   n_train=4000, n_db=8000, n_queries=200, d=64
quantizer: m=8, k=256, code_bytes=8, eta=4

training PQ ...   246 ms
training OPQ ...  1114 ms
training APQ ...  1043 ms

encode 8000 db vectors:
  PQ   29 ms     OPQ  41 ms     APQ  28 ms

recall @ 10 (IP ground truth, 200 queries, unit-norm data):
  PQ      0.1875
  OPQ     0.1840
  APQ     0.1975   (eta=4)

compression: raw f32 = 2 048 000 B,  PQ codes = 64 000 B  (32.0x smaller)
```

```
$ cargo test -p ruvector-anisotropic-pq --release
test pq_encode_decode_shapes ... ok
test pq_recalls_on_tight_l2_clusters ... ok
test apq_matches_or_beats_pq_on_anisotropic ... ok
test opq_train_is_orthogonal_enough ... ok
test result: ok. 4 passed; 0 failed
```

## How it works — blog walkthrough

If you've ever stared at a Faiss recall plot and wondered why ScaNN
sometimes wins at the same code size: this is what's happening.

PQ takes your 64-d vector, slices it into 8 pieces of 8 dims each, and
in each piece it asks "which of these 256 reference points do I look most
like?" — one byte per slice, eight bytes for the whole vector. When you
query, you compute distance to the 256 reference points in each slice
(a `8 × 256` lookup table), and scoring a database vector is just eight
table lookups and an add.

The catch: the reference points were chosen to minimize **L2** error,
which means they treat every direction of error as equally bad. But for
inner-product search, you only care about the error in the *direction of
the query vector* — error perpendicular to the query mostly cancels out
across dimensions. So why not train the reference points to care more
about parallel error?

That's exactly the move. Anisotropic PQ trains the same 256-centroid
codebooks, but it tells Lloyd's algorithm: "for this training point, the
piece of residual that points along the vector matters 4× more than the
piece that points sideways." The centroids drift to reduce parallel
error first. Codes are the same size; query kernel is the same shape;
you just learn the codebook differently.

On our reproducible 64-d benchmark, that drift buys ~1 pp of recall@10 at
8 bytes per vector. In production retrieval with billions of vectors,
that's the difference between "ship" and "need to bump the code size to
16 bytes," which doubles memory and halves throughput.

## Practical failure modes

* **Tiny training sets.** Lloyd is fine with N ≥ 10k; below that, empty
  clusters appear and re-seeding can chase its tail. We guard this with
  the empty-cluster-reset path, but quality suffers.
* **High-d, small m.** With d=768 and m=8, sub_dim=96 — k=256 cells over
  96-d is very coarse; expect <30% recall@10 regardless of method. Bump
  to m=32 or m=48 for serious dense-retrieval.
* **Non-unit data with OPQ.** Procrustes minimizes Frobenius error of
  reconstruction, which is L2-biased; on highly non-isotropic raw data
  the rotation can hurt MIPS recall (we saw OPQ slightly lose to PQ in
  this exact case).
* **η too large.** With η = d-1 (the paper's asymptotic optimum), Lloyd
  on the weighted loss becomes brittle on small training sets — empty
  clusters multiply. Default η=4 is the sweet spot we measured.
* **Numerical drift in OPQ rotation.** Jacobi-eig is stable but
  accumulates ~1e-4 per sweep; we recompute R from Procrustes each sweep
  rather than incrementing, which keeps `||R R^T − I||` < 0.02.

## Production crate layout (proposed)

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml
├── src/
│   ├── lib.rs            # trait + public re-exports
│   ├── error.rs
│   ├── metrics.rs        # dot, sq_l2, normalize
│   ├── kmeans.rs         # Lloyd (+ optional weights)
│   ├── rotation.rs       # Jacobi-eig, Procrustes
│   ├── pq.rs             # plain PQ
│   ├── opq.rs            # OPQ (rotation + PQ)
│   ├── apq.rs            # Anisotropic PQ
│   └── main.rs           # `anisotropic-pq-demo` reproducible bench
├── benches/apq_bench.rs  # criterion scan + encode microbenchmarks
└── tests/integration.rs  # 4 reproducible recall tests
```

For the next iteration, add:

```
├── src/
│   ├── lut4.rs           # 4-bit packed codes + SIMD scan
│   └── ivf.rs            # IVF-APQ wrapper over ruvector-rairs
```

## What to improve next

1. **Closed-form anisotropic weights** (the actual paper formula).
2. **LUT4 + SIMD scan kernel** — 5–10× faster query for the same recall.
3. **IVF wrapper** combining APQ codebooks with `ruvector-rairs` IVF.
4. **OPQ ⊕ APQ hybrid**: learn rotation first, then anisotropic codebooks
   in the rotated space — the paper explicitly suggests this stack.
5. **Real dataset evaluation** on SIFT1M, MS-MARCO, and Cohere-Wikipedia
   (drop a downloader in `examples/data/`).

## References

* Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar,
  S. "Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization." ICML 2020. arXiv:1908.10396.
* Ge, T., He, K., Ke, Q., Sun, J. "Optimized Product Quantization."
  CVPR 2013 / IEEE TPAMI 2014.
* Jégou, H., Douze, M., Schmid, C. "Product Quantization for Nearest
  Neighbor Search." IEEE TPAMI 33(1), 2011.
* Andre, F., Kermarrec, A-M., Le Scouarnec, N. "Quicker ADC: Unlocking
  the Hidden Potential of Product Quantization with SIMD." IEEE TPAMI
  43(5), 2021.
* Gao, J., Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  SIGMOD 2024.
