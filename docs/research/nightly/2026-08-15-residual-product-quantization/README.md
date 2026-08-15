# Residual Product Quantization for Compressed ANN — a measured study

*Nightly research, 2026-08-15. Crate: `crates/ruvector-rpq`. ADR-306.*

## Abstract

We add a dependency-free Rust reference implementation of two-level
**residual product quantization** (RPQ2) to `ruvector`, alongside a
single-level **product quantizer** (PQ) and an 8-bit **scalar
quantizer** (SQ8) baseline. All three share a common `Quantizer` trait
and asymmetric distance computation (ADC). A `cargo run --release`
benchmark on a 20 000-vector clustered-Gaussian synthetic (dim = 64)
measures encode/query time, memory, and recall@10 against exact
brute-force ground truth. The main empirical finding is a candid
negative result: **RPQ2 with (m₁, m₂) = (8, 8) does not beat plain PQ
with m = 16 at equal 16-byte storage** (5.6 % vs 8.2 % recall@10).
RPQ2's proper regime is IVFADC-style deployments where the coarse
layer is amortised across all items in a coarse cell — a follow-up
integration path this PoC directly enables.

## SOTA survey

* **Product Quantization for nearest-neighbor search** (Jégou, Douze,
  Schmid, *TPAMI 2011*, [arXiv:1102.3828](https://arxiv.org/abs/1102.3828))
  — introduced PQ + ADC and the coarse+residual IVFADC construction that
  RPQ2 generalises.
* **Optimized Product Quantization** (Ge, He, Ke, Sun, *TPAMI 2013*)
  — jointly learns a rotation and PQ codebooks; complementary to RPQ2.
* **Additive/Residual Vector Quantization** (Chen, Guan, Wang, *ECCV 2010*;
  Babenko & Lempitsky, *CVPR 2014*, "Additive quantization for extreme
  vector compression") — generalises RPQ to `K` stages with an
  iterative encoding search; higher accuracy, higher encoding cost.
* **DiskANN / SPANN / SCaNN / FAISS-IVFPQ / Milvus IVF_PQ / Qdrant PQ /
  Weaviate PQ / Pinecone PQ** all expose or internally use a two-level
  quantizer of some flavour — evidence that RPQ variants are practical
  workhorses even in 2026, not merely historical.
* **RaBitQ** (Gao & Long, *SIGMOD 2024*) — the current 1-bit frontier;
  already available in-tree as `ruvector-rabitq`. Complementary to
  RPQ2 (different bit/recall regime).
* **LVQ / Turbo-PQ** (Aguerrebere et al., *VLDB 2023–2024*, Intel Scalable
  Vector Search) — locally adaptive scalar variants; complementary
  memory-latency envelope.

Recent competitor release notes (Qdrant 1.11 PQ improvements, Milvus 2.4
IVF_PQ tuning guide, Weaviate 1.24 PQ retraining, LanceDB 0.10 PQ) confirm
sustained industry interest in classical PQ variants — hence the value of
a clean, in-tree, benchmarked reference.

## Proposed design

A single-file library exposes:

```rust
pub trait Quantizer: Send + Sync {
    fn dim(&self) -> usize;
    fn code_bytes(&self) -> usize;
    fn encode(&self, x: &[f32], out: &mut [u8]);
    fn adc_sq_distance(&self, query: &[f32], encoded: &[u8]) -> f32;
    fn name(&self) -> &'static str;
}
```

Three backends implement it:

* `Pq { m, k = 256, codebooks }` — classic PQ with 8-bit codes per subspace.
* `Rpq2 { pq1, pq2 }` — first PQ encodes `x`, second PQ encodes
  `x - PQ1⁻¹(PQ1(x))`. Layout: `code = [code₁ ‖ code₂]`.
* `Sq8 { lo, scale, inv_scale }` — per-dim uniform 8-bit scalar quantizer.

An `Rpq2Scorer` groups database items by their coarse code so the
residual LUT is built once per coarse code rather than once per item —
the standard amortisation trick.

## Implementation notes

* No external crates. `#![forbid(unsafe_code)]`. Deterministic seed
  (xorshift64*).
* K-means++ seeding + Lloyd iterations for codebook training. Empty
  clusters re-seeded from a random training point.
* Every allocation is tied to a `Vec<f32>` / `Vec<u8>`; no arena tricks.
* `Pq::compute_lut(query)` + `Pq::adc_from_lut(lut, code)` expose the
  amortised query path that a production scorer uses.

## Benchmark methodology

* Synthetic dataset: mixture of 64 isotropic Gaussians (σ = 0.5) in
  ℝ⁶⁴, centroids drawn N(0, 3²). 10 000 train, 20 000 database, 200
  queries — all disjoint. Deterministic seeds.
* Ground truth: exact brute-force top-10 squared-L2 per query.
* Metric: mean recall@10 over 200 queries.
* Timing: monotonic clock via `std::time::Instant`. Release profile
  (opt-level = 3, thin LTO, codegen-units = 1 inherited from workspace).
* Hardware: Apple Silicon developer laptop (macOS). Numbers are
  per-run; variance across three runs was ≤ 5 % on all rows.
* Reproduce: `cargo run --release -p ruvector-rpq --bin rpq-bench`.

## Results

```
ruvector-rpq bench: dim=64 n_train=10000 n_db=20000 n_q=200 k=10
                    pq_small=8 pq_fair=16 rpq2=(8,8) iters=25

Results (dim=64, n_db=20000, n_q=200, k=10):
------------------------------------------------------------------------
pq-8         | code= 8B  | train=1028.9 ms | encode= 71.1 ms | query=0.08 ms/q | recall@10=0.036
pq-16        | code=16B  | train=1394.2 ms | encode= 96.7 ms | query=0.12 ms/q | recall@10=0.082
rpq2 (amort) | code=16B  | train=1957.4 ms | encode=155.1 ms | query=53.23 ms/q| recall@10=0.056
sq8          | code=64B  | train=   0.4 ms | encode=  0.4 ms | query=0.38 ms/q | recall@10=0.894
------------------------------------------------------------------------

raw f32 database: 4.88 MiB
  pq-8            ->  0.15 MiB (compression 32.0x)
  pq-16           ->  0.31 MiB (compression 16.0x)
  rpq2 (amort)    ->  0.31 MiB (compression 16.0x)
  sq8             ->  1.22 MiB (compression  4.0x)
```

### What the numbers say

* At **equal coarse-layer budget** (8 B), RPQ2 raises recall from
  3.6 % → 5.6 %, a **1.56× gain**, at the cost of doubling storage
  and encode time.
* At **equal total storage** (16 B), single-level `pq-16` beats
  `rpq2` on recall (**8.2 % vs 5.6 %**) and is 400× faster at query
  time. This is a candid negative result for RPQ2 as a *drop-in*
  replacement for PQ.
* `sq8` remains the recall king at 4× compression — the classic
  reminder that if you can afford the bytes, uniform scalar
  quantization is remarkably competitive.
* The amortised `Rpq2Scorer`'s query cost is dominated by residual-LUT
  rebuilds. In this in-memory sweep the coarse codes are effectively
  unique (256⁸ addressable, 20 k items), so each item pays a full LUT.
  In an IVF context (thousands of items per coarse cell) this
  amortises to near-PQ cost — the setting where RPQ2 is designed to win.

## How it works (blog-readable walkthrough)

Imagine you want to compress each 64-D float vector (256 bytes) down to
16 bytes and still find approximate nearest neighbours.

1. **Split** the 64 dimensions into 8 chunks of 8. Train a small k-means
   (k = 256) on each chunk. Now each 8-D chunk becomes one byte (its
   nearest centroid). That's plain **PQ**: 8 bytes per vector.
2. **Score** a query by computing 8 tiny distance tables once, then
   summing the 8 bytes' table entries per database item. That's ADC.
3. **RPQ2** adds a second layer: encode the *residual* (what the first
   PQ got wrong) with another PQ. You spend another 8 bytes but the
   codes now describe finer structure.
4. To score in RPQ2 you rebuild the second-layer table with a query
   that has been shifted by the coarse-layer reconstruction. Different
   coarse codes → different shifted queries → different second-layer
   tables. If many items share the same coarse code (IVFADC), the
   second-layer table is built once and reused. That's the payoff.

Our benchmark exposes the trap: if you use RPQ2 in a *flat* index with
no coarse sharing, the second-layer LUT overhead dominates and you're
better off just doubling `m` in a single-level PQ. RPQ2 belongs in an
IVF or graph-coarse-cell context.

## Practical failure modes

* **Flat-index misuse.** RPQ2 in a flat scan is *slower and less
  accurate* than PQ at the same storage. Do not deploy it without a
  coarse partition.
* **Under-trained coarse layer.** If PQ1 quantizes badly, the
  residuals inherit that bias and PQ2 has less headroom. Train PQ1
  with enough Lloyd iterations (≥ 20) and a training set ≥ 10× the
  number of centroids per subspace.
* **Degenerate subspaces.** With `sub_dim < 4`, k = 256 centroids
  overfit; recall drops. Our benchmark's `pq-16` at sub_dim = 4 still
  outperforms `rpq2 (8,8)` at sub_dim = 8, showing that "wider" PQ
  can matter more than "deeper" PQ at small dimensions.
* **Non-Gaussian data.** Real embeddings (e.g., normalized cosine)
  benefit substantially from an OPQ-style rotation *before* PQ. We
  did not measure that here.
* **Concurrency.** The current crate is `Send + Sync` but not
  parallel; encoding a million vectors is CPU-bound on one core.

## What to improve next (roadmap)

1. **OPQ rotation.** Learn an orthonormal rotation matrix jointly with
   PQ codebooks; drop-in replacement for `Pq`. Expected +2-5 % recall.
2. **Rayon-parallel k-means and encoding.** Preserve the zero-dep bar
   with a feature flag `parallel = ["rayon"]`.
3. **AQ / RVQ multi-stage.** Iterate coarse↔fine encoding with beam
   search; higher recall than RPQ2 for the same total bits, at higher
   encoding cost.
4. **IVFADC integration.** Wire `Rpq2` as the per-cell fine codec into
   a future `ruvector-ivfpq` crate; that's the deployment path where
   the numbers reverse.
5. **AVX-512 / NEON `adc_from_lut`.** The inner sum is trivially
   vectorisable (m ≤ 32 gathers per item); expected 3-6× speed-up.
6. **Sub-quantized queries (SDC).** Support symmetric distance computation
   for pure-code / pure-code re-ranking in graph beam search.

## Production crate layout proposal

If this graduates from PoC to product:

```
crates/
  ruvector-rpq/                # this PoC — kept as reference impl.
  ruvector-rpq-simd/           # AVX-512 / NEON kernels, feature-gated
  ruvector-opq/                # rotation-learning wrapper
  ruvector-ivfpq/              # IVF coarse layer + Rpq2 fine layer
```

Trait `Quantizer` stays stable; the SIMD crate provides an
`AdcKernel` extension trait for hot paths.

## References

1. H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest
   Neighbor Search.* IEEE TPAMI, 33(1):117-128, 2011.
2. T. Ge, K. He, Q. Ke, J. Sun. *Optimized Product Quantization for
   Approximate Nearest Neighbor Search.* IEEE TPAMI, 2013.
3. Y. Chen, T. Guan, C. Wang. *Approximate Nearest Neighbor Search by
   Residual Vector Quantization.* Sensors, 2010.
4. A. Babenko, V. Lempitsky. *Additive Quantization for Extreme Vector
   Compression.* CVPR 2014.
5. J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound.* SIGMOD 2024.
6. C. Aguerrebere, I. Bhati, M. Hildebrand, M. Tepper, T. Willke.
   *Similarity Search in the Blink of an Eye with Compressed Indices
   (LVQ).* VLDB 2023.
7. FAISS documentation, `IndexIVFPQ`, `IndexResidualQuantizer`.
8. Milvus 2.4 IVF_PQ tuning notes; Qdrant 1.11 PQ notes; Weaviate 1.24
   PQ notes; LanceDB PQ compression documentation.

## Reproducibility

Everything in this study is reproducible from a fresh checkout of the
CrossGen-ai fork on branch `research/nightly/2026-08-15-residual-product-quantization`:

```
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-08-15-residual-product-quantization
cargo test --release -p ruvector-rpq
cargo run  --release -p ruvector-rpq --bin rpq-bench
```

All six unit tests pass. The `rpq-bench` binary emits the exact table
above (± machine noise).
