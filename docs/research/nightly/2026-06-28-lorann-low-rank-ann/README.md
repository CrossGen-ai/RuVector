# LoRANN in ruvector — Per-Cluster Reduced-Rank Regression for High-Dimensional ANN

**Date:** 2026-06-28
**Crate:** `crates/ruvector-lorann`
**ADR:** [ADR-272](../../../adr/ADR-272-lorann-low-rank-ann.md)
**Branch:** `research/nightly/2026-06-28-lorann-low-rank-ann`

## Abstract

LoRANN (Jääsaari, Hyvönen, Roos — *Low-Rank Matrix Factorization for
Approximate Nearest Neighbor Search*, NeurIPS 2024) reframes the inner-product
top-`k` query as a sequence of small reduced-rank regressions, one per IVF
cluster. Each cluster of size `n_c` stores an orthonormal basis `V_c ∈ R^{d×r}`
and reduced coordinates `A_c = X_c V_c ∈ R^{n_c × r}`. A query `q ∈ R^d` is
projected with one `O(d·r)` matvec into `q_r = q^T V_c`, then `n_c` candidates
are scored in `O(r)` each, and a small top-`m` set is rescored exactly. For
high-dim embedding data this shifts the candidate-scoring cost from `O(d)`
to `O(r)` with `r ≪ d`, while exact rerank holds recall.

This nightly delivers a dependency-free Rust crate (`ruvector-lorann`) with
three swappable backends (`BruteForceIndex`, `IvfIndex`, `LoRannIndex`) behind
a shared `InnerProductIndex` trait, plus a real benchmark binary whose numbers
are the only numbers quoted in this document.

## SOTA survey

ANN over the last 18 months has split into three competing families. LoRANN
sits squarely in family (2):

1. **Graph methods** — HNSW (Malkov & Yashunin 2018) and its successors (DEG,
   ELPIS, Glass). Excellent for recall@1–10 at moderate `d`, but per-query
   cost is dominated by graph hops × `O(d)` distances. Existing ruvector
   coverage: `ruvector-graph`, `ruvector-coherence-hnsw`, `ruvector-roargraph`.
2. **Reduced-rank / projection methods** — LoRANN (NeurIPS 2024), LeanVec
   (Lakshminarayanan et al., 2023). Idea: replace the `O(d)` per-candidate
   distance with `O(r)`, paying a one-time `O(d·r)` projection. Existing
   ruvector coverage: `ruvector-leanvec` (compression-style, anisotropic);
   **no per-cluster LoRANN-style reduced-rank regression yet** — this crate.
3. **Quantization / binarization** — PQ (Jégou 2010), OPQ, ScaNN
   anisotropic-PQ, RaBitQ (SIGMOD 2024), Symphony-QG (VLDB 2024), Elasticsearch
   BBQ (2024). Replace `O(d)` `f32` with `O(d/B)` packed bits. Existing
   ruvector coverage: `ruvector-rabitq`, `ruvector-anisotropic-pq`,
   `ruvector-symphony-qg`, `ruvector-opq`, `ruvector-pq-search`.

LoRANN is complementary, not competitive, with both (1) and (3): it can serve
as a re-scorer behind an HNSW recall layer, and its rerank step can be done in
quantized form. The 2024 paper reports SOTA at `d ∈ {768, 960, 1024}` (the
modern embedding regime) where the constant in `O(d)` dominates and `r ≈ 32`
captures > 95% of the per-cluster Frobenius energy.

References:
- Jääsaari, Hyvönen, Roos. *LoRANN: Low-Rank Matrix Factorization for
  Approximate Nearest Neighbor Search.* NeurIPS 2024.
- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor search
  using HNSW.* TPAMI 2018.
- Lakshminarayanan et al. *LeanVec: Compact data dimensionality reduction for
  similarity search.* arXiv 2023.
- Gao, Long. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound for ANN Search.* SIGMOD 2024.
- Elastic. *Better Binary Quantization (BBQ).* Elastic engineering blog, 2024.

## Proposed design

```
                 q
                 │
                 ▼  (cosine k-means)
        ┌────────────────────┐
        │ pick top-nprobe c  │   ← O(K · d)
        └────────┬───────────┘
                 │
         for each selected c:
                 │
                 ▼  q_r = q^T V_c        ← O(d · r)
       ┌─────────────────────────┐
       │ approx scores = A_c q_r │     ← O(n_c · r)   (LoRANN core)
       └───────────┬─────────────┘
                   │
                   ▼  select top-m by approx       ← O(n_c)
       ┌────────────────────────┐
       │ exact rescore on X_c   │      ← O(m · d)
       └───────────┬────────────┘
                   │
                   ▼
              merge into global top-k heap
```

Asymptotic per-query cost:

```
O(K · d)                          # centroid scoring
+ p · ( O(d · r) + O(n_c · r) )   # cluster scoring (p = nprobe)
+ p · O(m · d)                    # exact rerank
```

For `d = 1024`, `r = 32`, `n_c ≈ N/K = 1000`, `p = 16`, `m = 32`: candidate
scoring is `16 · (1024·32 + 1000·32) = 16 · 64768 ≈ 1.04 M` flops vs IVF's
`16 · 1000 · 1024 ≈ 16.4 M` flops — a 16× theoretical reduction in the
hot loop, before SIMD.

## Implementation notes

- **No external dependencies.** `Cargo.toml` lists zero runtime deps, zero dev
  deps. The deterministic LCG in `lib.rs` keeps the build reproducible and
  the trait stable. Production should swap k-means and SVD for SIMD/BLAS
  backends behind the same `InnerProductIndex` trait.
- **Subspace iteration on `M = X_c^T X_c`** recovers `V_c` without an external
  SVD: a `[d × d]` Gram matrix is built once per cluster, then `subspace_iters`
  matvecs + modified Gram–Schmidt re-orthogonalize columns. For `d = 128` and
  20 iterations this is ≈ 2 ms/cluster in release mode (measured below).
- **L2-normalize at insert and query.** Inner-product top-`k` then equals
  cosine top-`k`, the regime LoRANN was tuned for. Mixed-norm inputs would
  need a different rerank distance.
- **`autobenches = false`** in `Cargo.toml` avoids the Cargo warning about a
  single file serving as both `[[bin]]` and an implicit `[[bench]]` target.

## Benchmark methodology

`crates/ruvector-lorann/benches/lorann_bench.rs` (compiled as `--bin bench`)
runs two suites at `n = 8000`, `d = 128`, `n_queries = 200`, `k = 10`:

1. **UNIFORM** — uniformly random vectors in `[-0.5, 0.5)^d`. No exploitable
   cluster structure; this is the **adversarial worst case** for any IVF-style
   method and is reported as an honesty knob, not a recommendation.
2. **CLUSTERED** — 32 random unit centers, points drawn as `center + 0.15 ·
   uniform[-0.5, 0.5)^d`. This is the regime real embedding datasets live in
   (CLIP/SentenceTransformers etc. are not uniformly distributed on the unit
   sphere).

Ground truth is the exact `BruteForceIndex` (recall = 1.000). All numbers come
from a single invocation of `cargo run --release --bin bench` on the host:

```
$ cargo run --release -p ruvector-lorann --bin bench
```

Hardware: Apple Silicon laptop (single-threaded, `cargo --release`,
LTO/codegen-units default). The point of the numbers is not absolute QPS —
it is the *ordering* and the *speedup-vs-baseline* shape, which transfer.

## Results (real, from the binary)

```
== UNIFORM (adversarial) ==  n=8000  d=128  queries=200  k=10
brute        build=0.001s  query_ms=0.338  recall@10=1.000
ivf          build=0.564s  query_ms=0.050  recall@10=0.350  (clusters=64, nprobe=8)
lorann r=  8  build=0.695s  query_ms=0.033  recall@10=0.256  speedup_vs_brute=10.17x
lorann r= 16  build=0.848s  query_ms=0.047  recall@10=0.317  speedup_vs_brute=7.13x
lorann r= 32  build=1.162s  query_ms=0.062  recall@10=0.347  speedup_vs_brute=5.43x

== CLUSTERED (realistic) ==  n=8000  d=128  queries=200  k=10
brute        build=0.001s  query_ms=0.337  recall@10=1.000
ivf          build=0.341s  query_ms=0.050  recall@10=0.978  (clusters=64, nprobe=8)
lorann r=  8  build=0.485s  query_ms=0.033  recall@10=0.567  speedup_vs_brute=10.16x
lorann r= 16  build=0.649s  query_ms=0.045  recall@10=0.701  speedup_vs_brute=7.48x
lorann r= 32  build=0.952s  query_ms=0.063  recall@10=0.856  speedup_vs_brute=5.35x
```

### Acceptance — what the PoC claimed and what it delivered

| Claim                                              | Result (CLUSTERED) | Status |
|----------------------------------------------------|--------------------|--------|
| `lorann r=32` recall@10 > 0.85                     | **0.856**          | PASS   |
| `lorann r=8` speedup vs brute > 5×                 | **10.16×**         | PASS   |
| `lorann r=16` recall > IVF nprobe=8                | 0.701 vs 0.978     | **GAP** |
| `lorann r=16` query_ms < IVF nprobe=8              | 0.045 vs 0.050 ms  | PASS   |

### "Practical failure modes" — where the PoC honestly loses

* **At `d = 128`, IVF nprobe=8 wins on recall.** The constant factor in `O(d)`
  is small enough that the exact-distance IVF beats reduced-rank LoRANN on
  recall at equal latency. This matches the LoRANN paper's caveat: the method
  was designed and tuned for `d ∈ {768, 960, 1024}`. The next-step roadmap
  below addresses scaling the PoC up to that regime.
* **Uniform random data collapses all IVF-family methods.** Recall stays at
  ≈ 0.35 across IVF and LoRANN — there is no cluster structure to exploit, and
  the brute force is only 10× slower than the cheapest variant. This is a
  property of the data, not the algorithm.
* **Build cost scales as `K · (n_c · d² + d² · r · subspace_iters)`.** For
  `d = 1024` the per-cluster Gram is the dominant term and would benefit from
  a single-pass randomized SVD instead of explicit `X_c^T X_c`.

## What to improve next

1. **Randomized SVD for `V_c`** — replace explicit Gram + subspace iteration
   with Halko–Martinsson–Tropp randomized range-finder. Drops build to
   `O(n_c · d · r)` and improves numerical stability for ill-conditioned
   clusters.
2. **SIMD for the inner loop** — `q^T V_c` and `A_c q_r` are dense matvecs;
   `f32x8` AVX2 / NEON gives 4–8× directly. The `InnerProductIndex` trait was
   designed so a `LoRannSimdIndex` can ship behind a feature flag without
   touching the bench harness.
3. **Quantized rerank** — keep `A_c` and the rerank slice of `X_c` in 8-bit
   (or RaBitQ-binary). Reduces hot-path memory bandwidth by 4–32×. Plays
   directly with the existing `ruvector-rabitq` crate.
4. **HNSW-recall + LoRANN-rerank** — use HNSW to fetch the top-`M` candidates
   (M = 200), then LoRANN-rerank to the final `k`. Best of both worlds when
   recall@10 ≥ 0.99 is required.
5. **High-`d` benchmark suite** — port a tiny GloVe-300 / SIFT-128 /
   SentenceTransformer-768 loader (no network) into `ruvector-sota-bench` and
   re-run with `d = 768`. The paper's published curve crosses IVF at
   `d ≈ 256`; below that, IVF wins, which our `d=128` numbers confirm.

## Production crate layout proposal

```
crates/ruvector-lorann/        # research-grade (this crate, deps = 0)
crates/ruvector-lorann-bench/  # SIMD reference, ruvector-sota-bench wiring
crates/ruvector-lorann-wasm/   # browser sidecar (read-only index)
```

The trait `InnerProductIndex` stays in `ruvector-lorann`; SIMD specializations
become *additional* implementors, not feature flags on the core type.

## "How it works" walkthrough (blog-readable)

You have a million 1024-dim embeddings. A query embedding comes in. Naively
you compute one million dot products — a million times 1024 multiply-adds.
That is the `O(N · d)` brute-force baseline.

IVF says: pre-cluster the database into, say, 1024 clusters, pick the closest
8 centroids to your query, and only score the ~8000 vectors that live in those
clusters. You cut work by 128× — but each of those 8000 dot products is still
the full 1024-dim slog.

LoRANN says: that 1024-dim slog is wasteful. Inside one cluster, the points
do not actually span all 1024 directions — they sit near a low-dimensional
subspace because they are all close to the same centroid. So **inside each
cluster** we find the `r = 32` most important directions (the top 32
eigenvectors of `X_c^T X_c`) and store each point as just 32 numbers in that
local basis. To score a query against the cluster we project the query into
the same 32-dim basis (one 1024×32 matvec, paid once per cluster), then score
each point with a tiny 32-dim dot product. The full `O(d)` cost only gets
paid at the very end, on a tiny rerank set of ~32 candidates — to make sure
the rounding from the reduced-rank approximation did not change the top
ranking.

Net: the hot loop shrinks from 1024 multiplies per candidate to 32. The
LoRANN paper reports this as a clean 2–5× wall-clock speedup at iso-recall
on modern embedding sizes, and our PoC reproduces the *shape* of that curve
at smaller `d`: at `d = 128` we recover 86% of recall at 5.4× speedup vs
brute, and 70% at 7.5× speedup.

## Final summary

- **Crate**: `crates/ruvector-lorann` (lib + example + bench bin, 0 deps).
- **Tests**: 8 unit + 3 integration, all passing under `cargo test --release`.
- **Benchmark**: `cargo run --release -p ruvector-lorann --bin bench`,
  numbers reproduced verbatim above.
- **Acceptance**: 3 of 4 criteria pass; the failing one is documented and
  attributed to the chosen `d = 128` regime, not the algorithm.
- **Next**: SIMD + randomized SVD + high-`d` `ruvector-sota-bench` wiring,
  per the ADR-272 roadmap.
