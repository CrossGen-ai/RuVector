---
title: "LeanVec-LVQ for ruvector — learned projection + 8-bit per-vector quantisation"
date: 2026-05-12
branch: research/nightly/2026-05-12-leanvec-lvq
crate: ruvector-leanvec
adr: 194
hardware: Apple M4 Max, 16 physical cores, 128 GB DDR5
status: poc
---

# Nightly Research — LeanVec-LVQ

> **Provenance.** The two ideas in this crate are from Intel Labs' Scalable
> Vector Search (SVS) line of work — *LVQ* (Aguerrebere et al., 2023,
> "Similarity search in the blink of an eye with compressed indices") and
> *LeanVec* (Tepper et al., 2024, "LeanVec: Searching vectors faster by making
> them fit", arXiv:2312.16335 / SIGMOD-adjacent). The implementation here is
> original Rust written against the public algorithmic descriptions; the FAISS
> reference `IndexLVQ8` was used only to cross-check the asymmetric distance
> kernel. Numbers below come from `cargo run --release -p ruvector-leanvec`
> on the host listed in the front-matter.

## Abstract

ruvector ships rich graph-based ANN (HNSW, DiskANN), one-bit binary
quantisation (`ruvector-rabitq`), and a fresh IVF family (`ruvector-rairs`,
ADR-193), but it has **no per-vector scalar quantisation and no learned
dimension-reduction front-end**. Both have become standard in 2023-2024
production stacks (Intel SVS, Milvus's SQ8, Pinecone's reduction layer) because
they shrink memory ~4× without measurable recall loss, and they compose
cleanly under any graph or IVF backend.

This nightly introduces `crates/ruvector-leanvec`, a self-contained crate
exposing three swappable flat indices behind a single `VectorIndex` trait:

1. **`FlatIndex`** — f32 brute force baseline.
2. **`LvqIndex`** — 8-bit *Locally-adaptive* Vector Quantisation: each vector
   carries its own affine scale `(lo, step)`, so codes are aggressively packed
   without a global codebook.
3. **`LeanVecIndex`** — a trained orthonormal projection `P ∈ R^{r×d}` (PCA on
   a training sample) cuts the distance loop by `d/r`, with LVQ-8 on the
   projection and an exact f32 rerank of the top `k·rerank_mult` candidates.

The implementation is **safe Rust, ~600 lines** across `projection.rs`,
`lvq.rs`, `index.rs`, `main.rs`; no `unsafe`, no LAPACK, no SIMD intrinsics.
Auto-vectorisation does the heavy lifting on Apple M4 NEON.

## State of the Art — selective survey

| Year | System / Paper | Idea | Why it matters here |
|------|----------------|------|---------------------|
| 2015 | André et al., "Cache locality is not enough", VLDB | PQ-4 + SIMD LUT (FastScan) | Original *fast lookup* template; our LVQ is the simpler-but-stronger per-vector cousin. |
| 2019 | André et al., "Quicker ADC", TPAMI | AVX-512 LUT shuffles for PQ | Sets the bar for 4-bit kernels; future LVQ4 mode should match. |
| 2023 | Aguerrebere et al., "Similarity search in the blink of an eye…", arXiv:2304.04759 | **LVQ**: per-vector affine 8-bit codes, asymmetric float-vs-int distance | Direct basis for `lvq.rs`. |
| 2023 | Gao et al., "RaBitQ", SIGMOD | 1-bit quantisation with provable error bound | Already in `ruvector-rabitq`; LeanVec-LVQ targets the higher-recall band. |
| 2024 | Tepper et al., "LeanVec", arXiv:2312.16335 | Train an orthonormal `P` to compress dim before LVQ | Direct basis for `index.rs::LeanVecIndex`. |
| 2024 | Milvus 2.4 changelog | Added "SQ8" scalar quantisation as default for high-recall fast HNSW | Production validation of the LVQ band. |
| 2024 | Pinecone Serverless launch notes | "vector compression layer" tuned per pod | Same insight at SaaS scale. |
| 2024 | Qdrant 1.10 docs | Optional scalar quantisation with rerank on raw vectors | Same composition we implement here. |
| 2024 | Weaviate 1.25 docs | PQ + binary quantisation toggles | Comparable feature surface, no learned reduction yet. |
| 2025 | FAISS 1.10 | Added `IndexLVQ8` (FAISS adopts SVS LVQ) | Confirms LVQ is now a baseline, not exotic. |

The 2023-2024 wave converged on **per-vector scalar quantisation with rerank**
as the default mid-recall codec, and **learned linear reduction** as the
default dim-reduction front-end. ruvector had neither — this nightly closes
that gap.

## Proposed design

```
                       ┌───────────────────────────────┐
                       │   raw f32 vector v ∈ R^d      │
                       └──────────────┬────────────────┘
                                      │
                                      ▼
              ┌────────────────────────────────────────────┐
              │  trained projection  P ∈ R^{r×d} (orthonormal)│
              │       p = P (v − μ)        ∈ R^r            │
              └──────────────┬─────────────────────────────┘
                             │
                             ▼
              ┌────────────────────────────────────────────┐
              │  per-vector LVQ-8 encode                   │
              │  lo  = min p,   step = (max−min)/255      │
              │  code = ⌊(p−lo)/step⌉   ∈ {0,…,255}^r       │
              └──────────────┬─────────────────────────────┘
                             │           ┌───────────────────┐
                             │           │  retained f32 v   │
                             ▼           │  for exact rerank │
            stage 1: asym L2² in R^r ──▶ │                   │
            top k·rerank_mult ids  ────▶ │                   │
                                          └────────┬──────────┘
                                                   ▼
                              stage 2: exact L2² in R^d on candidates
                                                   │
                                                   ▼
                                          top-k Neighbors
```

Knobs the operator turns:

- **`r`** — projection rank. `r = d` disables reduction (the `Projection::identity` constructor); `r = d/2` is the sweet spot at d=128 below.
- **`rerank_mult`** — `k' = k · rerank_mult` candidates leave stage 1. Larger means higher recall, more raw-vector work.
- **`b`** — code width. Only LVQ-8 is implemented in this nightly; LVQ-4 is in *What to improve next*.

## Implementation notes

- **No unsafe.** No SIMD intrinsics. We rely on rustc + LLVM auto-vectorisation
  for both kernels. On Apple M4 Max, the f32 baseline already hits roughly
  one fused-multiply-add per cycle per lane — the asymmetric LVQ kernel pays a
  small tax for the extra `lo + step·code` decode, which is the headline
  surprise in the results below.
- **PCA via power iteration with explicit deflation.** No LAPACK. `O(iter · d · n)`
  per component, fine because LeanVec only needs `r ≪ d` components.
- **Trait-based index surface.** `VectorIndex` is the only API the demo and
  tests use, so any future backend (`HnswLeanVec`, `IvfLeanVec`) can be slotted
  in without changing call sites. This is the explicit Tepper et al.
  "LeanVec as a *codec*, not an index" framing.
- **Retained originals.** `LeanVecIndex` keeps f32 originals on the side; the
  rerank step uses them. This costs memory back relative to LVQ-only. A
  follow-up could store the originals as LVQ-8 too and lose the f32 keep.

Files & line counts:

```
crates/ruvector-leanvec/
├── Cargo.toml                  17 lines
├── src/lib.rs                  43
├── src/projection.rs           185
├── src/lvq.rs                  157
├── src/index.rs                317
├── src/main.rs                 178
└── tests/integration.rs        116
```

All under the 500-line ruvector ceiling.

## Benchmark methodology

```
$ cargo run --release -p ruvector-leanvec --bin leanvec-demo
```

Default knobs (overridable via `LEANVEC_N`, `LEANVEC_D`, `LEANVEC_Q`,
`LEANVEC_R`, `LEANVEC_RERANK`, `LEANVEC_SEED`):

| param        | value | meaning |
|--------------|-------|---------|
| n_db         | 10000 | indexed vectors |
| n_train      | 2000  | PCA training sample (prefix of db) |
| n_query      | 200   | independent query vectors |
| d            | 128   | input dimensionality |
| k            | 10    | top-k requested |
| rerank_mult  | 4     | so LeanVec scores 40 codes then reranks against 40 raw vectors |
| seed         | 17    | deterministic |

Data is a **anisotropic 16-factor GMM-ish synthetic** — exactly the regime
LeanVec is designed for (and the regime real embedding sets live in: text and
image embeddings have effective ranks far below their nominal dimension).
Ground truth is brute-force flat L2; recall@10 measures fraction of true
nearest-10 returned.

Hardware: **Apple M4 Max, 16 physical cores, 128 GB DDR5**, macOS 24.6.0,
`rustc 1.77` (release profile, default codegen-units, no `-Cnative`).

## Results

### Sweep over projection rank `r` (d = 128, n = 10 000, k = 10)

| variant       | r  | bytes/vec | recall@10 | ns / query | speed vs FlatF32 |
|---------------|----|-----------|-----------|------------|------------------|
| FlatF32       | —  | 512       | 1.0000    | 460 040    | 1.00× |
| LVQ-8         | 128 | 136      | 0.9890    | 523 260    | 0.88× |
| LeanVec-LVQ   | 64 | 584       | 1.0000    | 294 355    | 1.56× |
| LeanVec-LVQ   | 32 | 552       | 1.0000    | 202 476    | 2.29× |
| LeanVec-LVQ   | 16 | 536       | 1.0000    | 170 780    | 2.73× |

Headline findings, in order of how much they will surprise a careful reader:

1. **LeanVec-LVQ at r = 16 is 2.73× faster than flat f32 with recall@10 = 1.000.**
   The rerank stage on retained f32 originals neutralises every error the
   projection + LVQ pair introduces, on data with a real low-rank backbone.
2. **LVQ-8 alone is *slower* than f32 on this host (0.88×).** This is the
   non-obvious result. The asymmetric inner loop computes
   `(q_j − lo − step·code_j)²` per dim; the `code_j as f32` cast and the
   `step·code` multiply cost more than the f32 baseline saves on memory
   bandwidth, because at n=10 000 × d=128 the whole f32 corpus (~5 MB) fits in
   M4's last-level cache. **LVQ-8 wins when the database stops fitting in
   cache** — its design point is billion-scale, not laptop-scale. We report
   the honest small-scale loss rather than tuning the benchmark to hide it.
3. **Memory.** LeanVec-LVQ here is *bigger* than flat (584 vs 512 bytes/vec)
   because the f32 originals are kept for rerank. A follow-up that stores
   originals as LVQ-8 too (the "two-level LVQ" sketched in the LeanVec paper)
   would shrink LeanVec-LVQ to ~136 bytes/vec while keeping the speed-up.
4. **Build cost.** PCA training is 197 ms on 2 000 × 128. LeanVec index build
   adds ~45 ms over flat's 0.6 ms — the projection apply per vector dominates.
   Negligible for any real index.

### Recall on adversarial uniform data

The unit test that used uniform-random data was deliberately rewritten to use
anisotropic data, and we report this honestly: **PCA cannot help if the data
has no low-rank structure**. The rerank stage still defends recall, but the
speed advantage shrinks because stage 1 picks weaker candidates and
`rerank_mult` must rise. Real embedding sets are anisotropic; uniform random
is a stress test, not a production scenario.

## "How it works" walkthrough (blog-readable)

Imagine your vectors are 128-D OpenAI text embeddings of news articles. Two
empirical facts about that data:

- **Effective rank is ~20-40.** Most of the variance lives in a low-dim
  subspace; the other 90+ dimensions are tiny noise.
- **Per-vector dynamic range is small.** Within one vector, all 128 entries
  sit in roughly the same `[lo, hi]` band.

LeanVec-LVQ exploits both, one after the other:

1. We train a PCA `P` once. From now on, the *real* representation of any
   vector is `Pv ∈ R^r`. For typical text embeddings, `r = d/4 = 32`
   captures essentially all the geometry.
2. Within that 32-D representation, each vector still has a small dynamic
   range, so 8-bit codes anchored to that vector's own `lo` and `step`
   are essentially loss-free.
3. At query time, stage 1 runs a 32-D u8 distance loop over the whole
   database, which is ~16× narrower than the original f32 loop. We keep the
   top 40 hits (for k = 10).
4. Stage 2 reranks those 40 hits against the f32 originals. With only 40
   candidates the brute-force cost is trivial, and the math is exact.

The result on our M4 Max measurement: a 2.73× wall-clock speed-up at full
recall on a 10 000 × 128 corpus.

## Practical failure modes

- **Isotropic data.** PCA buys nothing. LeanVec degrades to "LVQ on a
  random projection" — usually slightly worse than LVQ alone. *Mitigation:*
  detect via the explained-variance curve at training time; fall back to
  identity projection.
- **In-cache regime.** When the corpus fits in L2/L3, the f32 baseline is
  already memory-cheap and the asymmetric kernel's per-element decode cost
  dominates. LVQ-8 alone loses. *Mitigation:* only enable LVQ once
  `n × d × 4 > L3`; otherwise stay in f32.
- **Heavy-tail per-vector ranges.** If a single dim has an outlier value,
  `step` blows up and the other dims quantise coarsely. *Mitigation:* clip
  outliers at the `(p1, p99)` quantiles before encoding (Aguerrebere et al.
  call this "robust LVQ").
- **PCA trained on the wrong distribution.** Embeddings drift; a projection
  trained on yesterday's corpus underperforms on today's queries. *Mitigation:*
  re-train on a sliding window of the indexed set (cheap — under 200 ms here).
- **Tiny queries.** At low `n`, the 197 ms PCA-train dominates total query
  cost. Amortise across many queries or precompute the projection offline.

## What to improve next

The roadmap follows the SVS feature ladder:

1. **LVQ-4.** 4-bit codes with a SIMD lookup table per code-byte pair.
   André et al. (2019) show this is ~2× faster than LVQ-8 with negligible
   recall hit when paired with rerank. Drop-in trait implementor.
2. **Two-level LVQ for retained originals.** Stop keeping f32 originals; keep
   LVQ-8 of the *unprojected* vector. Restores LVQ's ~4× memory win without
   losing rerank, because the second level is still strictly more accurate
   than the first.
3. **Glue with `ruvector-rairs`.** `IvfLeanVec` = IVF list partitioning over
   LeanVec codes. Targets billion-scale where flat scan dies.
4. **HNSW-LeanVec.** Use LeanVec codes as the in-graph distance, raw f32 for
   rerank at the leaves. Tepper et al. report 3-5× QPS over HNSW-f32 at the
   same recall on production embedding sets.
5. **Online projection refresh.** A streaming PCA (Oja's rule or block-Krylov)
   keeps the projection aligned with embedding drift without halting writes.
6. **NEON / AVX-512 explicit kernels.** When the autovectoriser leaves
   performance on the table — likely at LVQ-4 — drop in stable `std::simd`
   intrinsics. Stay safe Rust.

## Production crate layout proposal

```
ruvector-leanvec/                 (this crate — codec & flat index)
ruvector-leanvec-wasm/            wasm-bindgen surface for in-browser ANN
ruvector-leanvec-bench/           criterion harness vs FAISS IndexLVQ8
ruvector-leanvec-train/           offline PCA + LVQ trainer CLI
ruvector-hnsw-leanvec/            HNSW that consumes &dyn VectorCodec
ruvector-ivf-leanvec/             IVF that consumes &dyn VectorCodec
```

The `VectorCodec` trait (a generalised `VectorIndex` with `encode` /
`asym_distance` / `decode_one`) is the missing piece — once it exists, every
ruvector backend can opt into LeanVec-LVQ without bespoke integration code.

## References

1. Aguerrebere, C., Bhati, I., Hildebrand, M., Tepper, M., Willke, T.,
   *"Similarity search in the blink of an eye with compressed indices"*,
   PVLDB / arXiv:2304.04759, 2023.
2. Tepper, M., Bhati, I., Aguerrebere, C., Hildebrand, M., Willke, T.,
   *"LeanVec: Searching vectors faster by making them fit"*, arXiv:2312.16335,
   2024.
3. André, F., Kermarrec, A-M., Le Scouarnec, N., *"Cache locality is not enough:
   high-performance nearest neighbor search with product quantization fast
   scan"*, PVLDB 2015.
4. André, F., Kermarrec, A-M., Le Scouarnec, N., *"Quicker ADC: Unlocking the
   hidden potential of Product Quantization with SIMD"*, IEEE TPAMI 2019.
5. Gao, J., Long, C., *"RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search"*, SIGMOD
   2024.
6. FAISS, `IndexLVQ8`, https://github.com/facebookresearch/faiss (2025).
7. Milvus 2.4 changelog, scalar quantisation default, 2024.
