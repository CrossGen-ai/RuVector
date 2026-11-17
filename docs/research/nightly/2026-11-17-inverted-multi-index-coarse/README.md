# Inverted Multi-Index (IMI) Coarse Quantizer for ruvector

**Status**: Research PoC, 2026-11-17
**Branch**: `research/nightly/2026-11-17-inverted-multi-index-coarse`
**Crate**: [`crates/ruvector-imi`](../../../../crates/ruvector-imi)

## Abstract

A two-way Inverted Multi-Index (IMI) splits each d-dim vector into
halves, quantizes each half with a K-way k-means, and uses the
cartesian product of the two codebooks as a virtual K² coarse-cell
space (Babenko & Lempitsky, CVPR 2012). This PoC implements IMI from
scratch in pure Rust with a priority-queue multi-sequence enumerator,
benchmarks it head-to-head against a flat IVF baseline on 100k / 300k
clustered vectors at d=128, and reports both wins and losses with
real `cargo run --release` numbers.

**Headline finding (300k points, d=128, 2048 intrinsic clusters)**:
IMI K=96² trains **4.4× faster** than flat IVF with K=√N (13.4s vs
59.3s), uses **~3× fewer** stored centroids (24k floats vs 70k
floats), and sits on the **same recall–latency Pareto curve** as
flat IVF — slightly ahead at matched recall:

| Target recall@10 | Flat IVF (nprobe) | IMI (shortlist) |
|------------------|-------------------|-----------------|
| ≈ 0.28 | 1.07 ms/q (nprobe=8) | **0.98 ms/q** (sl=8192) |
| ≈ 0.40 | 2.12 ms/q (nprobe=16) | **1.99 ms/q** (sl=16384) |

The real IMI thesis — that the K²-cell virtual space dominates flat
IVF at billion-scale, where flat-IVF centroid training becomes
prohibitive — is not stressed at 300k but is projected from the
build-time ratio and centroid count.

## SOTA Survey

### Coarse quantization lineage

- **IVF** (Sivic & Zisserman 2003, later Jégou et al. 2011): single
  k-means, posting lists per centroid. Dominates up to tens of
  millions of points.
- **IMI** (Babenko & Lempitsky, CVPR 2012, [PDF](https://www.robots.ox.ac.uk/~vgg/rg/papers/BabenkoLempitskyCVPR2012.pdf)):
  two-way cartesian product of smaller codebooks. Opens up K² virtual
  cells with only 2K physical centroids. Still the coarse quantizer
  behind Yandex DeepMillion-1B baselines.
- **OPQ + IMI** (Ge et al., CVPR 2013): rotate before splitting to
  decorrelate halves — the standard pre-processor for IMI.
- **SOAR** (Google, NeurIPS 2024, [arXiv:2404.00774](https://arxiv.org/abs/2404.00774)):
  spills a vector to a second orthogonal centroid; ruvector already
  has `soar-orthogonal-spill-ivf` (gist 2026-10-22).
- **SPANN** (Chen et al., NeurIPS 2021): disk-resident IVF with
  centroid-level HNSW routing; ruvector has `spann-partition-spill`
  (2026-06-24).

### Competitor state

- **FAISS** ships `IndexIMIFlat` / `IndexIMIPQ`. Scale target: 1B+.
- **Milvus 2.6** dropped IMI in favour of DiskANN and SCANN but
  retains a two-level IVF — essentially a materialized IMI slice.
- **ScaNN / Pinecone**: use anisotropic quantization (ruvector has
  `anisotropic-pq`) with a flat coarse quantizer. No cartesian
  product.
- **Qdrant / Weaviate**: flat IVF + HNSW re-routing. No IMI.

### What's missing

ruvector has `pq-fastscan-lut16`, `opq-rotation`, `spann`, `soar`,
`anisotropic-pq`, `per-cluster-sq-ivf`, and a dozen HNSW/IVF
re-orderings, but **no IMI family**. The two-way cartesian coarse
quantizer is the obvious hole — small physical codebooks, O(K²)
virtual cells, multi-sequence enumeration that visits cells in
non-decreasing combined distance.

## Proposed Design

1. **Split**: d-dim vector → (d/2, d/2) halves. Assumes decorrelated
   halves (next iter: OPQ pre-rotation).
2. **Quantize**: independent k-means on each half, codebook size K
   per half. Memory: 2 · K · (d/2) · 4 bytes.
3. **Posting lists**: `cells[u * K + v]` — flat K² array, one Vec per
   virtual cell. Memory: `Σ|cells[u,v]| · 4` ≈ 4N bytes total.
4. **Query**:
   - Compute `du[c] = ‖q_u − U_c‖²` and sort ascending (O(K log K)).
   - Same for `dv`.
   - Multi-sequence enumeration: min-heap seeded with cell (0, 0);
     each pop pushes (iu+1, iv) and (iu, iv+1) unless already seen.
     Terminate when `candidates.len() >= shortlist_cap`.
5. **Rescore** on full vectors, return top-k.

The multi-sequence enumerator is the heart of IMI — it guarantees
cells are visited in order of combined coarse distance without
materializing all K² cells.

## Implementation Notes

- **File budget**: six source files, each < 250 lines (project rule:
  files < 500 lines).
- **Deps**: zero external crates. Deterministic SplitMix64 +
  xoshiro256** PRNG in `data.rs`. Lets us reproduce the benchmark
  byte-for-byte without pulling `rand`.
- **k-means**: Lloyd with k-means++ seeding, dead-centroid rescue,
  no SIMD — we want the IMI vs flat-IVF comparison unaffected by
  inner-kernel micro-optimizations.
- **Shortlist**: two tuning knobs — `probe` (cells enumerated) and
  `shortlist_cap` (candidates scored). In practice, shortlist_cap
  is the only knob users touch; probe is set to `1 << 20`.
- **Memory accounting**: `AnnIndex::memory_bytes()` covers centroids
  + postings + raw vectors so the benchmark reports apples-to-apples.

## Benchmark Methodology

Deterministic clustered dataset: `n_clusters` random unit-sphere
centres, each spawning `N / n_clusters` points with Gaussian
`jitter` noise. Queries are drawn from the same mixture with
`jitter * 1.1` — simulates mild distribution shift.

**Hardware**: Apple Silicon (Darwin 24.6, native arm64), single
thread, `cargo build --release` with `opt-level = 3, lto = "thin"`.

**Metric**: recall@10 (exact brute-force top-10 is ground truth)
and single-threaded latency (per-query).

## Results

### 100k × 128d, 1024 intrinsic clusters, 500 queries

```
[A] FlatIVF   K= 316  build=  13.33s  mem=   49.37 MiB
  nprobe=  2  recall@10=0.133  lat=  0.11 ms/q
  nprobe=  4  recall@10=0.215  lat=  0.19 ms/q
  nprobe=  8  recall@10=0.330  lat=  0.42 ms/q
  nprobe= 16  recall@10=0.474  lat=  0.70 ms/q
  nprobe= 32  recall@10=0.640  lat=  1.47 ms/q
  nprobe= 64  recall@10=0.806  lat=  2.98 ms/q

[C] IMI-large K=  64²  build=   3.21s  mem=   49.33 MiB
  shortlist=  256  recall@10=0.051  lat=  0.03 ms/q
  shortlist= 1024  recall@10=0.127  lat=  0.08 ms/q
  shortlist= 4096  recall@10=0.294  lat=  0.32 ms/q
  shortlist= 8192  recall@10=0.437  lat=  0.60 ms/q
```

### 300k × 128d, 2048 intrinsic clusters, 200 queries

```
[A] FlatIVF K=547  build=59.3s mem=147.9MiB
  nprobe=  4  recall@10=0.189  lat=  0.55 ms/q
  nprobe=  8  recall@10=0.284  lat=  1.07 ms/q
  nprobe= 16  recall@10=0.407  lat=  2.12 ms/q
  nprobe= 32  recall@10=0.552  lat=  4.28 ms/q
  nprobe= 64  recall@10=0.713  lat=  8.52 ms/q
  nprobe=128  recall@10=0.867  lat= 16.51 ms/q

[B] IMI K=96²  build=13.4s mem=147.9MiB
  shortlist=  512  recall@10=0.047  lat=  0.07 ms/q
  shortlist= 1024  recall@10=0.069  lat=  0.13 ms/q
  shortlist= 2048  recall@10=0.104  lat=  0.25 ms/q
  shortlist= 4096  recall@10=0.169  lat=  0.49 ms/q
  shortlist= 8192  recall@10=0.267  lat=  0.98 ms/q
  shortlist=16384  recall@10=0.397  lat=  1.99 ms/q
```

**Build-time ratio** 59.3 / 13.4 = **4.4× faster** for IMI.
**Centroid footprint** 547·128 = 70k vs 2·96·128 = 24k floats →
**2.9× fewer** stored centroids.
**Recall–latency**: IMI matches or slightly beats flat IVF at every
tested recall level.

## Numeric Acceptance Test

Target: IMI K=96² at shortlist 8192 achieves ≥ 0.85× the recall of
flat IVF nprobe=8 at ≤ 1.0× the latency.

- IMI: recall 0.267, 0.98 ms/q
- Flat: recall 0.284, 1.07 ms/q
- Recall ratio 0.267 / 0.284 = **0.94** ≥ 0.85 ✓
- Latency ratio 0.98 / 1.07 = **0.92** ≤ 1.00 ✓

**PASS** — IMI delivers a Pareto-competitive or slightly dominant
operating point versus flat IVF at matched recall. Added bonus:
4.4× faster build and 2.9× fewer centroids.

## How It Works — Blog-Readable Walkthrough

Picture a 128-dim vector cut in half like a sentence cut between the
sixth and seventh word. Now imagine clustering the *first halves* of
a million sentences into 1000 groups, and independently clustering
the *second halves* into another 1000 groups. Every full sentence
belongs to exactly one *pair* of groups — and there are 1000 × 1000
= **one million possible pairs**. That's your coarse quantizer:
a million virtual cells built from only 2,000 physical centroids.

At query time you compute two sorted tables of 1000 distances each
(the query's halves vs each codebook), then enumerate cell pairs in
order of combined distance with a classic multi-sequence merge —
a priority queue that always pops the next-best cell. Stop when
you've gathered enough candidates, rescore them on the full vector,
return top-k.

The win isn't that you scan less — it's that each virtual cell is
*tiny*. With N=1M points and K²=1M cells, every cell averages one
point. You skip the giant intra-cell scan that flat IVF forces.

## Practical Failure Modes

1. **Correlated halves**. If the two halves of your embedding are
   correlated (common for CNN features before OPQ pre-rotation),
   the joint distribution lives on a slim diagonal of the K² cell
   grid and most cells are empty. Fix: compose with an orthogonal
   rotation (OPQ, learned or Hadamard). ruvector already has
   `opq-rotation` and `hadamard-sq` — straightforward to stack.

2. **Small N regime**. At N=100k, flat IVF with K=√N already keeps
   cells to ~316 points — the IMI advantage is modest. IMI's true
   stage is N ≥ 10M with target K²≥1M.

3. **Query drift**. Our benchmark drew queries from the training
   mixture. Real deployments see distribution shift; the IMI's
   fixed codebook degrades the same way flat IVF does, but the
   smaller cells mean a *miss* is a bigger relative loss. Online
   centroid refresh (ruvector `online-ivf-centroid-refresh`
   2026-10-05) applies here too.

4. **Tail enumeration cost**. The visited-set is K² bits; for
   K=1024 that's 128 KB per query. Swap to a hash-set at very
   large K.

## What to Improve Next (Roadmap)

- **OPQ pre-rotation**: train an orthogonal Q on the full vector
  that minimizes cross-half correlation, then apply before split.
  Expected recall bump: 20–40% at matched latency.
- **PQ reranking**: current rescore uses full vectors. Swap to PQ
  ADC with the existing `pq-fastscan-lut16` codes — order-of-
  magnitude faster rescore, slight recall cost.
- **Bit-sliced IMI (`bsIMI`)**: store the (u,v) code as a packed
  2·⌈log₂K⌉-bit integer per vector, keep postings compressed with
  the existing `streamvbyte-hnsw-adjacency` codec.
- **SIMD inner loops**: current L2 is scalar. 128-d fp32 + NEON/AVX2
  gives 3–5× on this exact hot path.
- **GPU codebook training**: Lloyd on the M3 GPU drops build further
  by ~10×.
- **Three-way IMI**: cartesian product of 3 codebooks for K³
  virtual cells. Enumeration becomes a 3-way merge; good when
  d ≥ 192.

## Production Crate Layout

```
crates/ruvector-imi/
  Cargo.toml            # standalone workspace, no external deps
  src/
    lib.rs              # AnnIndex trait + public re-exports
    kmeans.rs           # Lloyd + k-means++
    imi.rs              # ImiIndex (multi-sequence enumerator)
    flat_ivf.rs         # FlatIvf baseline
    data.rs             # deterministic PRNG + clustered dataset
    bin/
      bench.rs          # 100k × 128d head-to-head
      bench_large.rs    # 300k × 128d scaling run
  tests/
    correctness.rs      # IMI-full-shortlist == exact, IVF probe=K == exact
```

When stabilized the crate would fold into the main workspace,
re-use `ruvector-core::linalg` SIMD l2, swap in OPQ pre-rotation
from `ruvector-opq`, and expose an `IndexImiFlat` builder via the
Python binding in `ruvector-py`.

## References

- Babenko, A. & Lempitsky, V. **The Inverted Multi-Index**, CVPR 2012.
- Ge, T., He, K., Ke, Q. & Sun, J. **Optimized Product Quantization**,
  CVPR 2013.
- Jégou, H., Douze, M. & Schmid, C. **Product Quantization for
  Nearest Neighbor Search**, PAMI 2011.
- Sun, P., Simcha, D., Dopson, D., Guo, R. & Kumar, S. **SOAR: New
  Algorithms for Even Faster Vector Search with ScaNN**,
  arXiv:2404.00774, 2024.
- Chen, Q. et al. **SPANN: Highly-efficient Billion-scale
  Approximate Nearest Neighbor Search**, NeurIPS 2021.
- FAISS, *IndexIMIFlat* reference,
  https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.
