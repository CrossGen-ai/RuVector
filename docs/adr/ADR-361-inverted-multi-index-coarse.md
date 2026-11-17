# ADR-361: Inverted Multi-Index (IMI) Coarse Quantizer

- **Date**: 2026-11-17
- **Status**: Proposed (nightly research PoC)
- **Branch**: `research/nightly/2026-11-17-inverted-multi-index-coarse`
- **Crate**: `crates/ruvector-imi`

## Context

ruvector's existing coarse-quantizer story covers single-codebook IVF
(ADR-era crates: `ruvector-spann`, `ruvector-sh-ivf`,
`ruvector-ivf-progressive-shortlist`), per-cluster SQ, OPQ rotation,
SOAR orthogonal spill, and various posting-list codecs. All share one
weakness: the number of coarse cells is capped by the k-means we can
afford to train. At N = 10⁹ points a target K = 10⁶ cells means a
10⁶-way k-means — hours of training and a 512 MB centroid table at
d = 128.

The Inverted Multi-Index (Babenko & Lempitsky, CVPR 2012) sidesteps
this with a **cartesian product coarse quantizer**: split each vector
into two halves, run a small k-means on each half (K physical
centroids per half), and let the virtual cell space be K². A 1024²
IMI gives > 10⁶ virtual cells with only 2048 physical centroids.
Classic result, well-studied, still the coarse quantizer behind
Yandex's DeepMillion-1B baselines, but absent from ruvector.

## Decision

Add `crates/ruvector-imi` — a standalone Rust crate implementing:

1. A Lloyd k-means with k-means++ seeding (`kmeans.rs`).
2. `ImiIndex` with per-half codebook training, flat K² posting
   layout, and a priority-queue multi-sequence enumerator
   (`imi.rs`).
3. A `FlatIvf` baseline sharing the same `AnnIndex` trait
   (`flat_ivf.rs`) so apples-to-apples benchmarks are possible.
4. Deterministic synthetic benchmarks at N = 100k and N = 300k
   (`bin/bench.rs`, `bin/bench_large.rs`).

**Zero external dependencies**: an internal SplitMix64 +
xoshiro256** PRNG and a clustered dataset generator keep the crate
reproducible byte-for-byte.

## Consequences

**Pros**
- O(K²) virtual cells with O(2K) physical centroids — opens
  billion-scale coarse quantization without a billion-way k-means.
- **4.4× faster build** than flat IVF with K = √N at N = 300k
  (13.4s vs 59.3s, single-thread, measured).
- Sits on or slightly above the flat-IVF recall–latency Pareto
  frontier at matched recall (IMI 0.98 ms @ 0.267 vs Flat 1.07 ms
  @ 0.284 recall@10).
- Clean trait-based design — can later stack OPQ pre-rotation,
  swap scalar l2 for SIMD, or re-use PQ reranking from the
  existing `pq-fastscan-lut16` crate.

**Cons / costs**
- Assumes the two halves of the vector are decorrelated. CNN /
  Sentence-Transformers embeddings aren't — needs OPQ pre-rotation
  to realize the paper's recall. Next-iter work.
- Multi-sequence enumeration adds a `BinaryHeap<Cell>` per query
  (~K entries max). At K = 10⁴ this costs ~100 µs; dwarfed by
  rescore at the same scale, but noticeable at small N.
- Visited-set memory is K² bits; swap to hash-set at K ≥ 1024.

**Risk mitigation**
- Correctness tests: full-shortlist IMI must equal exact NN
  (passes in `tests/correctness.rs`), flat IVF with probe = K must
  equal exact NN (passes), posting lists must partition the
  dataset (passes).

## Alternatives Considered

1. **Skip IMI; extend SPANN with larger K.** Rejected: SPANN is
   disk-resident and solves a different problem (memory-bounded
   billion-scale); the cost of flat k-means at K = 10⁶ is still
   the bottleneck.
2. **Three-way IMI directly.** Rejected for the PoC: validates the
   two-way case first, revisits when d ≥ 192 bench shows two-way
   plateau.
3. **IMI with product quantization (PQ) in the halves.** Deferred:
   stacks naturally with existing `pq-fastscan-lut16`; better as
   a follow-up once two-way IMI lands.

## Acceptance Criteria

- `cargo build --release -p ruvector-imi` succeeds. ✓
- `cargo test --release -p ruvector-imi` passes with the three
  correctness tests. ✓
- `./target/release/imi-bench` produces real numbers (see
  `docs/research/nightly/2026-11-17-inverted-multi-index-coarse/README.md`). ✓
- Numeric acceptance: IMI K=96² sl=8192 at ≥ 0.85× flat-IVF
  nprobe=8 recall and ≤ 1.0× latency. **0.94× recall, 0.92×
  latency** — PASS.

## Follow-ups

- Stack `ruvector-opq` pre-rotation to decorrelate halves.
- Swap scalar l2 for the SIMD kernel in `ruvector-core::linalg`.
- Add PQ ADC rescore path.
- Three-way IMI for d ≥ 192.
