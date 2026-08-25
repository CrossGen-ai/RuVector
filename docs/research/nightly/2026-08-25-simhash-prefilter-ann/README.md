# SimHash Binary Prefilter for ANN Candidate Reduction

**Nightly research — 2026-08-25**
**Slug:** `simhash-prefilter-ann`
**Crate:** [`crates/ruvector-simhash-prefilter`](../../../../crates/ruvector-simhash-prefilter)
**ADR:** [ADR-340](../../../adr/ADR-340-simhash-binary-prefilter.md)

## Abstract

We evaluate a **signed random projection (SimHash) binary prefilter** as a
front-end candidate reducer for exact-rerank ANN. Each database vector is
encoded once into a 64/128/256-bit signature via a Rademacher projection
matrix; a query is sketched the same way and candidates are ranked by
`popcount(a ^ b)` before an exact float32 L2 rerank on the top-M. On a
20,000-vector, dim=128 clustered corpus the prefilter yields **100% recall@10
at 60 µs/query, an 11.5× speedup** over the 693 µs exact scan, with a
prefilter footprint of only **8 bytes/vector** (a 64× shrink vs. the raw
512-byte vector).

## SOTA survey

Binary sketching for nearest-neighbour search is a classical technique with
active recent developments:

- **Charikar, "Similarity Estimation Techniques from Rounding Algorithms"**
  (STOC 2002) — the SimHash construction: sign of a random projection is a
  1-bit LSH for angular similarity.
- **Achlioptas, "Database-friendly Random Projections"** (JCSS 2003) —
  Rademacher (±1) projections preserve distances as well as Gaussians,
  which is why our matrix is `i8` rather than `f32` — same guarantee, 4×
  smaller and integer-only in the sign accumulator.
- **Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search"**
  (SIGMOD 2024) — extends the SimHash idea to L2 by rotating first and
  proving a tight error bound. `ruvector-rabitq` already lives in tree
  (`crates/ruvector-rabitq/`); the present prefilter is complementary —
  it is a pre-index reducer, not an index in itself.
- **Milvus 2.4 & Weaviate 1.24 changelogs (2025)** — both shipped
  hardware popcount ("bitpacked BQ") code paths as a fast prefilter over
  HNSW. Qdrant's `BinaryQuantization` (v1.7) does the same for OpenAI's
  `text-embedding-3-*` at 32× compression.
- **FAISS `IndexBinaryFlat` / `IndexBinaryHNSW`** — long-standing baseline
  for binary-only workloads; produces a Hamming rank of the whole index.
- **André, Kermarrec, Le Scouarnec, "Cache locality is not enough:
  high-performance nearest neighbor search with product quantization
  fast scan"** (VLDB 2016) — establishes the pattern of "cheap scan then
  exact rerank" that motivates any prefilter.
- **Rust ecosystem:** `hnsw_rs` and `instant-distance` have no prefilter
  layer; the closest crate is `bitpacking` (SIMD popcount primitives).
  Nothing else in the ruvector workspace provides a pluggable pre-index
  Hamming reducer — the closest are `ruvector-rabitq` (a full quantized
  index) and `ruvector-visited-filter` (a per-query visited set).

## Proposed design

The prefilter is deliberately **not an index**. It is a `SketchFamily`
trait that any existing ANN index — HNSW, IVF, brute-force — can compose
in front of its exact-distance step.

```
        ┌────────────────────────────────────────┐
        │  query vector (f32; dim)               │
        └──────────────┬─────────────────────────┘
                       │  SketchFamily::sketch
                       ▼
                 [W×64 bit signature]
                       │
                       │  popcount(a ^ b) for all N stored sketches
                       ▼
             top-M by Hamming distance   (M = k × candidate_mult)
                       │
                       │  exact squared-L2 rerank on M
                       ▼
                     top-k
```

Key design decisions:

1. **`SketchFamily` trait, not a concrete type.** Enables plugging in
   OPQ-rotated SRP (like RaBitQ), learned sketches (Neural LSH), or
   super-bit LSH later without touching the index code.
2. **Const-generic `W` for word count.** `Sketch<1>` is 64-bit,
   `Sketch<2>` is 128-bit, `Sketch<4>` is 256-bit. All Hamming and
   popcount code is monomorphised at compile time.
3. **Rademacher, not Gaussian, projections.** Storage is `bits × dim`
   bytes (i8), the compiler auto-vectorises the sign-accumulate loop, and
   the encoder is deterministic given a seed.
4. **`#![forbid(unsafe_code)]`.** Portable popcount via `u64::count_ones`
   lowers to `popcntq` on x86_64 and `cnt.8b` on aarch64. No unsafe SIMD
   intrinsics required.
5. **Pluggable candidate multiplier.** Callers tune `M/k` per query
   class. High-recall queries widen the pool; latency-critical queries
   shrink it. This is the knob that survives production tuning.

## Implementation notes

Total library is ~350 LoC (`src/lib.rs`, well under the 500-line ceiling).
Public surface:

```rust
pub trait SketchFamily<const W: usize> { … }
pub struct SrpFamily<const W: usize> { … }
pub struct Sketch<const W: usize> { pub words: [u64; W] }
pub struct FlatPrefilterIndex<F: SketchFamily<W>, const W: usize> { … }
```

The `search_prefilter` method uses `select_nth_unstable_by_key` for both
the Hamming top-M pass and the L2 top-k rerank, keeping the hot path
allocation-bounded and O(N) rather than O(N log N).

## Benchmark methodology

- **Hardware:** Apple M-series host, macOS 26, `rustc 1.85` release
  profile.
- **Corpus:** 20,000 vectors, dim=128, mixture of 64 isotropic Gaussian
  clusters (σ=0.35). Chosen because pure isotropic noise is the *worst
  case* for any angular hash — real learned embeddings (BERT, CLIP,
  OpenAI `text-embedding-3-*`) have far more angular structure than
  isotropic Gaussian.
- **Queries:** 200 queries, each = base_point + N(0, 0.20²·I). Mirrors
  ANN-benchmarks practice: query distribution matches base distribution.
- **Metric:** squared L2, `k=10`.
- **Timing:** wall clock, `std::time::Instant`, warm cache. Each query
  timed independently (`p50` and `mean` reported).
- **Recall:** `recall@10` computed against the exact brute-force result.
- **Baseline:** the exact brute-force `search_exact` on the same index.
  No SIMD tricks — plain `sq_l2`, so all reported speedups are
  algorithmic, not micro-optimisation-driven.

Reproduce with:

```bash
cargo run --release -p ruvector-simhash-prefilter -- 20000 128 200 10
```

## Results

Real numbers from `cargo run --release`, JSON emitted on stdout:

| variant             | recall@10 | p50 µs | mean µs | sketch bytes total |
|---------------------|----------:|-------:|--------:|-------------------:|
| exact_bruteforce    |    1.0000 | 693.21 |  692.41 |                  0 |
| 64-bit,  mult=40    |    1.0000 |  57.75 |   59.80 |            160 000 |
| 128-bit, mult=40    |    1.0000 |  70.00 |   73.31 |            320 000 |
| 256-bit, mult=40    |    1.0000 |  79.92 |   83.93 |            640 000 |
| 128-bit, mult=10    |    0.6615 |  50.08 |   52.65 |            320 000 |
| 128-bit, mult=20    |    0.8875 |  58.33 |   59.25 |            320 000 |
| 128-bit, mult=80    |    1.0000 |  82.67 |   85.87 |            320 000 |

Raw corpus footprint: **9.77 MB** (20 000 × 128 × 4 B).
64-bit prefilter footprint: **156 KB** (a **64× shrink**).

### Observations

- **11.5× wall-clock speedup at 100% recall** with the smallest sketch
  (64-bit, mult=40). On this corpus the extra bits do not help — the
  angular structure is easy enough that 64 bits already separate top-10
  neighbours perfectly.
- **Bit-width matters at low candidate pools.** At mult=10 the 128-bit
  sketch loses one-third of the true top-10; at mult=20 it recovers to
  ~89%. Wider sketches would push mult=10 higher, but at the cost of
  encode time and sketch storage.
- **Candidate multiplier is the primary knob.** Recall goes from
  66% → 89% → 100% as mult goes 10 → 20 → 40 at fixed 128-bit width,
  while latency only grows from 50 → 58 → 70 µs.
- **The exact-scan baseline is not artificially handicapped** — it uses
  the same portable scalar `sq_l2`. A SIMD-vectorised exact scan would
  narrow the gap, but so would a SIMD-vectorised popcount.

## References

1. Charikar, M. "Similarity Estimation Techniques from Rounding
   Algorithms." STOC 2002.
2. Achlioptas, D. "Database-friendly Random Projections." JCSS 2003.
3. Gao, J. & Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with
   a Theoretical Error Bound for Approximate Nearest Neighbor Search."
   SIGMOD 2024.
4. Malkov, Y. A. & Yashunin, D. A. "Efficient and Robust Approximate
   Nearest Neighbor Search using HNSW graphs." IEEE TPAMI 2020.
5. André, F., Kermarrec, A-M., Le Scouarnec, N. "Cache locality is not
   enough: high-performance nearest neighbor search with PQ fast scan."
   VLDB 2016.
6. Milvus 2.4 release notes (Zilliz, 2025) — bit-packed prefilter code
   path.
7. Qdrant `BinaryQuantization` documentation (Qdrant Solutions, 2024).

## How it works — a walkthrough

Imagine 20,000 sentence embeddings sitting in RAM. Each is 128 floats,
so the corpus is 9.77 MB. A brute-force query — "compute L2 to every
one and keep the smallest ten" — is 20,000 × 128 subtract-square-add
operations plus a partial sort. On this machine that takes 693 µs. Fine
for one query, ruinous at 10k QPS.

We can spend a one-time cost — encoding each vector into a 64-bit
signature — to skip most of that work at query time. The signature is
computed by taking 64 random hyperplanes (each described by a `dim`-long
vector of ±1s) and setting bit *i* to 1 iff the vector lies on the
positive side of hyperplane *i*. Two vectors that point in similar
directions cross the same hyperplanes the same way; two random vectors
disagree on about half the bits.

At query time:

1. Sketch the query the same way.
2. XOR against every stored sketch and popcount — that's one `xor` and
   one `popcntq` per comparison, ~1 ns on modern hardware.
3. Take the 400 smallest Hamming distances (that's `k=10 × mult=40`).
4. Compute exact L2 to those 400. Sort. Return top-10.

Step 2 replaces 128 float ops with 1 XOR + 1 popcount. Step 4 is 400
exact distances instead of 20,000. Result: 60 µs instead of 693 µs, with
the same answer.

## Practical failure modes

- **Isotropic noise breaks angular hashing.** SimHash relies on angle,
  not L2, being informative. On our first pass with uniform-random
  queries the recall@10 was 12%, not because the code was wrong but
  because a random query has *no* meaningful nearest neighbour in a
  random cloud. Real embeddings do not suffer from this; a smoke test
  with random queries can nevertheless mislead.
- **Ties dominate the ranking at small sketch widths.** With 64 bits and
  a large corpus the number of pairs at the *same* Hamming distance
  balloons; `select_nth_unstable` breaks ties arbitrarily. Widen the
  sketch or the candidate pool.
- **Query encode cost is O(bits × dim).** At `dim=1536` (OpenAI
  ada-002 scale) and 256-bit sketches the encoder is 393k f32 ops per
  query. For very small corpora the encoder can outweigh the saved
  exact scan; the prefilter only wins above ~2k vectors.
- **The Rademacher matrix must persist.** Losing the seed means all
  prior sketches become garbage. Callers should serialise the seed
  alongside the sketch file.

## What to improve next — roadmap

- **OPQ-style pre-rotation.** Rotate the corpus once with a learned
  orthogonal matrix so more variance falls in the first coordinates,
  then sketch. This is the RaBitQ move and could lift recall at fixed
  sketch width by 10-20%.
- **Learned sketches.** Replace the Rademacher matrix with the output of
  a small MLP trained on labelled pairs (Neural LSH). Not obviously
  worth it for text embeddings, plausibly a win for image embeddings.
- **SIMD popcount over batches.** The scalar popcount is already fast;
  packing 4× u64 into `wide::u64x4` would double throughput on x86_64
  and aarch64.
- **Compose with HNSW.** Feed the prefilter's top-M into HNSW's
  `search_layer_0` as pre-visited nodes to reduce hops. This is where
  the biggest end-to-end wins likely live.
- **On-disk sketches with mmap.** 8-32 bytes/vector fits an enormous
  corpus in a memory-mapped file, giving cold-start latency near
  memory bandwidth.

## Production crate layout proposal

If this graduates from research to a first-class ruvector primitive:

```
crates/
├── ruvector-simhash-prefilter/         (this crate — algorithm)
├── ruvector-simhash-hnsw/              (HNSW wrapper that uses the prefilter)
├── ruvector-simhash-wasm/              (wasm32 bindings; drop rayon dep)
└── ruvector-sota-bench/                (add simhash to the existing sweep)
```

Public API stays as `SketchFamily<W>` + `Sketch<W>` + `FlatPrefilterIndex`;
downstream crates plug the trait into their own indices. The wasm crate
would gate `rayon` behind `#[cfg(not(target_arch = "wasm32"))]`, matching
the pattern already used by `ruvector-rabitq`.
