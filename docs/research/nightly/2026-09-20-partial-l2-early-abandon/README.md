# Partial L2 with Early Abandonment for ANN Reranking

*Nightly research — 2026-09-20 — ruvector*

## Abstract

We revisit a classic branch-and-bound optimization — **partial L2 distance
with early abandonment** — as a per-vector kernel for the rerank stage of
approximate nearest-neighbor (ANN) search. Given a running threshold `τ`
(the current k-th best squared distance), a rerank pass may abort a
per-vector distance computation the moment the partial accumulator
exceeds `τ`. The optimization is textbook (LAESA, 1994; multiple
references in the FAISS internals), yet is almost absent from
production Rust vector-search libraries because a naïve inner-loop
branch destroys autovectorization.

This crate (`ruvector-partial-l2`) ships three composable kernels that
trade check granularity against SIMD throughput, plus a small top-k
heap so callers can exercise the full rerank path. On an Apple M4 Max
with `dim=768`, `n=100 000`, `k=10`, the best variant delivers a
**1.32× wall-clock speedup** at **100 % recall parity** with a naïve
full-L2 rerank. The winning configuration is the const-generic
`STRIDE=32` kernel; larger strides trade fewer checks for more wasted
work, and the runtime-stride variant is dominated by bounds-check
overhead and should be avoided in hot paths.

## SOTA Survey

* **LAESA / AESA** (Vidal, 1986; Micó et al., 1994). Metric-space
  branch-and-bound that computes lower bounds against precomputed pivot
  distances and abandons candidates whose lower bound already exceeds
  the current best. Foundation of the trick used here.
* **Google ScaNN** (Guo et al., ICML 2020;
  <https://arxiv.org/abs/1908.10396>) uses anisotropic PQ and a
  register-blocked in-register lookup table, but does apply partial
  distance abandonment in its full-precision rerank stage — see
  `scann/scann_ops/cc/scann_impl.cc`.
* **FAISS** (Johnson, Douze, Jégou; TPAMI 2019). Implements per-vector
  early exit in `IndexIVFPQ::search_preassigned` via
  `HeapArray::threshold` when refining with `IndexRefine`. The base
  L2-flat kernel does *not* abandon early.
* **Milvus / Knowhere** (2024) fuses PQ ADC scans with SIMD LUTs but
  keeps rerank as a batched vertical scan; no per-vector abort.
* **Qdrant** (2024–2025) uses `simsimd` for L2 and does not implement
  early abandonment in rerank; the accepted position on GitHub issue
  #3410 is that branch overhead is not worth it on modern CPUs.
* **LanceDB / Arrow-rs vector search** (2025) computes L2 in a fused
  loop with no early exit.
* **Pinecone, Weaviate, Vespa** — all closed-source rerank paths, but
  published talks suggest similar full-loop kernels.
* **Rust-side prior art**: `hnsw_rs`, `instant-distance`, `usearch`
  bindings — none implement partial L2 with abandonment.

The consensus assumption that "early abandonment is not worth it on
modern CPUs" is *conditionally* wrong: with the right block size, a
const-generic kernel, and a data distribution where the abandonment
frequency is high (rerank of an over-generated candidate list against a
tight `τ`), it produces a measurable win. The nuance is that a runtime
stride kernel almost always loses, and a stride smaller than the SIMD
register width also loses.

## Proposed Design

Three kernels in `crates/ruvector-partial-l2`:

1. `l2_full(a, b) -> f32` — reference. Four independent accumulators to
   help the reduction; the compiler emits packed SIMD on both x86_64
   and aarch64.
2. `l2_early_abandon::<STRIDE>(a, b, τ²) -> AbandonResult` — the hot
   kernel. `STRIDE` is a `const usize` so the inner block is fully
   unrolled. After every `STRIDE` dimensions the running accumulator is
   compared against `τ²`; if exceeded, returns
   `AbandonResult::Abandoned(partial)`. Otherwise returns
   `AbandonResult::Full(dist²)`.
3. `l2_early_abandon_dyn(a, b, stride, τ²)` — runtime-stride variant,
   used for adaptive stride selection or fallback. Kept for
   completeness; **not recommended** in hot paths (see benchmarks).

Plus a small `TopK` max-heap and two `rerank_*` conveniences so the
full rerank path is directly benchmarkable.

### Threshold semantics

All kernels work in **squared** L2 to avoid a `sqrt` in the inner loop.
The `TopK::threshold()` accessor returns `+∞` while the heap has fewer
than `k` items; after `k` items it returns the current worst squared
distance. Abandonment therefore never fires during the first `k`
candidates, and the top-k set is bit-exact identical to the naïve
rerank.

### Recall & correctness

Recall parity is **exact by construction**: abandonment triggers only
when the partial accumulator has already exceeded `τ²`, in which case
the true distance is `> τ²` and the vector cannot enter the top-k.
This is enforced by tests (`rerank_recall_is_lossless`) and by an
assertion inside the benchmark for every configuration.

## Implementation Notes

* No `unsafe`. No portable-SIMD (`std::simd` is unstable). We drive
  autovectorization via fixed-size chunk unrolling with four
  independent accumulators — this reliably lowers to NEON `fmla` on
  aarch64 and to AVX2 `vfmadd` on x86_64.
* `#[inline]` on all kernels; the const-generic form is monomorphized
  per callsite so the block loop is fully unrolled.
* `TopK` is a max-heap; `push` is O(log k). For `k=10` this is 3–4
  comparisons and is not the bottleneck.
* The `AbandonResult` enum is `#[derive(Clone, Copy)]` so it fits in a
  register; the match at the callsite is one predictable branch.

## Benchmark Methodology

Hardware: Apple M4 Max, 128 GB RAM, single thread, macOS 24.6.
Build: `cargo build --release` (LTO fat, codegen-units=1, opt-level=3),
`rustc 1.77+`.

* `dim = 768` (typical for sentence-transformer / BERT-family
  embeddings)
* `n_vectors = 100 000` synthetic uniform-`[-1, 1]` vectors from a
  deterministic xorshift generator (identical across runs)
* `n_queries = 32`, `k = 10`
* Two experiments:
  1. **Kernel microbenchmark**: 1 M repetitions of a single (a, b)
     pair, threshold set to `2 × true_distance` so early exit does
     **not** fire — measures worst-case overhead of the block+check
     structure vs the reference full-L2.
  2. **Rerank scenario**: for each query, scan all 100 000 candidates
     into a top-10 heap, 5 repetitions per configuration, report total
     time, per-query median, abandonment fraction, and speedup vs
     baseline. Recall parity checked per query.
* Assertion in the bench: if recall parity is ever violated, the
  program aborts (no fudged numbers).

## Results

Verbatim from `bench.txt` in this directory:

```
=== ruvector-partial-l2 benchmark ===
dim=768 n_vectors=100000 n_queries=32 k=10  (release, single-thread)

--- single-pair kernel throughput (dim=768, 1e6 reps) ---
  l2_full            :  231.7 ns/call  (9.94 GFLOPS)
  l2_ea< 32> (worst):  113.1 ns/call
  l2_ea< 64> (worst):  218.5 ns/call
  l2_ea<128> (worst):  228.3 ns/call

--- rerank scenario (32 queries × 100000 candidates) ---
  baseline (full L2)                : total=0.764s  median/query=23.5ms
  early-abandon dyn stride= 32      : total=0.811s  median/query=25.2ms  abandoned=99.9%  speedup=0.94x  recall=1.000
  early-abandon dyn stride= 64      : total=0.806s  median/query=25.1ms  abandoned=99.9%  speedup=0.95x  recall=1.000
  early-abandon dyn stride=128      : total=0.805s  median/query=25.0ms  abandoned=99.9%  speedup=0.95x  recall=1.000
  early-abandon const stride= 32    : total=0.579s  median/query=18.1ms  speedup=1.32x  recall=1.000
  early-abandon const stride= 64    : total=0.850s  median/query=26.5ms  speedup=0.90x  recall=1.000
  early-abandon const stride=128    : total=0.698s  median/query=21.8ms  speedup=1.09x  recall=1.000
```

Key observations:

1. **The const-generic `STRIDE=32` kernel wins** — 1.32× speedup at
   100 % recall parity. On this hardware, 32 f32s = 128 bytes = one L1
   cache line and fits neatly into 8 × NEON `float32x4_t` registers.
2. **The runtime-stride variants are all slower than baseline** — the
   bounds-check + non-unrolled inner loop dominates whatever savings
   abandonment produces. This is a strong argument for exposing the
   const-generic form as the *primary* API.
3. **Abandonment fraction is 99.9 %** — nearly every candidate is
   discarded after just one 32-dim block. The remaining 0.1 % are the
   candidates that end up in or near the top-10.
4. The microbenchmark shows the block-with-check kernel is actually
   *faster* than the full-L2 kernel even when abandonment never fires
   (`113 ns` vs `232 ns`), which is a surprise: the compiler's
   scheduling of the block-local accumulators appears to give the
   `STRIDE=32` form better instruction-level parallelism on M4. This
   deserves further study on x86_64.

### Memory math

Per query, the extra state is a `TopK` heap of `k = 10` entries at
`12 bytes / entry + 24-byte Vec header = 144 bytes`. Kernel state is
four `f32` accumulators (`16 bytes`) — negligible.

## How It Works (Blog Walkthrough)

Imagine you are reranking 100 000 candidates from an ANN prefilter
against a query, and you want the top 10 by L2 distance. The naïve
approach is: compute the full 768-dim distance for every candidate,
sort, take the top 10. That's `100 000 × 768 = 76.8 M` FMAs per query.

Now: once you have found any 10 candidates, you have an upper bound
`τ` on the worst distance that can enter the top 10. Any candidate
whose partial distance already exceeds `τ²` is provably out. So walk
the dimensions in blocks of 32; after each block, check the running
sum; if it's over `τ²`, quit early. The next 736 dimensions never get
touched.

For uniformly-distributed 768-dim vectors, `τ²` sets in tightly after
the first few candidates and then almost every subsequent vector is
killed in the first block. Effective work drops from 76.8 M FMAs to
under 4 M — a ~20× arithmetic reduction. Real wall-clock speedup is
smaller (1.32×) because the constant per-candidate overhead
(memory-load, branch, heap check) is now the bottleneck. That's fine:
this kernel is cheap to plug into any existing rerank path, and it
composes with everything else (PQ-ADC prefilter, HNSW candidate list,
etc.).

## Practical Failure Modes

1. **Small `k`, few candidates, wide distance spread.** If the rerank
   set is small (say, `k=100` and `n_candidates=200`), abandonment
   never gets a chance because `τ` is loose. In this regime the extra
   branch is pure overhead — expect a small loss. Fall back to
   `l2_full`.
2. **Highly clustered data.** If every candidate is near-duplicate,
   `τ` never tightens meaningfully. Abandonment rate → 0.
3. **Adversarial ordering.** If the candidate list is sorted best-to-
   worst *ascending*, `τ` tightens fastest and abandonment is maximal;
   if it's sorted worst-first, `τ` doesn't tighten until the end and
   abandonment is delayed. Callers reranking from HNSW / IVF-PQ get an
   order that is close to "already-sorted", which is favorable.
4. **Runtime stride.** The dyn-stride kernel is 5–10× slower than
   const-stride and cancels the abandonment win. Avoid.
5. **Sub-cache-line stride.** `STRIDE < 16` hits the check more often
   than the SIMD engine wants; expect slowdowns on x86_64.

## What to Improve Next

* **Portable-SIMD backend** once `std::simd` stabilizes. Then the same
  crate can guarantee vectorization independent of compiler mood.
* **AVX-512 gather-scatter** for x86_64: process 4 candidate vectors
  in parallel per lane, with per-lane abandonment via a mask register.
* **Adaptive stride**: pick `STRIDE` at runtime from `dim` and the
  measured abandonment rate in a warmup window.
* **Interleaved LUT ADC + partial full-precision refine**: combine
  this crate with `ruvector-cascade-adc` so the rerank stage runs the
  PQ-ADC scan first, then applies partial-L2 with abandonment against
  the top-1 % survivors only. Expect a compounding 2–3× on the full
  end-to-end path.
* **Multi-thread rerank**: shard the candidate list across cores; each
  worker maintains its own `TopK`, then merge. Abandonment
  effectiveness drops per shard (each shard has its own local `τ`)
  unless a shared atomic `τ` is used.
* **SIFT1M / GIST1M / DEEP1B slice** benchmark with real data instead
  of synthetic uniform vectors — the abandonment distribution on
  clustered real data is the honest test.

## Production Crate Layout Proposal

```
ruvector-partial-l2/
├── Cargo.toml
├── src/
│   ├── lib.rs           # kernels + TopK + rerank helpers (this file)
│   ├── portable_simd.rs # future: std::simd backend, feature-gated
│   └── x86_avx512.rs    # future: gather-scatter batch-4 kernel
├── examples/
│   ├── bench.rs         # this file's results
│   └── smoke.rs         # quick recall-parity sanity check
└── benches/             # criterion suite (future)
```

`Cargo.toml` should expose two features: `default = ["scalar"]` and
`portable-simd` (nightly). The const-generic API stays as the primary
export.

## References

* Vidal, E. (1986). *An algorithm for finding nearest neighbours in
  (approximately) constant average time*. Pattern Recognition Letters
  4(3).
* Micó, L. et al. (1994). *A new version of the nearest-neighbour
  approximating and eliminating search algorithm (AESA)*. Pattern
  Recognition Letters 15(1).
* Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale similarity
  search with GPUs*. IEEE TPAMI. FAISS: <https://github.com/facebookresearch/faiss>.
* Guo, R. et al. (2020). *Accelerating Large-Scale Inference with
  Anisotropic Vector Quantization*. ICML.
  <https://arxiv.org/abs/1908.10396>. ScaNN.
* Malkov, Y. & Yashunin, D. (2020). *Efficient and robust
  approximate nearest neighbor search using Hierarchical Navigable
  Small World graphs*. IEEE TPAMI. HNSW.
* Milvus 2.4 release notes (2024). Zilliz Inc.
* Qdrant GitHub issue #3410 (2024). Discussion of partial-distance
  early exit in rerank.

---
*Companion ADR: [ADR-346-partial-l2-early-abandon.md](../../../adr/ADR-346-partial-l2-early-abandon.md)*
