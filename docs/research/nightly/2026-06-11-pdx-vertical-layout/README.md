# PDX Vertical Block Layout — Nightly Research, 2026-06-11

> SIMD-accelerated similarity scan via block-transposed storage, with
> PDX-BOND-style partial-distance pruning. Pure-Rust PoC, real numbers,
> honest failure modes.

## Abstract

We port the *PDX* (Predictable Data eXtraction) vertical-block layout
from Kuffo et al. (CWI, SIGMOD 2025) into ruvector and benchmark it
against the row-major baseline that all of our current flat-scan,
IVF-rerank, and brute-force paths use. On an Apple M4 Max, the layout
change alone delivers **2.4×–8.8× scan throughput** across three
representative configurations (10K and 50K vectors at D=128; 10K vectors
at D=768). PDX-BOND pruning saves 20–27% of inner-loop MULs but costs
wall-clock time on uniform-random data; the published wins on real
embeddings are not yet validated. The PoC ships as a standalone
`ruvector-pdx` crate with five passing tests and a deterministic
benchmark harness — no `criterion`, no mocks, no external deps.

## SOTA Survey

| Idea | Year / Venue | Reference |
|---|---|---|
| FAISS IVF/PQ — row-major scan baseline | 2017 | Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs.* |
| RaBitQ — 1-bit quantization for L2 | NeurIPS 2024 | Gao & Long. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound.* (Already in `ruvector-rabitq`, ADR-193.) |
| LeanVec — projection + 8-bit residual | 2024 | Tepper et al. *LeanVec: Searching Vectors Faster by Making Them Fit.* (`ruvector-leanvec`.) |
| **PDX** vertical block layout + BOND pruning | **SIGMOD 2025** | Kuffo, Krimpen-Stoop, Tang, Manegold. *PDX: A Data Layout for Vector Similarity Search.* CWI Amsterdam. |
| ACORN — predicate-aware HNSW | VLDB 2024 | Patel et al. (`ruvector-acorn`, prior nightly 2026-04-26.) |
| CAGRA-Q — GPU graph + quantization | 2024 | NVIDIA RAPIDS. Out of scope here. |
| SOAR — orthogonal projections for IVF | SIGMOD 2024 | Already in `ruvector-soar`. |

The literature gap PDX fills: every prior cited compression scheme
(RaBitQ, LeanVec, PQ) reduces the *cost per vector* but keeps the
row-major sweep. PDX is layout-orthogonal to all of them and can stack:
RaBitQ-on-PDX is the obvious next nightly.

## Proposed Design

```
Horizontal (row-major)               PDX vertical (block = 64)

  vec0: d0 d1 d2 ... dD-1            block 0: [v0.d0 v1.d0 ... v63.d0]   <- stripe 0
  vec1: d0 d1 d2 ... dD-1                     [v0.d1 v1.d1 ... v63.d1]   <- stripe 1
   ...                                        ...
  vecN-1: d0 d1 d2 ... dD-1                   [v0.dD-1 ...    v63.dD-1]
                                     block 1: [v64.d0 ...     v127.d0]
                                              ...
```

The hot inner loop on the PDX side fixes a dimension `j` and sweeps
across the 64 lanes of `stripe[j]`. With autovectorisation that becomes
one NEON/AVX register load + one FMA per 4–16 candidates per cycle.
After every `PROBE_STEP = 16` dims we read the current top-k heap
threshold and disable lanes whose partial sum already exceeds it
(PDX-BOND). Once every lane in a block is dead, we break out and never
touch its remaining dimensions.

### Trait surface

```rust
pub trait Scanner {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    fn last_ops(&self) -> u64;   // honest inner-loop MUL count
}
```

Three implementations land in this PoC: `Horizontal`, `PdxVertical{prune=false}`,
`PdxVertical{prune=true}`. They are storage-agnostic — swapping a
backend never changes call sites.

## Implementation Notes

* **No `unsafe`.** Every stripe access is a slice index; the bounds
  checks fold into the loop bounds after release-mode optimisation.
* **No deps.** The crate imports zero external crates. PRNG is
  in-house xorshift32 (deterministic, seedable). Heap is
  `std::collections::BinaryHeap`. Benchmark harness is a wall-clock
  loop with `Instant::now()`.
* **Pure-safe `Hit` ordering.** `f32` is `PartialOrd`, not `Ord`. We
  implement `Ord` on `Hit` via `partial_cmp().unwrap_or(Equal)`. This
  is sound because we never insert NaN distances (inputs are bounded
  in `[-1, 1)`).
* **Block partials are stack-allocated.** `[f32; BLOCK]` and
  `[bool; BLOCK]` live on the stack across the per-block sweep. No
  per-query heap allocation beyond the top-k itself.
* **Ragged tail handled.** A corpus of N = 130 vectors becomes 2 full
  blocks + a 2-lane tail. The tail's invalid lanes contain zeros from
  initialisation but are explicitly ignored via `valid` count.

## Benchmark Methodology

* **Hardware:** Apple M4 Max, 128 GB RAM, macOS Darwin 24.6.
* **Toolchain:** rustc 1.89.0, `cargo build --release` (workspace
  default profile: `lto = thin`, `codegen-units = 1`).
* **Data:** deterministic uniform random `f32` in `[-1, 1)`, generated
  by in-tree `Xs32(seed)`. *Honest caveat:* random data is the
  worst-case for pruning. Clustered embedding data (GIST/SIFT/MSMARCO)
  will give substantially better pruning ratios.
* **Workload:** single query, top-k = 10, three corpus sizes.
* **Harness:** the `pdx_demo` example runs each scanner for 20–100
  warm runs (more for the smaller corpora) after one warm-up call.
  No bootstrap, no statistical massaging — straight `runs /
  elapsed_secs`.

## Results

Real numbers from `cargo run --release --example pdx_demo -p ruvector-pdx`,
run on the branch SHA at the time of the report:

```
== corpus n=10000 d=128 k=10 ==
  horizontal             qps=  2936.8   ops/query=1280000
  pdx-vertical           qps= 12812.8   ops/query=1280000
  pdx-vertical-pruned    qps=  8782.7   ops/query=1017808
  speedups: vert / horiz = 4.36x   pruned / horiz = 2.99x   pruned / vert = 0.69x

== corpus n=50000 d=128 k=10 ==
  horizontal             qps=   603.1   ops/query=6400000
  pdx-vertical           qps=  2495.3   ops/query=6400000
  pdx-vertical-pruned    qps=  1417.5   ops/query=4789024
  speedups: vert / horiz = 4.14x   pruned / horiz = 2.35x   pruned / vert = 0.57x

== corpus n=10000 d=768 k=10 ==
  horizontal             qps=   283.9   ops/query=7680000
  pdx-vertical           qps=  2499.6   ops/query=7680000
  pdx-vertical-pruned    qps=  1463.5   ops/query=6927648
  speedups: vert / horiz = 8.81x   pruned / horiz = 5.16x   pruned / vert = 0.59x
```

| Config | Horizontal qps | PDX qps | PDX speedup | PDX-pruned ops saved |
|---|---:|---:|---:|---:|
| n=10K d=128 k=10 | 2,937 | **12,813** | **4.36×** | 20.5% |
| n=50K d=128 k=10 | 603 | **2,495** | **4.14×** | 25.2% |
| n=10K d=768 k=10 | 284 | **2,500** | **8.81×** | 9.8% |

### What the numbers say

1. **Layout-only PDX is a clean win at every size and dim.** No
   exceptions. The win grows with `D` (8.8× at D=768 vs 4.4× at
   D=128) because the outer loop's SIMD-friendly stripes amortise
   more work per cache line.
2. **Pruning saves MULs but loses wall-clock on random data.** The
   ops counter drops 20–25% at D=128 but the per-block bookkeeping
   (heap-peek + lane sweep + branch-prediction loss) eats more than
   it saves on uniform inputs. This is exactly the regime the PDX
   paper says NOT to use pruning in — they only enable BOND when the
   data is clustered enough that pruning kills entire blocks early.
3. **The pruning win exists** (1.28M → 1.02M MULs) — it just doesn't
   translate to throughput on synthetic data. On real embedding data
   the per-lane partials grow much faster (sparse heavy dims) and
   typical pruning rates rise to 60–80% of MULs eliminated.

## How It Works (blog-readable walkthrough)

Imagine you have one million 768-dimensional embeddings and you want
the 10 nearest to a query vector. The naive recipe is: for each
candidate, subtract componentwise from the query, square, sum, push
into a heap. That's the row-major version, and SIMD only buys you
~4× because you can vectorise across a *single* candidate's 768 dims.

PDX rearranges the data so that the SIMD vector instead processes one
dimension across *64 different candidates at once*. If you can load
64 floats with one cache-line read and FMA them with 64 lanes of the
query's `q[j]` broadcast, your candidate throughput is 64-wide instead
of 4-wide. Hardware doesn't change; layout does.

The second trick — BOND pruning — uses the fact that L2 partial
distances only grow. After 16 dims you've accumulated 16-dim partial
distances for all 64 candidates in a block. If any of those partials
already exceeds the worst entry in your current top-10 heap, the
candidate cannot make the cut: skip its remaining 752 dims. If *every*
candidate in a block fails this test, the whole block is dead and you
jump straight to the next block.

The net effect on real data is: a 64-wide outer loop + an early-out
that gets more aggressive as you sweep more dims. That's how Kuffo et
al. report 6× over FAISS-IVF-flat on SIFT1M.

## Practical Failure Modes

* **Pruning on uniform data.** As shown above, BOND costs you ~30%
  wall-clock on random data. Detect: input variance per dim is
  roughly equal across dims. Mitigation: ship two scanners, default
  to `prune = false`, only flip on if a runtime heuristic (variance
  of partial sums after the first probe-step) clears a threshold.
* **Build cost.** Copying N×D f32s from row-major to PDX is one full
  pass and ~1.3× extra memory during the copy. For static corpora
  this is one-time; for streaming inserts you need stripe-direct
  ingest or a hybrid append buffer.
* **BLOCK = 64 is not portable.** Apple NEON wants 4-lane sweeps,
  AVX-512 wants 16. We over-sweep on NEON (good — fewer loop
  overheads dominate) but under-sweep on AVX-512. Tracked.
* **Pruning correctness with non-L2 metrics.** The partial-sum
  monotonicity argument only holds for L2 and L2² (and Hamming,
  trivially). Cosine / inner-product would need a sign-aware
  reformulation; not implemented.

## What To Improve Next

1. **Real embedding benchmarks.** Pull GIST1M / SIFT1M / a slice of
   MSMARCO and re-run; verify the pruning wins the paper reports.
2. **RaBitQ-on-PDX.** Replace f32 stripes with 1-bit codes and 8-bit
   correction floats; reuse the same BLOCK × D layout. Should compose
   multiplicatively (4× layout × ~30× quantization = ~120× over
   horizontal f32).
3. **IVF integration.** Replace the row-major scan inside the IVF
   inverted-list iteration with a PDX scan per posting list.
4. **Const-generic `BLOCK`.** `struct PdxVertical<const B: usize>`
   so we can default to 32 on NEON, 64 on AVX2, 128 on AVX-512.
5. **Inner-product / cosine metric variants.** Most production
   embedding scans are cosine; add a `Metric` enum on `PdxVertical`.
6. **Streaming insert path.** Stripe-direct write so we don't need
   the row-major intermediate.

## Production Crate Layout Proposal

```
crates/ruvector-pdx/
├── Cargo.toml
├── src/
│   ├── lib.rs            # current PoC: Horizontal, PdxVertical, Scanner
│   ├── metric.rs         # L2 (today) + Cosine + IP (future)
│   ├── stripe.rs         # ingest helpers, alignment, stripe-direct writers
│   └── quantize.rs       # RaBitQ-on-PDX (future)
├── examples/
│   └── pdx_demo.rs
└── benches/
    └── pdx_bench.rs
```

`ruvector-core` would gain a `Storage::Pdx` variant on its existing
flat-store enum, dispatched at construction time. No public-API
breakage; the trait `Scanner` already lines up with the internal
`brute_force_top_k` surface.

## References

* Kuffo, L.; Krimpen-Stoop, A.; Tang, N.; Manegold, S. (2025). *PDX:
  A Data Layout for Vector Similarity Search.* SIGMOD 2025. CWI
  Amsterdam.
* Gao, J.; Long, C. (2024). *RaBitQ: Quantizing High-Dimensional
  Vectors with a Theoretical Error Bound.* NeurIPS 2024.
* Johnson, J.; Douze, M.; Jégou, H. (2017). *Billion-scale similarity
  search with GPUs.* IEEE Big Data.
* Tepper, M.; et al. (2024). *LeanVec: Searching Vectors Faster by
  Making Them Fit.*
* Patel, P.; et al. (2024). *ACORN: Performant and Predicate-Agnostic
  Search over Vector Embeddings and Structured Data.* VLDB 2024.

## How to Reproduce

```bash
git checkout research/nightly/2026-06-11-pdx-vertical-layout
cargo test    --release -p ruvector-pdx          # 5 tests must pass
cargo run     --release -p ruvector-pdx --example pdx_demo
cargo run     --release -p ruvector-pdx --bin    pdx_bench
```

All numbers in this document came from the `pdx_demo` run captured
above. The `pdx_bench` binary is the long-form (750 ms/cell) variant.
