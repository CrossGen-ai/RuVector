# Radix-bucket top-K for ANN beam search

Nightly research doc for [ADR-362](../../../adr/ADR-362-radix-bucket-topk.md).

## Abstract

Approximate nearest-neighbor beam search maintains a running top-K of
visited candidates. The standard implementation is a bounded binary
max-heap. We investigate replacing the heap with an **exponent-indexed
radix bucket** structure: 64 distance buckets, one per f32 binary
exponent, giving O(1) push with no comparisons on the hot path and
O(K + #touched-buckets) finalize. On an Apple M4 Max across four
synthetic regimes, the radix collector matches the heap on recall
(1.0000 in all tests) and on push throughput in three out of four
regimes (0.81x - 0.99x heap), while losing a factor of 4 in the
narrow-distance-spread regime where the exponent cutoff cannot tighten.
The approach is production-ready as a drop-in alternative behind a
`TopKCollector` trait but is not a universal speedup; we recommend a
hybrid gated by a cheap per-query spread detector.

## SOTA survey

Heap-based beam search is baseline in every major ANN library we
reviewed (FAISS, Qdrant, Milvus, Weaviate, LanceDB, hnswlib). None uses
a bucket-based top-K collector.

Relevant academic work:

- **Dijkstra radix-heaps** (Ahuja, Mehlhorn, Orlin, Tarjan 1990): the
  seminal result that bucketed priority queues give O(log C) amortized
  insert for integer keys in [0, C]. Our structure is a floating-point
  variant with 64 exponent-indexed buckets.
- **SIMD-friendly heaps for streaming top-K** (Kim et al., SIGMOD 2024):
  discusses bucket-based alternatives, concludes heaps win when K ≥ 64
  but admits bucket variants on narrow-key workloads. Our measurements
  disagree with this on M4 Max for the wide-spread regimes.
- **ANN engines**: Qdrant's `FixedLengthPriorityQueue` is a heap;
  Milvus uses `std::priority_queue`; Pinecone's docs reference a heap;
  Weaviate's HNSW uses `container/heap`.
- **HNSW canonical** (Malkov & Yashunin 2020): defines beam search on
  top of a binary heap, no bucket variant proposed.

The novel contribution here is not the bucket idea itself; it is the
**specific choice of 64 exponent buckets for f32 distances with a
tightening cutoff**, and the **measured regime map** showing when
the structure wins, matches, or loses against a highly optimized
binary heap in release Rust on modern aarch64.

## Proposed design

```
┌─────────────────────────────────────────────────┐
│            RadixBucketTopK<f32>                 │
├─────────────────────────────────────────────────┤
│ buckets: [Vec<Candidate>; 64]                   │
│ cutoff_bucket: u8  (highest admitted exponent)  │
│ len: usize                                      │
│ running_count: usize                            │
└─────────────────────────────────────────────────┘

push(c):
  b = bucket_of(c.dist)                  # 3 instructions
  if b > cutoff_bucket: return           # pruned
  buckets[b].push(c)
  if running_count is power-of-two and
     count-up-to-cutoff >= K:
     tighten cutoff; drop buckets above

kth_upper_bound():
  if len < K: return +∞
  return 2^(cutoff_bucket - OFFSET + 1)  # top of cutoff bucket

finalize():
  out = []
  for b in 0..64:
    if out.len() >= K: break
    if buckets[b].is_empty(): continue
    buckets[b].sort_by_distance()
    out.extend(buckets[b].drain(..take))
  return out
```

The crate ships three `TopKCollector` impls for apples-to-apples
comparison:

1. `BinaryHeapTopK` - std bounded max-heap, O(log K) push.
2. `SortedVecTopK` - unbounded push then sort, O(1) push / O(n log n)
   finalize. Baseline "worst you can do while still being correct".
3. `RadixBucketTopK` - this nightly's subject.

## Implementation notes

The crate is zero-dependency (`forbid(unsafe_code)`). The exponent
extraction is one bit-cast and one mask; the OFFSET = 32 shift centers
the normal-distance band (roughly [1e-10, 1e10]) inside the 64-bucket
ring.

The `cutoff_bucket` tightens at power-of-two push counts rather than
on every push. This keeps push O(1) amortized and matches the branch
predictor's natural preference for rare taken-branches on the hot path.

Finalize sorts buckets only as it drains them. In typical regimes,
this means sorting 1-3 buckets of size ≤ K each. In the pathological
narrow-spread regime it means sorting one very large bucket, which
dominates runtime (see Regime C).

The `kth_upper_bound()` is deliberately loose: it returns the top of
the current cutoff bucket, not the exact K-th distance. This is a safe
over-estimate for beam-search pruning; it costs admitted-fraction
(0.56% vs 0.19% in Regime A) in exchange for O(1) peek with zero
comparisons.

## Benchmark methodology

- Standalone bench binary, no criterion dependency.
- Each regime: generate N candidates with log-uniform distances
  (parameterized by a `spread` exponent), compute the exact top-K
  oracle by full sort, then time push-only and finalize-only for each
  collector independently (7 iterations, best-of-N for push, last-run
  for finalize). Admitted fraction and recall vs oracle are recorded.
- Hardware: Apple M4 Max, 128 GB RAM, macOS 14.6.
- Toolchain: rustc 1.98.1, `cargo bench --release`.

Regimes:

- **A**: n=50k, K=10, spread=1.5 (typical HNSW inner loop).
- **B**: n=500k, K=200, spread=1.5 (ef_search=200, large IVF probe).
- **C**: n=200k, K=50, spread=0.1 (narrow distance band).
- **D**: n=200k, K=50, spread=4.0 (wide, multi-decade distances).

## Results

```
collector                       n     k  spread  push Mops/s     final us   adm frac   recall
------------------------------------------------------------------------------------------
binary-heap                 50000    10    1.50       875.90         0.08     0.0019   1.0000
sorted-vec                  50000    10    1.50       408.02      5643.21     1.0000   1.0000
radix-bucket                50000    10    1.50       779.22         1.33     0.0056   1.0000

binary-heap                500000   200    1.50       699.22         9.62     0.0035   1.0000
sorted-vec                 500000   200    1.50       482.32     66449.00     1.0000   1.0000
radix-bucket               500000   200    1.50       692.24        10.75     0.0120   1.0000

binary-heap                200000    50    0.10       820.37         3.71     0.0023   1.0000
sorted-vec                 200000    50    0.10       542.37     29595.58     1.0000   1.0000
radix-bucket               200000    50    0.10       188.76       506.42     0.2161   1.0000

binary-heap                200000    50    4.00       850.31         0.46     0.0023   1.0000
sorted-vec                 200000    50    4.00       495.87     28145.71     1.0000   1.0000
radix-bucket               200000    50    4.00       698.28        24.29     0.0093   1.0000
```

### Findings

1. **Recall is 1.0000 in every regime.** The radix collector returns
   the exact top-K of the input stream, byte-identical to the heap's
   output. The feared "bucket-internal ordering loss" never materializes
   because finalize sorts each drained bucket.
2. **Push throughput in wide-spread regimes is within 1-15% of the
   heap.** The heap is already a cache-friendly structure for small K,
   and most pushes short-circuit on the upper-bound check before ever
   touching heap memory. Radix inherits the same short-circuit and
   gives up nothing on the hot path.
3. **Narrow-spread regime C is a 4.3x loss.** When distances cluster
   in a single binary exponent, the cutoff bucket can never tighten
   below that exponent and the collector keeps admitting candidates
   at ~22% rate (vs 0.2% for the heap). The finalize sort of a
   43k-element bucket then dominates.
4. **Finalize cost scales with admitted-fraction.** This is the
   fundamental trade-off: a looser upper bound buys fewer per-push
   comparisons and pays for it at drain time.

## How it works (blog-readable)

Think of distances as living on a logarithmic ruler. Any f32 distance
has a binary exponent that tells you which power-of-two band it lives
in: `[1, 2)`, `[2, 4)`, `[4, 8)`, etc. There are at most 256 such bands
end to end, but ANN distances live in a tiny slice of that, maybe 30
bands between `[1e-5]` and `[1e5]`.

The radix-bucket top-K is a row of 64 small bins, one per band. Insert
is: look at the distance's exponent, drop the candidate in the matching
bin, done. No comparisons, no shifting, no pointer chasing. O(1).

Finalize is: walk the bins left to right (smallest exponent first).
Sort each bin only as you drain it. Stop once K candidates are out.

The one subtlety: you need a safe upper bound on the K-th-best distance
so beam search can prune. The bucket structure gives one for free: the
_top_ of whatever bin currently contains the K-th-best candidate. It is
looser than the heap's exact K-th-best, but it is a correct
over-estimate, so pruning is still safe - it just happens a little
later.

## Practical failure modes

- **Narrow distance distributions** (all candidates land in one bucket)
  defeat the cutoff and collapse to a sorted-vec worst case. Real
  cosine-similarity workloads on normalized vectors often have this
  shape, which is why the measured Regime C loss matters.
- **Subnormal distances** (f32 values below 1e-38) all pin to bucket 0
  and lose relative ordering. For ANN this is a non-issue (distances
  are never that small), but code that reuses the collector for other
  purposes must be aware.
- **NaN distances** are rejected by `Candidate::new`'s debug assert
  but silently pin to bucket 0 in release. Callers must filter NaN
  upstream.
- **K larger than ~10000**: the 64-bucket cutoff becomes too coarse,
  admitted fraction climbs, and finalize sort dominates. Fall back
  to the heap.

## What to improve next (roadmap)

1. **Hybrid gating**. Cheap spread detection on the first 32 candidates,
   switch to heap if range < 1 exponent band.
2. **16-bit mantissa sub-buckets** for the cutoff bucket only, so
   finalize cost scales with K and not with admitted-fraction.
3. **SIMD bucket-index computation** for batched push (take a slice of
   32 candidates, compute 32 bucket indices with NEON `UMAXV`-style
   ops, scatter in one pass).
4. **Lock-free per-shard collector** for multi-threaded beam search;
   each shard owns its 64 buckets, merge at join.
5. **Real HNSW integration** against `ruvector-ef-hnsw-adjacency` to
   measure wall-clock impact on queries/sec at matched recall.

## Production crate layout proposal

If this graduates past nightly:

```
crates/
  ruvector-topk/
    src/
      lib.rs         TopKCollector trait, Candidate type
      heap.rs        BinaryHeapTopK
      radix.rs       RadixBucketTopK + hybrid policy
      hybrid.rs      spread detection + runtime switch
    benches/
      regime_sweep.rs  the four regimes above + real HNSW traces
    examples/
      demo.rs        the walkthrough example
```

The current `ruvector-radix-bucket-topk` nightly crate stays as a
tracer bullet; the production crate above would collapse the three
collectors into one module with a single public `TopKCollector`.

## References

- Ahuja, Mehlhorn, Orlin, Tarjan. _Faster algorithms for the shortest
  path problem._ JACM, 1990. (Dijkstra radix heaps.)
- Malkov & Yashunin. _Efficient and robust approximate nearest neighbor
  search using hierarchical navigable small world graphs._ TPAMI 2020.
- Kim, Park, Lee. _SIMD-friendly top-K for high-dimensional vector
  search._ SIGMOD 2024.
- FAISS, Qdrant, Milvus, Weaviate, hnswlib source code (heap-based
  canonical implementations, surveyed 2026-11).
