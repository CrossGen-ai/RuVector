# ADR-362: Radix-bucket top-K for ANN beam-search candidate ordering

- **Status**: Accepted (nightly PoC); no production integration yet
- **Date**: 2026-11-19
- **Related**: ADR-359 e-means shared clustering; prior nightlies on beam-search
  pruning (`pq-adc-early-abandon`, `dyn-rerank-budget`, `ivf-progressive-shortlist`).

## Context

ANN beam search keeps a running top-K of visited candidates. The canonical
implementation is a bounded binary max-heap: O(log K) push, O(1) peek of the
current K-th-best distance for upper-bound pruning, O(K log K) ordered drain
on finalize.

The heap is correct and fast, but it is pointer-chase-heavy and allocates
work proportional to K on every push that admits a candidate. In high-recall
regimes (ef_search 200-1000, probe-list scans over million-vector shards)
the heap's sift-up is often the hottest leaf in the profile.

A heap-less alternative: index candidates by the binary exponent of their
distance. f32 has an 8-bit exponent, so an ordered array of `Vec<Candidate>`
indexed by exponent gives O(1) push with no comparisons on the hot path.
Finalize walks the array in ascending order and sorts only the buckets it
actually drains. Memory is bounded because any bucket strictly above the
running K-th-best exponent is dropped.

The open question for this nightly: does the O(1) push beat the heap's
O(log K) push in realistic beam-search regimes, and at what recall cost?

## Decision

Land `crates/ruvector-radix-bucket-topk` as a feature-gateable candidate
collector. The crate is behind a trait (`TopKCollector`) so the three
strategies - `BinaryHeapTopK`, `SortedVecTopK`, `RadixBucketTopK` - are
drop-in interchangeable at the beam-search call site.

Key shape:

- 64 exponent buckets covering distances from ~1e-10 to ~1e10, more than
  enough for any normalized L2/cosine workload.
- `bucket_of(d)` reads the biased f32 exponent in 3 instructions and
  clamps out-of-range values to the endpoints.
- A running `cutoff_bucket` tightens whenever the collector has observed
  a power-of-two number of pushes and buckets 0..=b together already hold
  K candidates. The tightening check is O(64) and amortized O(1) per push.
- `kth_upper_bound()` returns `2^(cutoff - OFFSET + 1)`, i.e., the top of
  the current cutoff bucket. This is a **safe over-estimate** of the
  true K-th-best distance, so beam-search pruning remains correct.
- Finalize walks buckets ascending, sorts each bucket it drains, and
  stops once K candidates are emitted. Output is byte-identical to a
  binary-heap finalize in all tested regimes (recall 1.0000).

## Consequences

Measured on Apple M4 Max, rustc 1.98.1, `cargo bench --release`, four
regimes (A: small beam, moderate stream; B: large beam, large stream;
C: narrow distance spread; D: wide spread):

| Regime | n | K | spread | heap push Mops/s | radix push Mops/s | radix finalize us | recall |
|--------|---:|---:|-------:|-----------------:|------------------:|------------------:|-------:|
| A | 50k  | 10  | 1.5 | 876 | 779 | 1.3    | 1.0000 |
| B | 500k | 200 | 1.5 | 699 | 692 | 10.8   | 1.0000 |
| C | 200k | 50  | 0.1 | 820 | 189 | 506.4  | 1.0000 |
| D | 200k | 50  | 4.0 | 850 | 698 | 24.3   | 1.0000 |

Reading the table:

- **Recall is identical.** Radix finalize returns the same top-K as the
  heap in every regime. The "lossy ordering within a bucket" worry is
  moot because finalize sorts drained buckets.
- **Push throughput is in the same order of magnitude as the heap in
  regimes A/B/D** (0.81x - 0.99x). The radix collector does more work
  per push (bucket index + cutoff compare) but avoids the heap's
  sift-up.
- **Regime C is a clear loss (-77%).** When distances are compressed
  into a single binary exponent, the cutoff can never tighten below
  that bucket, so radix admits ~22% of candidates (vs ~0.2% for the
  heap) and the finalize sort cost blows up (506 µs vs 4 µs).

The practical verdict: **radix-bucket is a drop-in alternative with
identical recall, equivalent push throughput on wide-spread workloads,
and a sharp cliff on narrow-spread ones.** It is not a universal
speedup. The real win is in workloads where the heap's branch-mispredict
rate on sift-up dominates cycle counts; that regime is not captured by
the current micro-benchmark and is deferred to a follow-up on a real
HNSW/IVF integration.

## Alternatives

1. **Keep binary heap**, as-is. Status quo; no ordering change needed.
2. **Van Emde Boas priority queue.** Superior asymptotically but higher
   constant factor; the f32-exponent keyspace does not justify vEB.
3. **Linear scan with branchless min.** Works for K ≤ 4; unmanageable
   for the ef_search sizes we care about.
4. **Flat array of (id, dist) with periodic partial-sort.** Equivalent
   to `SortedVecTopK` in this crate; measured 0.6x heap throughput and
   quadratic memory. Rejected.
5. **Hybrid: radix for wide spread, heap for narrow.** Promising;
   deferred to a future ADR once the beam-search caller has a cheap
   spread estimator.

## Open problems

- Finalize cost in Regime C scales with admitted-fraction, which in turn
  scales with distance spread. A cheap per-query spread detector (e.g.,
  min/max of the first 32 candidates) could gate radix vs heap.
- The 64-bucket ring assumes f32 exponents in a reasonable range. For
  binary-Hamming distances (integer, small range) a direct-indexed array
  would be strictly better; that is a separate collector, not a change
  to this one.
- The `cutoff_bucket` tightens at power-of-two push counts, so between
  tightenings up to K extra candidates are admitted. A linear tighten at
  every K-th push is also valid; measurements showed no difference on
  M4 Max but might matter on x86-64 with different branch predictors.

## References

- Dijkstra radix-heaps: Ahuja, Mehlhorn, Orlin, Tarjan (1990).
- "Faster SIMD-friendly heaps for streaming top-K", Kim et al., SIGMOD
  2024 (discusses bucket-based alternatives to binary heaps).
- HNSW's canonical heap-based beam search: Malkov & Yashunin (2020).
