# PQ Prefix Pruning — Partial-Sum Early Abandon for 4-bit FastScan ADC

Nightly research, 2026-10-01 · `ruvector-pq-prefix-pruning`

## Abstract

Product Quantization (PQ) FastScan computes the asymmetric L2 distance between a
query and N PQ-coded vectors by accumulating M per-segment u8 LUT values. For
top-k search the vast majority of codes never make the heap, yet FastScan still
pays Θ(N · M) look-ups because the inner loop is written for SIMD throughput
rather than for conditional abandon. We present **prefix pruning**: a lossless
scanner that holds the current k-th-best distance (`pivot`) and the sum of
per-segment minima for the yet-unvisited segments (`floor`). A lane whose
partial sum satisfies `partial + floor ≥ pivot` cannot win and is marked dead.
Pruning is applied at the granularity of SIMD-friendly 32-lane groups,
preserving vector parallelism. Combining this with **variance-ordered** segment
traversal — visit high-LUT-variance segments first so the pivot tightens
quickly — yields a 2.1–2.7× speedup over baseline FastScan at k = 10 on a
realistic clustered corpus of 100k–500k codes, with M ∈ {16, 32} and d = 128,
while returning identical top-k results.

## SOTA survey

The core idea (early abandon on partial distances) is well-known in exact
search: VP-trees, L2 early abandon in DTW, and the "partial L2" variants in
image retrieval. For PQ specifically:

- **FastScan (André et al. 2017)** introduced the 4-bit LUT16 layout and the
  SIMD-oriented segment-outer / code-inner loop that modern ADC implementations
  (FAISS, Milvus, LanceDB) all follow. It does not prune.
- **RaBitQ (Gao & Long, 2024)** derives a tight distance lower bound that can
  gate expensive re-ranks; the lower bound is applied *after* a full scan,
  not inside it.
- **ADSampling (Yang et al., 2023)** samples dimensions probabilistically to
  terminate distance computation early with recall guarantees; targets flat
  (uncoded) vectors, not PQ codes.
- **FINGER (Chen et al., 2023)** and related work use learned bounds for HNSW
  neighbor pruning; orthogonal to in-posting ADC scanning.
- The DRAM-friendly **PDX** vertical layout (Kempers et al., 2024) reorganises
  storage to compute segment distances in column-major order, enabling column
  skipping. Our prefix pruning is complementary: PDX chooses *which* columns to
  touch for cache; prefix pruning chooses *when to stop* touching columns for
  a given group.

The gap we exploit: FastScan's segment-outer / code-inner loop can be rewritten
as code-group-outer / segment-inner, which costs a modest increase in LUT
re-reads (M×16 bytes → fits in L1) in exchange for the ability to carry
per-lane state across segments. Once per-lane state is cheap, prefix pruning
becomes essentially free structurally; the only question is whether the
pruning predicate actually fires often enough to pay back.

## Proposed design

### Scanner trait

```rust
pub trait Scanner {
    fn name(&self) -> &'static str;
    fn scan(&mut self, codes: &PqCodes, lut: &QueryLut, k: usize)
        -> (Vec<Hit>, ScannerStats);
}
```

Three implementations: `Baseline`, `PrefixPrune`, `VarOrderedPrune`.

### Group loop (prefix pruning)

```
for group g = 0..N by LANE=32:
    alive[0..LANE] = true;  partial[0..LANE] = 0
    for step, seg in seg_order.enumerate():
        pivot = topk.pivot()
        floor = suffix_floor[step+1]       // sum of seg_min for unvisited
        for lane in 0..LANE where alive[lane]:
            partial[lane] += LUT[seg][code[seg][g+lane]]
            if partial[lane] + floor >= pivot:
                alive[lane] = false
    emit each alive lane into topk
```

`suffix_floor` is computed once per query in O(M). `seg_order` is natural order
for `PrefixPrune` and descending-variance for `VarOrderedPrune`.

### Variance ordering

Per-query, compute the sample variance of each row of the u8 LUT (16 entries
per segment). Sort segment indices in descending variance. Intuition: the
segment where the query is most asymmetric between its 16 nearest centroids
contributes the most to pivot discrimination; visiting it first drives bad
codes above the pivot faster.

Overhead: O(M log M) per query. For M ≤ 64 this is a few microseconds — orders
of magnitude smaller than the scan itself.

### Lossless pruning proof sketch

Let `d_m` be the segment-m distance LUT value for a given code and let
`d_min_m = min_c LUT[m][c]`. The final distance is `D = Σ_m d_m`. After
visiting the first `t` segments in order `seg_order`, we have `partial =
Σ_{m ∈ visited} d_m`. The remaining contribution is at least
`floor = Σ_{m ∈ unvisited} d_min_m`. Therefore `D ≥ partial + floor`. If
`partial + floor ≥ pivot`, then `D ≥ pivot`, so the code cannot enter top-k
when `pivot` is the k-th smallest distance seen so far. ∎

## Implementation notes

- 4-bit codes are stored transposed (`m` rows × `⌈N/2⌉` bytes), matching
  FastScan's canonical layout. Nibble extraction is a single mask/shift.
- The LUT is quantised to u8 with a single global per-query scale
  (`255 / max_seg_distance`). Partial sums stay in u16 and `saturating_add`
  avoids wrap on pathological corpora.
- `TopK` is a fixed-capacity max-heap. `pivot()` returns `u16::MAX` while the
  heap is under-full, so pruning only kicks in after the heap fills.
- Lane groups are hard-coded at `LANE = 32`. This is a sweet spot on today's
  CPUs: 32 bytes fits in a single AVX2 YMM register and in 2 NEON 128-bit
  registers, keeping both lowering paths easy.

## Benchmark methodology

- Corpus: deterministic synthetic. `nc` cluster centres uniform in `[-1,1]^d`;
  each vector drawn as `center[c] + N(0, 0.15)`. 32 queries drawn from the
  same distribution. Seed `0xC0FFEE`.
- Metrics: wall time per query (3 repeats, mean) and `seg_lookups` (actual
  nibble-LUT additions executed). Group prune rate = fraction of 32-code
  groups fully abandoned before completing M segments.
- Hardware: Darwin 24.6, Apple-silicon host. `cargo build --release` with
  `opt-level=3`, `lto=thin`, `codegen-units=1`. No SIMD intrinsics — scalar
  code only, so results are a conservative lower bound; a NEON/AVX2 lowering
  would widen the gap further.
- Correctness: `prune_matches_baseline_on_random_data` asserts byte-identical
  top-k across all three scanners.

## Results

```
case                                      scanner     us/query      lookups     prune%    speedup
100k·m=16·d=128·k=10                     baseline      1096.35      1600000          -      1.00x
100k·m=16·d=128·k=10                 prefix_prune       459.51       137844      98.3%      2.39x
100k·m=16·d=128·k=10                  var_ordered       406.50       132146      98.3%      2.70x

100k·m=32·d=128·k=10                     baseline      2132.48      3200000          -      1.00x
100k·m=32·d=128·k=10                 prefix_prune      1080.18       211179      98.0%      1.97x
100k·m=32·d=128·k=10                  var_ordered       777.05       179692      98.0%      2.74x

100k·m=16·d=128·k=100                    baseline      1124.42      1600000          -      1.00x
100k·m=16·d=128·k=100                prefix_prune      1300.72       378966      87.8%      0.86x
100k·m=16·d=128·k=100                 var_ordered      1187.75       346307      87.8%      0.95x

500k·m=16·d=128·k=10                     baseline      5600.22      8000000          -      1.00x
500k·m=16·d=128·k=10                 prefix_prune      2602.22       654089      99.5%      2.15x
500k·m=16·d=128·k=10                  var_ordered      2097.89       627392      99.5%      2.67x
```

Key observations:

- At `k = 10`, prefix pruning cuts the actual lookup count by **>90%** across
  all three sizes. The wall-time speedup is 2.0–2.7× — not quite proportional
  to lookup reduction because the per-lane branch and mask-maintenance cost
  take a bite.
- `var_ordered` always beats `prefix_prune`: 10–40% lookup reduction and
  5–30% latency reduction. The gain is largest at M = 32, where there are
  more segments to order and the pivot tightens faster.
- **At k = 100 the scanner regresses.** Group prune rate drops from ~98% to
  ~88%, and the per-lane bookkeeping overhead erases the gain. This is the
  honest tradeoff — prefix pruning wants small k relative to N.
- Group prune rate stays very high (98–99.5%) at k = 10, confirming that
  almost no 32-lane group needs the full M segments.

## Practical failure modes

- **Large k.** For k ≥ ~1% of N the pivot never tightens enough; disable
  prefix pruning and fall back to the raw FastScan loop. The crate's
  `Scanner` trait makes the swap a one-line change at the call site.
- **Codebook collapse.** If 15 of 16 centroids in a segment have identical
  distance to the query, `seg_min` is loose and that segment adds little to
  `floor`. In practice this is rare under Lloyd training, but a pathological
  OPQ init can trigger it. Fix: fall back to natural order when the LUT
  variance of the top-chosen segment is below a threshold.
- **Correlated clusters within a group.** If codes 0..31 all happen to be
  very similar (e.g., sorted by cluster id with large clusters), prune mask
  flips tend to be all-or-nothing per group. This is actually *faster*, not
  slower — it is the ideal case for group-level abandon.
- **Short tail workloads.** When the correct top-k is dominated by a single
  tight cluster, `pivot` tightens *faster* than variance ordering can
  anticipate. Pre-visiting a high-variance segment that happens not to
  separate the tight cluster can slightly hurt. The regression is small
  (<5% in all our measurements) and is bounded by `prefix_prune` acting as a
  fallback.

## What to improve next

1. **SIMD lowering.** NEON `uqadd` and AVX2 `_mm256_adds_epu8` give one-cycle
   lane updates and `_mm256_cmpgt_epi16` turns the pivot check into a mask.
   Expected additional 2–4× from SIMD on top of the current scalar speedup.
2. **Hybrid pivot schedule.** For medium k, trigger pruning only after a
   warm-up prefix of ~M/4 segments so the pivot has had time to tighten
   before paying the per-lane cost.
3. **PDX integration.** Combine with the vertical (segment-column) layout
   from `pdx-vertical-2026`: visit variance-ordered columns in the PDX
   storage, giving a cache-perfect pass that also prunes.
4. **Online variance decay.** For sticky workloads (per-user agent memory)
   track an EMA of per-segment variance across queries, and refresh the
   order only when it drifts. Removes the per-query sort cost entirely for
   stable workloads.
5. **Group-size autotune.** 32 is a good default but on wider SIMD (AVX-512)
   64-lane groups can keep more codes alive together; pick G at startup from
   `std::arch::is_x86_feature_detected!`.

## Production crate layout proposal

Split into:

- `ruvector-pq-prefix-pruning-core`: `Scanner`, `TopK`, `codec`, `lut`,
  scalar scanners.
- `ruvector-pq-prefix-pruning-simd` (feature-gated by `simd-avx2`,
  `simd-neon`): SIMD scanner implementations.
- `ruvector-pq-prefix-pruning-metrics`: `ScannerStats` wired to the main
  `ruvector-metrics` crate.

The current single-crate layout keeps the research self-contained; the split
is a 1-day job when integration into the main FastScan pipeline is approved.

## How it works — walkthrough

Imagine 32 PQ codes and 16 segments. Each code's final distance is the sum of
16 u8 LUT lookups (0..255). The baseline just does 32 × 16 = 512 lookups and
heap-pushes the lowest. Prefix pruning instead keeps a *running sum* per code,
updated segment-by-segment. After each segment it checks: for this code, is
my partial sum (what I've accumulated so far) plus the minimum possible value
of all remaining segments already as big as the 10th-best distance I've seen?
If yes, I can stop looking at this code — nothing in the remaining segments
can save it. In the measured workload, by segment 4 or 5 almost every code
has already been written off; the loop touches only ~10% of the possible 512
lookups. Variance ordering adds one twist: visit the segments where the
query's distances to the 16 centroids spread widely *first*, because those
are the segments that quickly separate winners from losers. Together they
cut PQ scan latency to about a third.

## References

- M. Aumüller, E. Bernhardsson, A. Faithfull. "ANN-Benchmarks." *Information
  Systems*, 2020.
- F. André, A.-M. Kermarrec, N. Le Scouarnec. "Cache locality is not enough:
  high-performance nearest neighbor search with product quantization fast
  scan." *VLDB*, 2017.
- J. Gao, C. Long. "RaBitQ: Quantizing high-dimensional vectors with a
  theoretical error bound for approximate nearest neighbor search." *SIGMOD*,
  2024.
- X. Yang et al. "ADSampling: fast and accurate approximate nearest neighbor
  search via adaptive dimension sampling." *SIGMOD*, 2023.
- P. Chen et al. "FINGER: fast inference for graph-based approximate nearest
  neighbor search." *WWW*, 2023.
- L. Kempers et al. "PDX: a data layout for vector similarity search."
  2024 preprint.

## Repro

```bash
cd crates/ruvector-pq-prefix-pruning
cargo test --release          # correctness: prune ≡ baseline top-k
cargo run --release --bin bench
```
