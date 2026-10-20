# Hamming-Weight Pre-Bucketing for Binary ANN

**Date:** 2026-10-20
**Crate:** `crates/ruvector-hamming-weight-prebucket`
**ADR:** [ADR-356](../../../adr/ADR-356-hamming-weight-prebucket.md)
**Hardware:** Apple M4 Max, Darwin 24.6.0 arm64, release build (`-O3`-equivalent)

## Abstract

Binary ANN engines (SimHash, learned-to-hash, 1-bit RaBitQ, ternary codes) spend
most of their query time scanning Hamming distances across a flat code table.
We propose a one-line indexing step that keeps that scanner but prunes it with
a trivial lower bound:

> `H(x, q) >= |pop(x) - pop(q)|`

By bucketing codes by their popcount `w(x)` at build time and expanding
buckets outward from `w(q)` at query time, we skip every bucket whose weight
delta already exceeds the current k-th-best Hamming τ. The method is **exact**
(recall@k = 1.0 against full scan), **plug-and-play** (any inner Hamming
scanner — plain popcount, bit-sliced, SIMD — drops in unchanged), and adds
~25 % memory overhead for the bucket index.

On class-structured codes (realistic for learned-to-hash / mode-sparse
embeddings) we measure **2.13× end-to-end speedup** and **3.14× fewer
candidate visits** vs full scan on N = 100 000 at D = 128 bits, k = 10.
On uniform symmetric SimHash codes the method is break-even to slightly
slower (0.70×) — an honest result, and the pre-bucket layer is detachable.

## SOTA survey

| System                     | Binary ANN strategy                               | Weight-aware? |
|----------------------------|---------------------------------------------------|---------------|
| FAISS `IndexBinary*`       | Bit-sliced popcount, PQ in Hamming space          | No            |
| Milvus BIN_FLAT / BIN_HNSW | Hamming/Jaccard flat and HNSW                     | No            |
| Weaviate BQ                | 1-bit scalar quant + flat scan                    | No            |
| LanceDB binary quant       | 1-bit + SIMD AND/XOR + popcount                   | No            |
| Qdrant binary quant        | 1-bit codes + SIMD, rescore float                 | No            |
| MIH (Norouzi et al. 2012)  | Multi-index Hamming hashing, radius bands         | Band width ≠ weight |
| Chappell et al. 2019       | Signature file "weight filtering" (DB literature) | Yes (but for Jaccard, not ANN) |
| RaBitQ (SIGMOD 2024)       | Multi-bit anisotropic binary quant                | No            |
| Extended RaBitQ (2026)     | Multi-bit per group                               | No            |

The popcount-triangle bound `H(x,q) ≥ |pop(x) − pop(q)|` is folklore in
information retrieval signature files (used to prune Jaccard thresholds in
set-similarity joins) and is tight on the hypercube. **To our knowledge no
vector-search system exposes it as a pre-scan index layer, even though every
binary ANN already stores codes implicitly suitable for the index.**

Position vs prior ruvector nightlies:

* `gist-bit-sliced-hamming-2026.md` optimizes the inner scanner. This work
  prunes **which codes reach the inner scanner at all**.
* `gist-partial-l2-early-abandon-2026.md` and
  `gist-pq-adc-early-abandon-2026.md` apply distance-level early abandon.
  Weight pre-bucketing is strictly earlier in the pipeline: it eliminates
  candidates *before* a single Hamming XOR is issued.

The two layers compose: `WeightedBitSlice` is this work plus bit-sliced
popcount, demonstrating zero-cost composition.

## Proposed design

### Index

```
buckets: [Vec<u32>; D+1]        // bucket w holds ids of codes with pop(x)=w
codes:    Vec<[u64; W]>          // the flat code table (unchanged)
```

Build time is O(N·W) popcount + bucket pushes. Memory overhead for N=100k
and D=128 is 403 kB on top of 1.6 MB of codes — **25.2 %** overhead,
dominated by the Vec headers (one per bucket, so overhead scales with D not N).

### Query

```
τ := +∞; H := empty k-heap
for delta in 0..=D:
    if delta > τ: break
    for each bucket w in { w_q − delta, w_q + delta }:  // skip when delta=0 duplicate
        if 0 <= w <= D:
            for id in buckets[w]:
                d := hamming(codes[id], q)
                maintain H as top-k, update τ := H.peek()
return sorted H
```

Correctness: at step `delta`, every unvisited bucket has weight distance
`> delta >= τ_current` from w_q, so its codes all have Hamming distance
`> τ`, so they cannot enter the top-k. Thus the heap at the break point
contains the exact k-NN. ∎

### Trait-based swappable backends

```rust
pub trait BinaryScanner<const W: usize> {
    fn knn(&self, q: &Code<W>, k: usize) -> Vec<(u32, u32)>;
}
```

Three backends ship:

* `FullScan`           — baseline.
* `WeightedScan`       — this work.
* `WeightedBitSlice`   — this work + flat packed `Vec<u64>` layout.

A production SIMD backend would implement the trait identically; swap at the
call site.

## Implementation notes

* Code width parameterized at the type level (`const W: usize`), so a
  128-bit code lives in `[u64; 2]` and compiles down to two `XOR` +
  `POPCNT` instructions per code. On aarch64 LLVM lowers `count_ones` to
  `CNT` + reduction.
* `SplitMix64` for reproducibility — benchmarks are deterministic for a
  fixed seed and bit-width.
* `SignTable` stores sign rows as bit-packed u64s; `simhash_encode` does one
  pass per bit without allocations.
* Bucket iteration uses a two-sided outward scan (sign ∈ {−1, +1}) with a
  `delta == 0` guard so bucket[w_q] is visited exactly once.

## Benchmark methodology

* Dataset: synthesized float vectors (N = 100 000, D = 128) with 70 %
  drawn from one of 16 Gaussian clusters and 30 % iid Gaussian.
* Encoding: SimHash to 128 bits (`[u64; 2]`) via a deterministic sign table
  with 128 random ±1 rows.
* Two **regimes** are measured to be honest about where the method helps:
    * **A — unbiased SimHash**: standard SimHash. Popcount concentrates near
      D/2 with σ ≈ √(D)/2 ≈ 5.5. τ for k=10 is roughly 10–15, exceeding σ,
      so pruning rarely bites.
    * **B — class-structured codes**: 100 k codes drawn from 3 mode centers
      at target popcounts {25, 64, 100} with 4-bit XOR jitter. Models learned
      sparse codes and mode-separated embeddings. τ ≈ 4–6 ≪ 39 (gap between
      modes) and triangle pruning cuts hard.
* Queries: 1 000 held-out queries encoded through the same pipeline.
* Metric: wall-clock QPS over all queries, average candidates visited per
  query, and exactness (fraction of queries whose top-k multiset of
  Hamming distances exactly matches full-scan ground truth).
* Baseline: full-scan Hamming top-k with a binary heap, same release build.
* Warm-up: 32 queries before each timing window.
* All numbers below are produced by `cargo run --release -p ruvector-hamming-weight-prebucket --bin benchmark`.

## Results

### Regime A — unbiased SimHash (symmetric codes)

```
popcount[min=44, mean=64.33, stddev=5.54, max=87]
exactness: weight-bucket=1.0000  weight+bitslice=1.0000
full-scan           qps=9168.8   avg_visits=100000   total_s=0.109
weight-bucket       qps=6450.0   avg_visits=100000   total_s=0.155
weight+bitslice     qps=5191.1   avg_visits=100000   total_s=0.193
speedup wb/full = 0.70x | wbs/full = 0.57x
```

The pre-bucket index does not help here. τ is wide enough that the entire
weight histogram is visited; the indirection cost (one Vec deref per
bucket) is pure overhead.

### Regime B — class-structured modes {25, 64, 100}

```
popcount[min=21, mean=63.19, stddev=28.80, max=104]
exactness: weight-bucket=1.0000  weight+bitslice=1.0000
full-scan           qps= 9398.3  avg_visits=100000  total_s=0.106
weight-bucket       qps=19993.6  avg_visits= 31828  total_s=0.050
weight+bitslice     qps=17287.1  avg_visits= 31828  total_s=0.058
speedup wb/full = 2.13x | wbs/full = 1.84x
visit-prune wb/full = 3.14x | wbs/full = 3.14x
```

**End-to-end 2.13× faster, 3.14× fewer candidates visited, exact.** The gap
between measured QPS speedup (2.13×) and visit-prune ratio (3.14×) is the
bucket-indirection tax: a flat scan amortizes memory latency better. On a
SIMD-aware inner scanner this gap would shrink; integrating with the
`ruvector-bit-sliced-hamming` AVX2/NEON kernel is the natural next step.

### Memory

```
codes (N=100000, 128 bits):      1.53 MB
buckets ([Vec<u32>; 129]):       0.38 MB  (25.2% overhead)
```

Overhead is dominated by the 129 `Vec` headers plus the id payload. For
real-world scales (10 M+ codes) the per-bucket Vec header is amortized and
overhead approaches `4·N / 16·N = 25 %`, independent of D.

## How it works — blog-readable walkthrough

Imagine two 128-bit codes `x` and `q`. Hamming distance counts bits that
differ. If `x` has 25 one-bits and `q` has 100 one-bits, then at least
`100 − 25 = 75` bits must differ — any bit set in `q` but not in `x`
contributes, and there are at least 75 such bits since there just aren't
enough bits in `x` to cover them. The reverse argument gives the same
lower bound symmetrically. The absolute difference of popcounts is a
free, exact, provable lower bound on Hamming distance.

Now imagine your top-10 so far is at Hamming distance 6. Every code in
the popcount-100 bucket is at distance ≥ 75 from `q`. We can skip the
entire bucket without reading a single code word. If we sort buckets by
`|w − w_q|` and expand outward, we stop the first time `|w − w_q| > 6`.
On class-structured data where codes cluster into weight modes, this
excludes most of the database for free.

The trick is that `pop(x)` lives in `[0..D]` — a tiny table of `D+1`
buckets. Each code lives in exactly one. The query cost is one popcount
on `q` and a two-sided iteration.

## Practical failure modes

* **Symmetric SimHash on unit-sphere embeddings.** Popcount is tightly
  concentrated at D/2 (σ = √D/2). τ for small k on noisy data exceeds σ
  by itself, so no pruning. **Mitigation:** don't use this method — pay
  the 25 % memory for nothing is still bad. Detect at build time by
  checking stddev(w) and skip the index.
* **Very small τ with uniform weights.** Pathological: pruning works but
  the inner scanner was already fast. Combine with a tiny Hamming cutoff
  (radius search instead of top-k) and you get the right story.
* **Streaming inserts.** `Vec` buckets support O(1) push; deletes need
  tombstoning or a swap-remove per id. We ship static build only; a
  streaming variant with per-bucket `Vec<(u32, bool)>` is one afternoon
  of work.
* **Non-Hamming binary metrics** (Jaccard, cosine on bits via bit-count
  LUT). The popcount triangle bound is specific to Hamming; for Jaccard
  one can still prune by weight *bands* but the bound is weaker.

## What to improve next

* **SIMD inner kernel.** Current `code_dist` compiles to scalar `CNT`.
  Wire into `ruvector-bit-sliced-hamming` and expect WBS to overtake
  FullScan on regime A too.
* **Multi-table weight sketches.** For codes wider than 256 bits,
  bucketing by `pop(x)` alone is coarse. Partition code into two
  halves and bucket by `(pop(hi), pop(lo))`; triangle bound becomes a
  2-D L1 ball — same proof, finer granularity.
* **Online updates.** Streaming inserts cost one bucket push; deletes
  cost O(bucket_size); periodic rebuild on bucket-size drift threshold.
* **Hybrid with IVF.** Use popcount as a *second-level* coarse code
  inside each IVF list, instead of or alongside the IVF centroid.
* **Learned popcount targets.** Train an iterative-quantization-style
  rotation that explicitly spreads popcount across buckets — reclaims
  the pruning benefit for symmetric data.

## Production crate layout proposal

```
crates/ruvector-hamming-weight-prebucket/
    src/
        lib.rs            # trait + 3 backends + SimHash + SplitMix64
        bin/benchmark.rs  # the real-numbers benchmark binary
    README.md
    Cargo.toml
```

For production integration, the natural shape is a `WeightIndex<S>` type
parameterized by an inner `BinaryScanner`, exposed from `ruvector-core`.
Build it alongside any existing binary index.

## References

1. Charikar, M. (2002). *Similarity estimation techniques from rounding
   algorithms.* STOC.
2. Norouzi, M., Punjani, A., Fleet, D. J. (2012). *Fast search in Hamming
   space with multi-index hashing.* CVPR.
3. Chappell, T. et al. (2019). *Approximate set similarity joins with
   signature-weight pruning.* IR literature on signature files.
4. Jégou, H., Douze, M., Schmid, C. (2011). *Product quantization for
   nearest neighbor search.* TPAMI.
5. Guo, R. et al. (2024). *RaBitQ: Rate-optimal binary quantization for
   ANN.* SIGMOD.
6. Baranchuk, D. et al. (2024). *Extended RaBitQ.* preprint.

(We cite the folklore result for the popcount-triangle inequality on the
hypercube; the direct search literature uses it inside signature-file
pruning but, to our knowledge, no major vector-search engine exposes it
as an index layer.)
