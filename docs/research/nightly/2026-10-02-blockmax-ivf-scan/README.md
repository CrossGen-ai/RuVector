# Block-Max Scanning for IVF-PQ Posting Lists

**Date:** 2026-10-02
**Branch:** `research/nightly/2026-10-02-blockmax-ivf-scan`
**Crate:** `crates/ruvector-blockmax-ivf`
**ADR:** [ADR-353](../../../adr/ADR-353-blockmax-ivf-scan.md)

## Abstract

We port the Block-Max WAND (BMW) technique from text retrieval to IVF-PQ
vector search. The hypothesis: if posting lists are chunked into fixed-size
blocks with a precomputed per-block lower bound on asymmetric-distance
(ADC) scores, a scanner can skip whole blocks whose bound cannot improve the
current top-K. We implement three bound variants — heuristic (unsound),
per-query tight-min, and bitmask-accelerated tight-min — and benchmark
against a sequential IVF-PQ scanner.

Headline finding: **BMW does not port cleanly to PQ at common parameter
settings.** PQ centroid IDs are not distance-sortable, so the single
precomputed per-subspace "min code" is a wildly loose lower bound against
most queries, meaning any precomputed bound either (a) is unsound and
destroys recall, or (b) is sound but costs as much to compute per query as a
small fraction of the full scan it is trying to prune. We measure the
regime, publish real numbers, and sketch two follow-ups that could rescue
the approach (sub-blocks and lower-ksub residual codebooks).

## SOTA survey

- **Block-Max WAND (Ding & Suel, 2011).** Standard IR technique: inverted
  posting lists are split into blocks; each block carries an upper bound on
  per-term score; the WAND query processor uses these to skip blocks that
  cannot enter the top-K. Variants: BMM, VBMW. The idea has been ported to
  learned sparse retrievers (SPLADE serving).
- **Anytime / early-termination IVF-PQ** (Johnson et al. 2017, FAISS).
  Classic early-abandon: break out of the per-vector ADC inner loop when
  partial sum exceeds the K-th-best. This is the *vector-level* analogue of
  what we propose at the *block-level*.
- **Partial-L2 early-abandon** (ruvector, 2026-09-20) — vector-level
  early-abandon with cumulative subspace sums; the same mathematical idea at
  one granularity level down.
- **PDX / vertical PQ scan** (ruvector 2026-09-14) — subspace-major layout
  that trades scalar cache behavior for SIMD-friendly fast-scan; orthogonal
  to block skipping.
- **BMW-ANN (Mackenzie et al. 2022)** experiments with hybrid sparse-dense
  retrieval but keeps the dense side opaque; they do not block-prune on PQ
  codes.

To our knowledge there is no published Block-Max variant for pure IVF-PQ
posting lists. The reason — as our experiments show — is that the technique
fits inverted indexes because term scores *per document* are themselves
ordered (or at least representative of magnitudes), while PQ codes are
centroid identifiers whose ordering is arbitrary.

## Proposed design

### Index layout

* Standard IVF with k-means coarse centroids (`nlist` lists).
* Standard product quantization over *residuals* (`m` subspaces, `ksub=256`
  centroids per subspace, 1-byte codes).
* Each posting list is split into fixed-size *blocks* of `B` codes.
* **Per-block metadata.** We store two auxiliary structures:
  1. `lb_codes[j]`: the elementwise minimum code value (as a `u8`) seen in
     the block for subspace `j`. Used by the heuristic variant.
  2. `code_masks[j]`: a 256-bit bitmask of which code IDs appear in the
     block for subspace `j`. Used by the bitmask-tight variant.

Index-time storage overhead per block (m=16): 16 bytes for `lb_codes`,
512 bytes for `code_masks`. For a block of 128 codes × 16 bytes = 2048
bytes of actual payload, metadata is 25% overhead.

### Scan backends

For a query `q`, probing coarse cell `c`:

1. Build the standard ADC LUT: `tab[j][k] = ||q_sub_j − c_sub_j − cb[j][k]||^2`.
2. For each block in the posting list:
   * Compute a lower bound `lb(block)` on ADC distance for any code in the
     block. Three variants (below).
   * If `lb(block) >= topk.threshold()`, skip the entire block (no code
     evaluations).
   * Otherwise, evaluate every code in the block normally and push into the
     top-K heap.

**Variant A — heuristic `lb` (`BlockMaxScanner`).** Use the precomputed
`lb_codes`:  `lb = Σ_j tab[j][lb_codes[j]]`. Zero per-block overhead but
*unsound*: `lb_codes[j]` is just the min integer code value, not the code
whose distance under `tab[j]` is minimal.

**Variant B — tight per-query `lb` (`BlockMaxTightScanner`).** For each
subspace, scan every code in the block, look it up in `tab[j]`, and take
the minimum. Sum across subspaces gives the exact per-block lower bound.
Sound, provably correct — but the per-block cost is `m·B` LUT lookups,
exactly half the cost of the full ADC scan it tries to replace.

**Variant C — tight-bitmask `lb` (`BlockMaxBitmaskScanner`).** Use
`code_masks` to iterate only *unique* codes per subspace. Cost per block is
`Σ_j popcount(code_masks[j])` LUT lookups. In regimes where codes are
repeated within a block, this is strictly cheaper than Variant B while
giving the identical tight bound.

## Implementation notes

* All three scanners share the same `Scanner` trait and the same `IvfPq`
  index. Switching between them is one line.
* The `ScanStats` struct tracks blocks total, blocks skipped, and codes
  evaluated — ensures reported "skip rate" is honest and reproducible.
* The `TopK` is a tiny fixed-capacity max-heap; `threshold()` returns +∞
  until K hits are collected, so no spurious early skipping.
* The crate is a standalone workspace to avoid pulling the full ruvector
  monorepo build graph for a focused PoC. All deps are pure-Rust (`rand`,
  `rand_chacha`).

## Benchmark methodology

* **Hardware:** Apple M-class laptop, release build (`cargo run --release
  --example bench`), single-threaded.
* **Data:** two generators — uniform random `[-1, 1]` and Gaussian-mixture
  clustered (32 centers, spread 0.4).
* **Metric:** recall@10 vs. exact-L2 top-10 ground truth from brute-force
  scan over raw vectors (same seed as training set).
* **Shapes:** four configurations varying n, dim, block_size, clustering,
  and nprobe. Each shape runs 200–300 queries after a 5-query warmup.
* **PQ:** `m=8..16`, `ksub=256`, k-means with 15 iterations.
* All numbers are from a single real run; no synthetic averaging.

## Results

```
=== shape-small-random :: n=10000 dim=64 nlist=64 nprobe=8 m=8 block=64 ===
  sequential       time=0.018s qps= 11421.7 r@10=0.248 codes/q= 1253 skipped=  0.0%
  blockmax-heur    time=0.018s qps= 11158.5 r@10=0.076 codes/q=  145 skipped= 89.3%
  blockmax-tight   time=0.018s qps= 10867.5 r@10=0.248 codes/q= 1253 skipped=  0.7%
  blockmax-bitmask time=0.019s qps= 10268.0 r@10=0.248 codes/q= 1253 skipped=  0.7%

=== shape-medium-random :: n=50000 dim=128 nlist=128 nprobe=16 m=16 block=128 ===
  sequential       time=0.110s qps=  2732.9 r@10=0.173 codes/q= 6253 skipped=  0.0%
  blockmax-heur    time=0.095s qps=  3143.9 r@10=0.025 codes/q=  280 skipped= 95.8%
  blockmax-tight   time=0.118s qps=  2533.5 r@10=0.173 codes/q= 6253 skipped=  0.4%
  blockmax-bitmask time=0.124s qps=  2414.4 r@10=0.173 codes/q= 6253 skipped=  0.4%

=== shape-medium-clustered :: n=50000 dim=128 nlist=128 nprobe=16 m=16 block=128 ===
  sequential       time=0.107s qps=  2810.5 r@10=0.430 codes/q= 6560 skipped=  0.0%
  blockmax-heur    time=0.094s qps=  3178.3 r@10=0.126 codes/q=  204 skipped= 97.3%
  blockmax-tight   time=0.122s qps=  2457.2 r@10=0.430 codes/q= 6055 skipped=  9.2%
  blockmax-bitmask time=0.125s qps=  2392.1 r@10=0.430 codes/q= 6055 skipped=  9.2%

=== shape-clustered-nprobe32 :: n=50000 dim=128 nlist=128 nprobe=32 m=16 block=128 ===
  sequential       time=0.212s qps=  1415.7 r@10=0.430 codes/q=12894 skipped=  0.0%
  blockmax-heur    time=0.186s qps=  1616.8 r@10=0.126 codes/q=  204 skipped= 98.6%
  blockmax-tight   time=0.236s qps=  1272.0 r@10=0.430 codes/q=10112 skipped= 23.9%
  blockmax-bitmask time=0.244s qps=  1231.3 r@10=0.430 codes/q=10112 skipped= 23.9%
```

### What the numbers say

* **Heuristic is unsound.** It skips 89–98 % of blocks and drops recall from
  0.17–0.43 down to 0.025–0.13 — the stored min-code-value does not
  correspond to the min-distance code because PQ IDs are arbitrary.
* **Tight bound is correct.** Both tight variants return byte-identical
  top-K to the sequential scan on every tested query (verified in
  `tests/correctness.rs`).
* **Pruning rate depends on data distribution and nprobe.** 0.4 % on
  uniform data, 9.2 % on clustered data at nprobe=16, 23.9 % at nprobe=32.
  More probing → tighter threshold earlier → more skipping.
* **The pruning does not translate to speed.** On these shapes the tight
  variants are 7–11 % *slower* than sequential because the per-block
  lower-bound evaluation (`m·B` or `Σ popcount` LUT lookups per block)
  costs roughly as much as scanning the block it prunes.
* **Bitmask does not beat per-query min** at `B=128`, `ksub=256` because
  nearly every centroid gets used in a block of 128 points (popcount ≈ B).

## How it works — walkthrough

Imagine you are searching a stack of filing cabinets. Each cabinet is a
*posting list* (coarse cell). Each cabinet is divided into *drawers* of 128
folders (blocks). Each folder has a product-quantization-coded summary
inside: 16 bytes, each byte a centroid ID in `[0, 255]`.

The baseline scan opens every folder in every probed cabinet, computes a
distance from the summary to the query, and keeps the K smallest. Our
block-max idea: on the outside of each drawer, write a *lower bound* on
how close any folder inside could be. If that lower bound is already worse
than what we have, skip the drawer entirely.

The trick is: what do we write on the drawer? The naïve answer — "the
smallest centroid ID per subspace that appears inside" — turns out to be
useless, because PQ centroid IDs are not sorted by distance to anything.
The right answer — "the smallest value of `tab[subspace][k]` over all `k`
that appear in this drawer" — can only be computed *per query*, because
`tab` depends on the query. And computing it is almost as expensive as
just reading every folder.

That is the central result of this nightly: the beautiful IR technique
does not translate because the underlying ordering assumption breaks.

## Practical failure modes

* **Random/uniform data:** pruning rate stays near zero because the top-K
  threshold moves very slowly on uniform data — most blocks look equally
  bad.
* **Small block sizes (B ≤ 32):** per-block LB overhead per code grows,
  making tight bounds strictly a loss.
* **Large ksub (256):** bitmasks are saturated, so bitmask variant
  degenerates to per-query min variant.
* **Low recall operating points** (nprobe ≪ nlist): the threshold stays
  loose for too long and most blocks have `lb < threshold`.

## What to improve next (roadmap)

1. **Sub-blocks.** Split each 128-point block into 4×32 sub-blocks and
   store a bitmask per sub-block. Within a block, scan sub-block LBs
   first, prune at that granularity.
2. **Residual low-ksub codes.** With `ksub=16` the bitmasks are 16 bits
   each and typically 4–8 bits set in a 32-point block; now
   `Σ popcount ≪ B` and the bitmask variant should actually pay off.
3. **Code reordering.** During index build, cluster codes within a block
   by first subspace so the per-subspace unique-count collapses to ≪ B.
4. **SIMD min-reduce** of the per-query tight bound: with `m=16`, 4-way
   AVX2 can compute the min over a 128-element block in ~32 cycles,
   finally making the sound variant cheaper than the scan it replaces.
5. **Combine with partial-L2 early-abandon.** Block-level LB first, then
   code-level cumulative-sum early-abandon. Compounding effect.

## Production crate layout (proposed)

If we promote this to a production crate:

```
crates/ruvector-blockmax/
  Cargo.toml                     # workspace member
  src/
    lib.rs                       # re-exports
    bounds/
      mod.rs                     # trait LowerBound
      heuristic.rs               # Variant A (keep for diagnostics)
      per_query_min.rs           # Variant B
      bitmask.rs                 # Variant C
      sub_block_bitmask.rs       # Variant D (roadmap item 1)
      simd.rs                    # AVX2/NEON reductions
    layout/                      # block layouts, metadata serialization
    scan.rs                      # trait Scanner, generic over LowerBound
    bench/                       # criterion harness
  tests/correctness.rs
  benches/bmw.rs
```

Hook into existing `ruvector-pq-search` behind a feature `blockmax` so
callers opt in; keep sequential scan the default.

## References

- Shuai Ding, Torsten Suel. *"Faster Top-k Document Retrieval Using Block-
  Max Indexes."* SIGIR 2011.
- Jeff Johnson, Matthijs Douze, Hervé Jégou. *"Billion-scale similarity
  search with GPUs."* IEEE TBD, 2017 (FAISS; discusses early-abandon
  in ADC scans).
- Antonio Mallia et al. *"Faster BlockMax WAND with Longer Skipping."*
  ECIR 2017.
- ruvector internal: `gist-partial-l2-early-abandon-2026.md` — the
  vector-level analogue to this proposal.
- ruvector internal: `gist-pq-fastscan-lut16-2026.md` — the orthogonal
  vectorized-scan path.
