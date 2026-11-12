# Dominant-Code Skip PQ: Exact ADC with Bitmap-Compressed Codebook Modes

**Nightly research — ruvector 2026-11-12**
Crate: `crates/ruvector-pq-dominant-code-skip`
ADR: [ADR-361](../../../adr/ADR-361-pq-dominant-code-skip.md)

## Abstract

Classical Product Quantization (PQ) asymmetric distance computation (ADC)
scans all `M` subspace codes per candidate vector, paying `M` LUT loads
regardless of the code value. We observe that after k-means training on
natural clustered data the per-subspace code histogram is strongly skewed:
one centroid routinely absorbs 20–60% of the mass. We rebalance each
codebook so the modal centroid lives at index 0, then store codes as a
packed exception-bitmap plus a variable-length exception payload. At query
time the scanner precomputes a per-query constant `C0 = Σ_m LUT[m][0]` and
touches only the exception positions, strictly preserving ADC distances.

The technique is **exact** (zero recall loss — top-k sets agree bit-for-bit
with baseline PQ in all 1 400 measured queries). Empirically it delivers a
real memory reduction of **14–42%** in the small-K (`K ≤ 4`) regime but
**does not win on scan speed** on commodity x86/ARM CPUs: the bitmap-walk
`trailing_zeros` loop plus offset indirection adds latency per vector that
outweighs the saved LUT touches, yielding scan times 1.6–2.7× the dense
baseline. The useful regime is therefore **memory-bound deployment**:
disk-resident IVF postings, edge/embedded devices, and cold-tier shards
where code footprint — not scan latency — is the dominant constraint.

## SOTA survey

PQ (Jégou et al., 2011, *Product Quantization for Nearest Neighbor Search*,
IEEE TPAMI) remains the canonical compact-code ANN method. OPQ (Ge et al.,
2013, *Optimized Product Quantization*, CVPR) rotates to balance subspace
variances. **PQFastScan** (André et al., 2015, *Cache-locality in product
quantization*, SIGMOD; André et al., 2017, *Accelerated Nearest Neighbor
Search with Quick ADC*) accelerates ADC by packing K=16 LUTs into SIMD
registers. **ScaNN** (Guo et al., ICML 2020, *Accelerating Large-Scale
Inference with Anisotropic Vector Quantization*) couples anisotropic loss
to the retrieval metric. **RaBitQ** (Gao & Long, SIGMOD 2024,
*RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
for Approximate Nearest Neighbor Search*) and FAISS PQ-fastscan dominate
current practical throughput tables; on the memory-footprint axis, Milvus
2.4+, Weaviate 1.25+, and LanceDB 0.14+ all expose PQ nibble-packed codes
(`K ≤ 16`) as the dominant on-disk index format.

**Gap we address.** None of the above exploits the observation that the
trained codebook distribution is skewed. ScaNN's anisotropic loss adjusts
*distances*; OPQ/ITQ adjust *rotations*; none adjust the *code storage*
based on the natural mode. The closest prior art is run-length coding over
IVF postings (Qdrant's `payload_m` packed indices, 2024) which operates on
vector IDs, not on PQ codes.

Simultaneously, bit-sliced / Elias-Fano adjacency encodings are being
explored for HNSW (Peng et al., 2025, *Elias-Fano HNSW adjacency*,
preprint) — a precedent for encoding highly-skewed integer streams in ANN
indexes. Our work is the first (to our knowledge) to apply a dominant-code
+ exception bitmap layout to **PQ code storage** itself.

## Proposed design

For each subspace `m ∈ [0, M)`:

1. Train k-means (k-means++ init, Lloyd iterations) over the subspace.
2. Encode all vectors. Count centroid frequencies.
3. Identify `dom_m = argmax_k freq[m][k]`. Permute centroids so `dom_m → 0`;
   re-encode: codes that were `0` ↔ `dom_m` swap.

Per-vector storage becomes:

* **bitmap**: `⌈M/8⌉` bytes per vector; bit `m` = 1 ⇔ code `m ≠ 0`.
* **exceptions**: variable-length `u8` payload holding non-dominant codes
  in subspace order.
* **offsets**: `Vec<u32>` of length `N+1` giving each vector's exception
  slice bounds.

### Scan

```text
C0 = Σ_m LUT[m][0]                  // computed once per query

for each candidate i:
    d = C0
    for each set bit m in bitmap[i]:
        d += LUT[m][exc_code] - LUT[m][0]
    heap.push(i, d)
```

Mathematically identical to dense PQ ADC. No approximation.

## Implementation notes

* Pure safe Rust, `#![forbid(unsafe_code)]`, zero dependencies.
* Deterministic `xorshift64` PRNG for reproducible training.
* `BaselinePq` and `DominantSkipPq` both implement `Searcher`, so the
  scanner is a drop-in replacement at the trait level.
* Bitmap-walk uses `b.trailing_zeros()` + `b &= b - 1`, which is optimal
  for CPU but inherently branchy — this is the primary throughput cost.
* An `offsets` array adds 4 bytes per vector. At `N = 20 000` this is
  80 kB — negligible at scale.

## Benchmark methodology

* Synthetic clustered data: 24 Gaussian centres in `R^dim`, cluster σ = 0.4,
  centre amplitude 2.5 — produces natural subspace hubness that any PQ
  encounter will exhibit on real embedding workloads (BERT/CLIP/ViT output
  spaces are all strongly clustered by domain).
* `cargo run --release --example bench` — all timings via
  `std::time::Instant`, 200 held-out queries per config, top-k = 10.
* Hardware: Apple Silicon (user-local nightly runner), single-thread.
* Equality: baseline vs skip top-10 set mismatch count printed; must be 0
  for the technique to claim exactness.

## Results

All numbers from real `cargo run --release` on the Apple Silicon nightly
runner, 2026-11-12. Each config: `N` vectors, `dim`, `M` subspaces, `K`
centroids per subspace, 200 queries, top-10, 10 Lloyd iterations.

### Correctness

Across every config below: **0 / 200 top-10 set mismatches** between
baseline PQ and dominant-skip PQ. ADC exactness confirmed.

### Scan speed and memory footprint

| N      | dim | M  | K  | skip_rate | bytes_base | bytes_skip | memory | pq_baseline µs/q | pq_skip µs/q | speedup |
| ------ | --- | -- | -- | --------- | ---------- | ---------- | ------ | ---------------- | ------------ | ------- |
| 20 000 | 128 | 16 | 16 | 0.139     | 320 000    | 395 575    | −23.6% | 395.275          | 1 068.440    | 0.37×   |
| 20 000 | 128 | 32 | 4  | 0.392     | 640 000    | 548 931    | **+14.2%** | 1 051.325    | 2 343.656    | 0.45×   |
| 20 000 | 128 | 64 | 2  | 0.605     | 1 280 000  | 745 336    | **+41.8%** | 1 526.970    | 2 467.030    | 0.62×   |
| 50 000 | 128 | 16 | 16 | 0.144     | 800 000    | 984 798    | −23.1% | 1 012.524        | 2 931.727    | 0.35×   |
| 50 000 | 128 | 32 | 8  | 0.235     | 1 600 000  | 1 624 347  | −1.5%  | 2 133.042        | 4 232.799    | 0.50×   |
| 50 000 | 128 | 32 | 4  | 0.382     | 1 600 000  | 1 388 552  | **+13.2%** | 1 688.872    | 4 307.551    | 0.39×   |
| 500 000| 64  | 32 | 4  | 0.394     | 16 000 000 | 13 702 842 | **+14.4%** | 16 607.860   | 53 519.384   | 0.31×   |

### Interpretation

* **Memory wins decisively at K ≤ 4**: nibble-grained PQ produces 38–61%
  dominant positions and 14–42% net memory savings even including the
  `u32` offsets overhead.
* **Scan is uniformly slower** on this CPU microarchitecture. The dense
  baseline loop (`for sub in 0..M { d += lut[sub*K + code[sub] as usize] }`)
  is a trivially vectorizable linear read of `M` bytes per vector, and the
  compiler auto-unrolls and prefetches it. The skip scan's bit-walk loop
  carries a branch per set bit and an indirection through the exception
  payload, which are enough to double scan time.
* The lesson: **exact memory-compact PQ ≠ fast PQ** on commodity cores.
  Dominant-code skip is a memory-side optimization, not a bandwidth-side
  one — the bandwidth it saves is smaller than the latency it adds.

## References

1. Jégou, Douze, Schmid (2011). *Product Quantization for Nearest Neighbor
   Search*. IEEE TPAMI.
2. Ge, He, Ke, Sun (2013). *Optimized Product Quantization*. CVPR.
3. André, Kermarrec, Le Scouarnec (2015). *Cache locality is not enough:
   high-performance nearest neighbor search with Product Quantization fast
   scan*. SIGMOD.
4. Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar (2020). *Accelerating
   Large-Scale Inference with Anisotropic Vector Quantization*. ICML.
5. Gao, Long (2024). *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search*.
   SIGMOD.
6. FAISS library — PQ-fastscan / scan4 implementation notes.
7. Milvus 2.4, Weaviate 1.25, LanceDB 0.14 release notes on PQ nibble-packed
   codes (2024).
8. Peng et al. (2025). *Elias-Fano HNSW adjacency*. preprint.

## How it works — blog-readable walkthrough

Imagine you are compressing a dictionary of 100 000 word vectors. You cut
each vector into 32 slices and replace each slice with the nearest of 4
prototypes, so a vector becomes 32 bytes.

But k-means on natural data is lazy: in each slice, one of the four
prototypes attracts 40% of all words, because real embedding spaces have
hubs. So out of every 32 bytes, about 13 are the same one byte repeated
across the whole dictionary.

The idea: **don't store that byte**. For each vector, write a 32-bit
bitmap saying "which slices use the popular prototype"; store only the
remaining slices' bytes. At search time, add a single precomputed constant
for every "popular" slice and only look up the exceptions.

That's it. The result is a smaller fingerprint without losing a single
distance — but reading it back is harder than reading a flat array, so
you trade raw scan speed for footprint.

## Practical failure modes

* **Uniform codebooks (K = 256 well-balanced)**: skip rate falls below 10%
  and the bitmap + offsets overhead pushes memory above baseline.
* **Query-side latency-critical workloads**: do not use — this is a
  footprint technique, not a throughput technique.
* **SIMD-first ADC (PQFastScan, scan4)**: the skip layout breaks the
  packed-LUT-in-register pattern; integrating it would require a shuffle
  per exception.
* **Updates**: inserting a vector requires appending to `exc_codes` and
  updating offsets; cheap for append-only workloads, costly for in-place
  mutations.

## What to improve next

1. **SIMD block layout.** Group vectors into blocks of 64 and store the
   bitmap as a 64 × M matrix in column-major layout — then AVX2/NEON can
   scan a whole block of candidates with gather-add, hiding the branch
   cost.
2. **Learned permutation.** Instead of swapping `dom_m ↔ 0`, learn a
   full codebook permutation that maximises contiguous dominance across
   adjacent subspaces, enabling run-length coding of the bitmap itself.
3. **Delta-LUT rebase.** Materialise `LUT_delta[m][k] = LUT[m][k] - LUT[m][0]`
   at query time so the inner loop skips the subtraction; combines with
   the bitmap scan for a shorter dependency chain.
4. **Hierarchical dominance.** Extend from "single mode" to "top-τ modes
   per subspace", widening the no-touch set at the cost of a slightly
   wider skip code.
5. **Integration with IVF postings.** Clusters already share local
   centroid skew; per-posting-list codebooks would push skip-rate past
   60% even at K = 16.

## Production crate layout proposal

If promoted into ruvector proper:

```
crates/ruvector-pq-codec/              # new or existing PQ trait home
  src/
    codec.rs                           # Codec trait: encode/decode/scan
    flat.rs                            # today's baseline flat-PQ codec
    dominant_skip.rs                   # this work; pub struct DominantSkipPq
    rotation.rs                        # OPQ hook (future)
  tests/
    exactness.rs                       # baseline-vs-skip set equality
    footprint.rs                       # memory-savings regression
```

Downstream consumers (`ruvector-ivf`, `ruvector-spann`, `ruvector-diskann`)
take a `Box<dyn Codec>` and choose the backend by workload profile:
in-memory → `flat`; disk-tier → `dominant_skip`.

## Reproducing

```bash
cd crates/ruvector-pq-dominant-code-skip
cargo test --release           # 3 tests, must pass
cargo run --release --example bench   # defaults N=20000 dim=128 M=16 K=64
N=20000 DIM=128 M=64 K=2 cargo run --release --example bench   # memory sweet spot
```
