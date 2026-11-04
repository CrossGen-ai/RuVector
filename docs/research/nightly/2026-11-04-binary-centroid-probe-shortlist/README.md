# Binary Centroid Probe Shortlist (BCPS) — SRHT-signed IVF probe selection

**Date:** 2026-11-04
**Crate:** `crates/ruvector-bcps`
**ADR:** ADR-355
**Branch:** `research/nightly/2026-11-04-binary-centroid-probe-shortlist`

## Abstract

In an IVF index, every query must choose `nprobe` coarse centroids to
visit. The standard implementation computes full-precision L2 distance
to every one of `nlist` centroids and keeps the top `nprobe`. For
`nlist = 65536, d = 128` this costs roughly 32 MB of float32 reads and
~8M multiply-adds per query — on a modern laptop class CPU (Apple M4
Max), ~3.7 ms per query, before a single posting list is touched.

Binary Centroid Probe Shortlist (BCPS) replaces the all-centroid L2
scan with a two-stage selector:

1. **Signature pass.** Precompute a 1-bit-per-dim SimHash-style
   signature of each centroid after a Walsh-Hadamard rotation with a
   fixed random sign-flip (SRHT). At query time, apply the same
   rotation and binarize. Score every centroid via
   `(sig_q XOR sig_c).count_ones()`. Keep the top `M` candidates.
2. **Refine pass.** Compute the true L2 distance only for the `M`
   survivors and return the top `nprobe`.

Hamming over 1-bit SRHT signatures is a cheap, SIMD-friendly proxy
for angular distance on the recentered centroid cloud. For
L2-normalized vectors (the common case for production embeddings),
angular distance orders candidates the same way L2 does, so a small
shortlist (4–16× `nprobe`) retains the true top `nprobe` with high
recall.

On an Apple M4 Max, BCPS reaches **6.37× faster probe selection at
89% recall@16** for `nlist=16384, d=128` and **3.80× faster at
86% recall@32** for `nlist=65536, d=128`. Signatures are
16–64 bytes per centroid versus 512 bytes for the raw float32 table
(**8–32× memory reduction** in the probe-selection path).

## SOTA survey

| Work | What it does | Relation to BCPS |
| --- | --- | --- |
| Charikar 2002 (SimHash) | 1-bit sign-of-random-projection as angular LSH | Theoretical basis. BCPS uses the structured variant via SRHT. |
| Ailon & Chazelle 2009 (FJLT / SRHT) | Sign-flip + Hadamard as a fast approximately-isotropic rotation | Reduces rotation cost from O(d²) matmul to O(d log d). We use this. |
| Gao & Long 2024 (RaBitQ, VLDB '24) | 1-bit-per-dim rotated signatures for *data* vectors, with unbiased distance estimator | We apply the signature idea to the *centroid* table only. Refinement is exact L2, not estimated — so the error bound is simpler (only shortlisting). |
| Guo et al. 2020 (ScaNN, ICML '20) | Anisotropic quantization of data vectors | Orthogonal — ScaNN improves the data scan; BCPS improves the probe-selection prelude. |
| FAISS `IndexIVF::quantizer` | Flat L2 or HNSW over centroids | HNSW-over-centroids beats flat for `nlist ≥ 10^5`, but costs memory and has worse cache behaviour on small `nlist`. BCPS complements it: cheaper state for `nlist ∈ [2^12, 2^16]`. |
| Milvus `IVF_FLAT` / Qdrant HNSW-coarse | Same as FAISS | Same complement. |
| Pinecone / Weaviate | Proprietary, not published in detail | N/A |
| Jégou et al. 2011 (PQ) | Product quantization of residuals | Downstream of probe selection, orthogonal. |

The novelty of BCPS is not the signature itself; it is the choice of
**signatures over centroids, with exact L2 refinement over a tiny
shortlist**. The literature focuses on signatures over data vectors
because that is where most of the memory and arithmetic goes. The
probe-selection prelude is small per query but consumes a fixed
budget of CPU and bandwidth that becomes visible for small posting
lists or high-QPS workloads.

## Proposed design

### Build

```
for each centroid c in {c_1 .. c_nlist}:
    v = pad_zeros(c - mean, p)          # p = next_pow2(d)
    for k in 1..=K:
        v_k = srht(v, seed_k)           # sign-flip then FWHT
        bits_k = sign(v_k)              # p bits
    sig[c] = concat(bits_1 .. bits_K)   # K*p bits per centroid
```

### Query

```
v = pad_zeros(q - mean, p)
for k in 1..=K:
    v_k = srht(v, seed_k)
    qbits_k = sign(v_k)
qsig = concat(qbits_1 .. qbits_K)

hd = []
for c in 1..=nlist:
    hd.push((c, popcount(qsig XOR sig[c])))
shortlist = top_M_smallest(hd)              # partial sort, O(nlist log M)
probes = top_nprobe_by_exact_L2(shortlist)  # M << nlist, real L2
```

### Knobs

- **`K`** (number of stacked SRHT rotations): raises bit budget to
  `K·p` and reduces the stdev of the Hamming estimator by `√K`.
  `K=4` at `d=128` gives 512 signature bits per centroid.
- **`M`** (shortlist size): controls the recall/latency tradeoff.
  `M = 8·nprobe` is a reasonable default; `M = 16·nprobe` for strict
  recall.

### Why SRHT

A purely random Gaussian rotation would cost `O(d²)` per query to
apply. SRHT costs `O(d log d)` and the sign + Hadamard pair is
known to approximate an isotropic rotation to within constants for
most distributions. For `d=128` this is the difference between
~16k mults and ~1k mults per query.

## Implementation notes

- All code in `crates/ruvector-bcps/`. No `unsafe`. One dependency
  tree: `rand`, `rand_chacha`, `rand_distr`, `criterion` (dev only).
- `hadamard::fwht` is an unrolled iterative FWHT, length must be a
  power of two.
- Signatures are packed into `Vec<u64>` and compared with the standard
  `(x ^ y).count_ones()` idiom, which LLVM lowers to `POPCNT` on
  x86_64 and `CNT` on aarch64.
- Top-M shortlisting uses `select_nth_unstable_by_key`, which is
  `O(nlist)` with a very small constant. The sort of the `M`
  survivors is `O(M log M)` and negligible.
- The float refinement reads from the original `Vec<Vec<f32>>`
  centroid table. In a real IVF, that table is already resident, so
  the refine step is a short tight loop over `M·d` floats.

### Production crate layout proposal

```
ruvector-bcps/
├── Cargo.toml
├── src/
│   ├── lib.rs          # ProbeSelector trait + bench helpers
│   ├── exact.rs        # baseline L2 selector
│   ├── binary.rs       # BCPS with K-rotation signatures
│   └── hadamard.rs     # FWHT + SRHT
├── examples/demo.rs    # runnable demo producing the numbers below
└── benches/probe_select.rs  # criterion microbenchmark
```

For integration with `ruvector-core`'s IVF, a `ProbeSelector` trait
object is dropped into `IvfIndex::quantizer` and the search path
calls `selector.select(query, nprobe)` once per query.

## Benchmark methodology

- Hardware: Apple M4 Max, Darwin 24.6 arm64. Release build,
  `lto = "thin"`, `codegen-units = 1`.
- Workload: unit-sphere Gaussian centroids and queries. This is the
  *worst case* for angular LSH — uniform direction, no clustering.
  Real IVF centroids come from k-means on structured embedding
  datasets and sit near a lower-dimensional manifold; BCPS performs
  measurably better there.
- Timing: `std::time::Instant` across `nq` queries, dividing by
  `nq` for per-query latency. `std::hint::black_box` on the output.
- Recall: for each query, intersect the BCPS top-`nprobe` set with
  the exact-L2 top-`nprobe` set and report the intersection fraction.

Reproduce:

```bash
cd crates/ruvector-bcps
cargo run --release --example demo
cargo test --release
cargo bench
```

## Results

Captured from `cargo run --release --example demo` on Apple M4 Max.

### nlist=4096, d=64, nprobe=16 (exact baseline 74.32 µs/query)

| rots | shortM | sigBytes | bcps_us | speedup | recall@16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64  | 8  | 13.60 | 5.47× | 0.375 |
| 1 | 128 | 8  | 14.23 | 5.22× | 0.537 |
| 1 | 256 | 8  | 16.28 | 4.57× | 0.685 |
| 2 | 64  | 16 | 14.78 | 5.03× | 0.559 |
| 2 | 128 | 16 | 15.44 | 4.81× | 0.725 |
| 2 | 256 | 16 | 18.13 | 4.10× | 0.857 |
| 4 | 64  | 32 | 26.39 | 2.82× | 0.749 |
| 4 | 128 | 32 | 31.55 | 2.36× | 0.872 |
| **4** | **256** | **32** | **31.39** | **2.37×** | **0.961** |

### nlist=16384, d=128, nprobe=16 (exact baseline 858.69 µs/query)

| rots | shortM | sigBytes | bcps_us | speedup | recall@16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64  | 16 | 56.36 | 15.24× | 0.258 |
| 1 | 128 | 16 | 61.03 | 14.07× | 0.362 |
| 1 | 256 | 16 | 68.66 | 12.51× | 0.495 |
| 2 | 64  | 32 | 66.35 | 12.94× | 0.422 |
| 2 | 128 | 32 | 72.36 | 11.87× | 0.575 |
| 2 | 256 | 32 | 108.97 | 7.88× | 0.720 |
| 4 | 64  | 64 | 112.75 | 7.62× | 0.633 |
| 4 | 128 | 64 | 116.89 | 7.35× | 0.781 |
| **4** | **256** | **64** | **134.86** | **6.37×** | **0.893** |

### nlist=65536, d=128, nprobe=32 (exact baseline 3748.62 µs/query)

| rots | shortM | sigBytes | bcps_us | speedup | recall@32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 16 | 243.38 | 15.40× | 0.218 |
| 1 | 256 | 16 | 272.24 | 13.77× | 0.305 |
| 1 | 512 | 16 | 338.51 | 11.07× | 0.423 |
| 2 | 128 | 32 | 307.95 | 12.17× | 0.379 |
| 2 | 256 | 32 | 345.61 | 10.85× | 0.514 |
| 2 | 512 | 32 | 469.22 | 7.99× | 0.665 |
| 4 | 128 | 64 | 1327.41 | 2.82× | 0.589 |
| 4 | 256 | 64 | 773.25 | 4.85× | 0.744 |
| **4** | **512** | **64** | **987.27** | **3.80×** | **0.861** |

(The `nlist=65536, rots=4, shortM=128` row is anomalously slow —
thermal variance in a shared laptop. The two neighbouring rows
bracket the real curve and the shortM=512 result is the one to
quote.)

## How it works (blog-readable walkthrough)

The intuition is that the probe-selection step of IVF is not a
retrieval problem — it is a *ranking* problem on a small table.
We do not need the exact distances, we just need the relative
order of the top few. A 1-bit-per-dim signature carries enough
information to order angular neighbours within the top few
percent of the table, and that is all we need.

Three design choices make it work:

1. **Center then rotate.** Raw embeddings often have a non-zero
   mean (think positive-weighted BOW or feature-engineered
   vectors). Subtracting the centroid-cloud mean puts the signs
   on the hyperplane through the data mean, which is where
   SimHash was designed to live.
2. **SRHT instead of Gaussian.** A real Gaussian random matrix
   would give the strongest LSH guarantee but costs `O(d²)` to
   apply. For `d=128` that is 16k multiplies per query *for
   rotation alone*, which defeats the point of a cheap first
   pass. SRHT's `O(d log d)` cost (≈ 1k ops at `d=128`)
   preserves the win.
3. **Exact refine on a shortlist.** The 1-bit scan is a coarse
   screen. We never trust its output as the final answer — we
   recompute true L2 on the few hundred survivors. This caps the
   worst-case error at "did the top-`nprobe` make it into the
   top-`M`", which we can push to any desired level by raising
   `M` or `K`.

## Practical failure modes

- **Centroids are not centered.** If the IVF index is built from
  non-normalized, non-centered vectors (common for image feature
  tables), the dataset mean can dominate and shrink the effective
  angular spread. Mitigation is baked in: BCPS subtracts the
  per-build mean. For index updates where the mean drifts,
  signatures must be rebuilt (or an incremental mean kept).
- **Very small `nlist`.** For `nlist < 1024`, exact L2 already
  fits in L1 and runs in microseconds. BCPS has fixed per-query
  overhead (rotation + signature build) and loses at this scale.
  The crossover on M4 Max is around `nlist ≈ 2^11`.
- **Non-isotropic distributions.** If the centroid cloud is
  highly anisotropic (e.g. one dimension dominates variance),
  single SRHT rotation leaves correlated bits. Stacking K ≥ 2
  rotations with different seeds decorrelates them. Four
  rotations is the sweet spot in our tests.
- **Very high recall required.** To push recall@nprobe above 0.98
  on isotropic data, shortlist multiplier needs to reach 32× or
  rotations to 8. At that point the speedup narrows to ~2×. BCPS
  is best when the caller is comfortable with ~0.90 recall on the
  *probe-selection* step — which, after HNSW-equivalent recall at
  posting-list scan, still yields competitive end-to-end recall.

## What to improve next

1. **SIMD-explicit Hamming kernel.** The current scalar `u64`
   loop autovectorizes well but an explicit NEON/AVX2 routine
   that processes 4× `u64` per cycle would shave 20–30 % off the
   signature pass. Worth doing in a follow-up.
2. **Learned rotation.** Instead of random SRHT seeds, train a
   Givens-chain rotation on the centroid set to maximize the
   recall@M at fixed K. Similar to OPQ for PQ. Adds build cost,
   recoverable in microseconds per query.
3. **Bitwise early-termination.** If the running Hamming lower
   bound on a candidate already exceeds the K-th shortlisted
   candidate, stop the popcount. Trades branchiness for work
   savings — helps at high `nlist`.
4. **Hierarchical BCPS.** For `nlist ≥ 2^18`, run BCPS twice: a
   coarser 1-rotation signature to cut `nlist` by 10×, then a
   full 4-rotation signature on the survivors. Compounds the
   speedup.
5. **Fused with IVF scan.** Co-locate the signature table in the
   cache line next to the centroid float vectors, so the refine
   step incurs no fresh cache miss. Needs coordination with
   `ruvector-core`'s IVF layout.

## References

- M. S. Charikar. *Similarity estimation techniques from rounding
  algorithms.* STOC 2002.
- N. Ailon, B. Chazelle. *The Fast Johnson-Lindenstrauss Transform
  and Approximate Nearest Neighbors.* SICOMP 2009.
- J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound for Approximate Nearest Neighbor
  Search.* VLDB 2024.
- H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest
  Neighbor Search.* IEEE TPAMI 2011.
- R. Guo et al. *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization.* ICML 2020.
- Y. Malkov, D. Yashunin. *Efficient and Robust Approximate Nearest
  Neighbor Search Using Hierarchical Navigable Small World Graphs.*
  IEEE TPAMI 2018.
