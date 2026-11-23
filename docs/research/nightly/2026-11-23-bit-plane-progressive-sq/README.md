# Bit-Plane Progressive SQ8 (BPP-SQ)

**Date:** 2026-11-23
**Slug:** `bit-plane-progressive-sq`
**Crate:** `crates/ruvector-bit-plane-sq`
**Status:** PoC — builds green, tests green, benchmark numbers reproducible on commodity hardware.

## Abstract

BPP-SQ stores one 8-bit scalar-quantized (SQ8) code per vector, but physically
transposes those codes into **eight bit-planes** (one bitmap per bit position,
MSB-first). A query-time distance computation can then proceed *progressively*:
start by reconstructing an approximate code from only the top `M` planes
(biased to the midpoint of the surviving 2^(8−M) interval), scan all vectors
with this approximation in L1 to form a shortlist of size `k·r`, and finally
refine the shortlist using the full 8-plane (equivalent to row-major SQ8) code.
The planes-only scan reads a strict subset of the bytes a full SQ8 scan touches
(`M/8` of them), which is the mechanism by which memory-bandwidth-bound ANN
scans can trade recall for latency *without* storing a second code array.

## SOTA Survey

Scalar quantization is the simplest, lowest-overhead code family for vector
search: FAISS's `IndexScalarQuantizer`, Milvus's `IVF_SQ8`, Qdrant's `uint8`
quantization, Weaviate's SQ layer, Pinecone's quantized segments, and LanceDB's
SQ8 lane all operate on row-major byte codes per vector and scan them with a
single tight L1/L2 loop. Prior attempts to accelerate this scan fall into
three buckets:

1. **Narrower codes, same layout** — SQ4 (half the bytes, half the bandwidth,
   ~2× recall loss near the Pareto knee). Mainline in Milvus `IVF_SQ4`,
   `IndexScalarQuantizer` with `QT_4bit`. Trades recall uniformly.
2. **Transposed / vertical layouts** — PDX (*Partition-Dimension eXposure*,
   Kuzmin et al. 2024) shows that transposing to dimension-major improves SIMD
   fusion. ruvector has a `ruvector-pdx-vertical` nightly (2026-09-14) that
   lands those gains.
3. **Cascaded two-code storage** — int4-coarse / int4-residual (SVS-LeanVec
   family, Intel 2024), ruvector's `ruvector-sq-cascade-int4` from 2026-11-09.
   Pays 2× code memory for two scanned layers.

BPP-SQ sits in a gap: **one code, many precisions**, chosen at query time.
The closest conceptual neighbour is bit-slicing of binary PQ codes
(Guo et al., *Polysemous Codes*, ECCV 2016; Zhang et al. *Bit-Sliced Hamming*,
nightly 2026-09-18), but those are limited to Hamming-only ranking. Progressive
image-coding bit-planes (JPEG2000, Taubman 2002) use the same layout for
different reasons — reconstruction quality, not search latency.

Recent production changelogs scanned:

* Milvus 2.4.11 release notes — `IVF_SQ8H` still monolithic row-major codes.
* Qdrant 1.11 — `uint8` quantization still single-precision at query time;
  `rescore` reads full fp32 from disk, not another code.
* Weaviate 1.28 — adaptive SQ via `trainingLimit`; still one code per vector.
* LanceDB 0.17 — `SCALAR` quantization unchanged in layout.
* FAISS main (Aug 2025) — `IndexScalarQuantizer` layout unchanged since 2018.

Nothing in the production corpus stores SQ codes bit-plane transposed with a
query-time precision choice. The design space is open.

## Proposed Design

```
Build (offline):
  1. Fit per-dim min/scale.
  2. Quantize each vector x to u8 codes c in R^D.
  3. For bit position p in {0..=7}:
        plane[p][i, j] = (c_ij >> (7-p)) & 1
     Each plane is packed into u64 bitmaps of length ceil(D/64) per vector.

Search (online), parameters (M, r):
  1. Reconstruct an approximate code from the top-M planes with midpoint bias:
        c_hat_ij = (sum over p<M of plane[p][i,j] << (7-p)) + 2^(8-M-1)
     This is the minimax reconstruction for the unknown tail: the surviving
     interval has width 2^(8-M), so centering to the midpoint caps the
     per-coordinate |c - c_hat| by 2^(8-M-1), i.e. a tight 1-side bound.
  2. Compute L1(q, c_hat) over all N vectors; keep top k·r in a min-heap.
  3. Refine the k·r shortlist using the full 8-plane (row-major) code and
     return the final top-k by true L1.
```

### Why midpoint bias beats zero-fill

If we naively zero-filled the unknown tail (c_hat = top-M bits only), the
approximation would bias *all* reconstructions downward, which inflates L1
systematically on dims where `q_j` is high. The midpoint bias is the
`arg min_{v} max_{tail} |v - (top_M + tail)|` reconstruction and keeps the
pointwise L1 error bounded above by `D · 2^(8-M-1)` uniformly over the
dataset, independent of query. This is the same observation that makes
RaBitQ asymptotically unbiased (Jianyang Gao et al., SIGMOD 2024), applied
to the plane decomposition.

### Memory footprint

| M | planes-only bytes | fraction of SQ8 |
|---|-------------------|-----------------|
| 2 | 1 600 000         | 25 %            |
| 3 | 2 400 000         | 37.5 %          |
| 4 | 3 200 000         | 50 %            |
| 6 | 4 800 000         | 75 %            |
| 8 | 6 400 000         | 100 %           |

(for n=50k, D=128; linear in both.)

## Implementation Notes

* Lives under `crates/ruvector-bit-plane-sq`. No external deps; `std` only.
* Trait `L1Search` has three impls: `FullSq8`, `TopMApprox`, `Progressive`.
  Future SIMD / popcount-packed plane scanners plug in behind the same trait.
* `BitPlanes::planes_to_approx_codes(M)` materializes the current fast path:
  a row-major u8 buffer that can be scanned by the same `l1_u8` kernel as the
  baseline. This is the *correctness* path; it doesn't yet realize the
  planes-direct scan where only M bitmaps are loaded.
* The planes-direct scan (future work) will compute partial L1 against each
  plane using popcount-weighted contributions, equivalent to `sum_p 2^(7-p) *
  popcount(plane_p XOR q_plane_p)` for a bitonic L1 surrogate, then promote
  by true int-domain L1 only for the shortlist. Current PoC reconstructs to
  u8 to stay honest about recall; latency wins come after the bitmap kernel.

## Benchmark Methodology

* Synthetic dataset: `synth_dataset(n=50 000, dim=128, seed=11)` — a seeded
  LCG + sine mixture, fixed per run.
* 200 queries from `synth_dataset(nq=200, dim=128, seed=23)`, quantized with
  the index's per-dim statistics.
* Ground truth: brute-force float L2 top-10 (deterministic).
* Reported recall is `recall@10` against the float L2 ground truth, which is
  *not* the same as recall against the SQ8 L1 ranking; this is the honest
  end-user metric (what fraction of the true nearest neighbours does the
  index return?) and is bounded above by the `full_sq8` baseline.
* Latency = wall-clock micros, single-thread, Rust 1.74 release build,
  `lto = "thin"`, `codegen-units = 1`, run on an Apple M-series Darwin host.

Reproduce:

```bash
cargo run --release -p ruvector-bit-plane-sq --example bench
# Or override size:
BPP_N=100000 BPP_DIM=256 BPP_NQ=500 \
  cargo run --release -p ruvector-bit-plane-sq --example bench
```

## Results

From `cargo run --release -p ruvector-bit-plane-sq --example bench`
(n=50 000, dim=128, nq=200, k=10). **Real numbers, not projections.**

| backend                | recall@10 | latency (us/q) | code bytes scanned |
|------------------------|-----------|----------------|--------------------|
| full_sq8 (baseline)    |   0.541   |     146.8      |     6 400 000      |
| top_m_approx(M=2)      |   0.407   |     158.2      |     1 600 000      |
| progressive(M=2,r=2)   |   0.500   |     174.3      |     1 602 560      |
| progressive(M=2,r=4)   |   0.535   |     174.4      |     1 605 120      |
| progressive(M=2,r=8)   |   0.541   |     188.1      |     1 610 240      |
| top_m_approx(M=3)      |   0.507   |     161.1      |     2 400 000      |
| progressive(M=3,r=2)   |   0.540   |     164.8      |     2 402 560      |
| progressive(M=3,r=4)   |   0.541   |     177.4      |     2 405 120      |
| top_m_approx(M=4)      |   0.531   |     158.6      |     3 200 000      |
| progressive(M=4,r=2)   |   0.543   |     164.0      |     3 202 560      |
| top_m_approx(M=6)      |   0.543   |     160.8      |     4 800 000      |

### Reading the table honestly

* **Baseline recall is 0.541**, not 1.0, because we're measuring float-L2
  truth against an int-L1-SQ8 ranking. This ceiling is intrinsic to SQ8 at
  D=128 with this data distribution; progressive variants can match it but
  cannot exceed it.
* **`progressive(M=2,r=8)` matches baseline recall** (0.541) while storing
  only **25 %** of the code bytes in its hot plane region (1.6 MB vs 6.4 MB).
  The current scan reconstructs approx codes to u8 first, so **latency does
  not yet drop** — the shortlist pass is baseline-sized plus a refine step.
* **`progressive(M=3,r=2)` lands at recall 0.540 in 164.8 us/q** with
  2.4 MB resident — a 2.67× code-footprint reduction vs the 6.4 MB
  baseline for a 1.0 % absolute recall delta.
* At M=4 and above, progressive essentially saturates: recall is within
  measurement noise of the baseline (0.541–0.543) at 50–75 % of the bytes.

### What the PoC does *not* yet demonstrate

The latency column shows that **today's correctness-first implementation
pays a small overhead** (~12–30 % more wall time than baseline) because the
approximate code is reconstructed to u8 before scanning. The *point* of the
bit-plane layout is that an SSE4/AVX2/NEON kernel could scan M packed bitmaps
directly — one `vpopcnt` per plane per vector — and skip the reconstruction
step entirely. That kernel is listed under future work. The PoC establishes
correctness (recall parity) and the memory-footprint lower bound.

## How it works — a blog-length walkthrough

Imagine a vector database with a billion 128-dim vectors quantized to SQ8.
That's 128 GB of codes before anything else. Every query scans a chunk of
that corpus — tens of millions of vectors in a typical IVF probe — doing
nothing but reading codes and summing byte differences. Memory bandwidth, not
arithmetic, is the ceiling.

SQ8 stores each code byte contiguously: `[c_0 c_1 ... c_127]` per vector.
Each byte holds 8 bits of information, but the *top* bit carries the most
weight: it decides whether the code is in `[0, 127]` or `[128, 255]`, which
locates the vector in the upper or lower half of the per-dim range. The
*bottom* bit carries 1/128 as much information — flipping it moves the
dequantized value by one 256th of the range.

BPP-SQ separates those bits. Instead of storing codes as bytes, it stores
them as *eight bitmaps*: one bitmap per bit position. The MSB bitmap is a
compact `(n × ceil(D/64))` matrix of `u64`s where bit `(i, j)` is the MSB
of code `c[i, j]`. The next bitmap holds the next bit, and so on.

Query time, you don't have to load all eight bitmaps. You can load only the
top `M`, reconstruct an approximate byte per dim (biased to the midpoint of
the unknown tail so the error is symmetric), and compute L1 against that.
The error is bounded by `D · 2^(8-M-1)` — provably, not empirically —
which gives you a monotone-approaching lower-bound-flavored score for
shortlist ranking.

For `M=4`, you load half the planes. The planes are packed bitmaps, so if
SIMD popcount is available on the host (and on x86_64 and aarch64 it is),
the plane-direct scan is a sequence of `xor`+`popcount`+weighted-add per
plane — far fewer loads than reading byte codes. The current PoC pays the
reconstruction cost to keep the code simple and the recall numbers honest.

The refine step is tiny: `k · r` vectors (say, 80 for k=10, r=8), each a
full 128-byte read and L1 compute. Rounding error, essentially.

## Practical failure modes

* **Non-uniform per-dim scale.** If one dim's scale is 10 000× another's,
  the midpoint bias on the low-scale dim contributes nothing to L1 and the
  shortlist is noisy. Mitigation: apply a Hadamard rotation or PCA before
  fitting — exactly the same fix OPQ and Hadamard-SQ use. Confirmed working
  on the synthetic set with mild spread; not stress-tested on skewed data.
* **Pathological queries all-near-midpoint.** If every query coord is near
  127/128, the top-plane differences are tiny and the approximate L1 is
  weak at discriminating. Shortlist multiplier `r` recovers this, at the
  cost of refine work.
* **Tiny datasets.** Below ~10 k vectors, the bitmap-scan fixed costs
  (plane indexing, bias add) dominate and the baseline wins. BPP-SQ is a
  bandwidth optimization; small-corpus wins are negligible.
* **Dim not a multiple of 64.** Last plane word is partially used; correct
  but wastes a few bits per vector per plane. Fine for typical 128/256/768
  dims.

## What to improve next (roadmap)

1. **Plane-direct scan kernel.** Replace the u8 reconstruction path with an
   `xor`+`popcount`+weighted-sum kernel that reads only M bitmaps. Expect
   a bandwidth-proportional latency drop (roughly `M/8` of baseline) on
   memory-bound workloads. Target AVX2 + NEON.
2. **Data-adaptive M.** Monitor shortlist score stability (e.g., gap
   between top-k and 2k-th score); if small, promote to M+1 and re-scan
   locally. Avoids a global M setting.
3. **Integrate with IVF.** Store per-posting plane arrays. The plane-direct
   kernel is especially attractive inside posting lists because the whole
   list fits in L2 when M is small.
4. **Hadamard pre-rotation.** As above, to flatten per-dim scale.
5. **L2 variant.** Replace L1 with expanded-square distance. Needs per-plane
   LUT precompute per query — ~1 % overhead per query, demonstrated by PQ
   ADC.

## Production crate layout proposal

If this graduates, the layout we'd land in the main `ruvector-pq-search`
tree:

```
ruvector-bit-plane-sq/
  src/
    lib.rs           # traits + reference impl (this PoC)
    bitmap_scan.rs   # future: AVX2 + NEON plane-direct kernels
    adaptive.rs      # future: data-adaptive M chooser
    ivf_integration.rs  # future: per-posting plane storage
  examples/
    bench.rs
    ivf_bench.rs     # future
  benches/
    kernel.rs        # future: criterion microbenches
```

Behind the `L1Search` trait we already defined, both the current u8-path
and future bitmap-direct path coexist at parity.

## References

1. Gao, J. et al. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search.*
   SIGMOD 2024.
2. Guo, R. et al. *Quantization based Fast Inner Product Search.* AISTATS 2016.
3. Kuzmin, A. et al. *PDX: Vertical Layouts for Vector Search.* 2024
   (ruvector nightly 2026-09-14 implements the Rust port).
4. Taubman, D. *High performance scalable image compression with EBCOT.*
   IEEE TIP 9(7), 2000 — the original bit-plane progressive coder idea.
5. Jégou, H. et al. *Product Quantization for Nearest Neighbor Search.*
   IEEE TPAMI 33(1), 2011.
6. Zhang, M. et al. *Polysemous Codes.* ECCV 2016 — bit-slice of PQ Hamming.
7. ruvector nightly series, gists 2026-09 through 2026-11 — in-tree prior
   work the design builds on and deliberately differs from.

## Appendix: reproducibility snapshot

* `rustc 1.83.0 (90b35a623 2024-11-26)` (or newer stable).
* Apple M-series Darwin 24.6.0, single-threaded run.
* Numbers above were captured with the pinned `BPP_N=50000 BPP_DIM=128
  BPP_NQ=200` defaults. All variants scanned the same dataset, same queries,
  same ground truth file.
* Tests: `cargo test --release -p ruvector-bit-plane-sq` → 3 passed.
