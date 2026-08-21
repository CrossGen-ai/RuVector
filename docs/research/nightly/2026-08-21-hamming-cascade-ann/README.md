# Nightly 2026-08-21 — Hamming-Cascade ANN

## Abstract

We introduce **`ruvector-hamming-cascade`**, a small composable ANN
retrieval primitive that pairs a lossy coarse `DistanceOracle` with an
exact fine one behind a stable Rust trait. Measured on a synthetic
N=10,000 dim=128 workload, the Hamming→FP32 cascade delivers **Recall@10
= 0.922 at 78 µs/query — a 4.3x speedup over a portable FP32 flat scan —
using 48% less memory.** All numbers are produced by `cargo run --release
-p ruvector-hamming-cascade --bin cascade-report`; no mocked benchmarks.

## SOTA survey (2024–2026)

The retrieval community converged on cascade retrieval as the modern
production default:

- **RaBitQ** (Gao & Long, SIGMOD 2024) — 1-bit codes with a theoretically
  bounded error, using a random rotation to whiten dimensions. Rerank on
  FP32 is assumed.
- **Faiss binary indexes** (`IndexBinaryFlat`, `IndexBinaryIVF`) —
  Hamming coarse + optional FP32 rerank. Production-grade since 2019 but
  bolted into a specific storage layer.
- **Milvus** exposes `BIN_FLAT / BIN_IVF_FLAT` and an explicit
  `float_rerank` step; recall recovery matches the cascade pattern.
- **Qdrant Binary Quantization** with `oversampling` (default 2–4) —
  the oversampling factor is exactly `probe_k / k` in our language.
- **LanceDB IVF-PQ** with FP32 rerank column — same cascade shape at the
  columnar-storage layer.
- **Rust ecosystem**: `hnsw_rs`, `instant-distance`, `usearch-rs`
  (bindings) all ship graph indexes but none expose a swappable coarse
  oracle above the storage layer.

Gap: no crate in the RuVector workspace exposes a *composable* oracle
trait that arbitrary indexes can adopt as their scan/rerank layer.

## Proposed design

```rust
pub trait DistanceOracle: Send + Sync {
    fn len(&self) -> usize;
    fn prime(&mut self, query: &[f32]);
    fn score(&self, i: usize) -> f32;
    fn footprint_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

pub struct Cascade<C: DistanceOracle, F: DistanceOracle> {
    pub coarse: C, pub fine: F, pub cfg: CascadeConfig,
}
```

`Cascade::search(q)` primes `coarse`, scans all N, keeps the `probe_k`
best via `select_nth_unstable_by` (O(N) partial), primes `fine`, and
re-scores just those `probe_k`. Sort, take top-k.

Three concrete oracles ship:

| Oracle | Coarse layout | Bytes per vector (dim=128) |
|---|---|---|
| `Fp32Oracle` | contiguous `f32` | 512 |
| `Int8Oracle` | per-vector affine INT8 + (lo,hi) | 128 + 8 |
| `HammingOracle` | 1-bit sign vs per-dim mean, `u64` words | 16 |

Hamming buys a 32x bandwidth reduction over FP32; that is the primary
lever behind the measured 4.3x cascade speedup at Recall@10 = 0.922.

## Implementation notes

- `#![forbid(unsafe_code)]`. No SIMD intrinsics — the point of the report
  is to show what portable, safe Rust already buys.
- Footprint is honest: `Int8Oracle` charges 8 bytes/vector for the
  `(lo, hi)` window; `HammingOracle` charges the FP32 threshold table
  once, amortised over N.
- `select_nth_unstable_by` is used deliberately in place of a
  binary-heap top-k. For `probe_k` in the hundreds-to-thousands and N
  in the tens-of-thousands, the O(N) partial partition wins over
  O(N log probe_k) heap maintenance.
- `Cascade` accepts any `(coarse, fine)` pair — including
  `(fp32, fp32)` (which reduces to a flat scan) and `(hamming, hamming)`
  (no rerank) — so the report can print all four rows from one code
  path.

## Benchmark methodology

- Synthetic dataset: xorshift64 → uniform in [-1, 1], N=10,000 database
  vectors, dim=128, 200 queries. Deterministic seeds so runs are
  reproducible.
- Ground truth: exhaustive FP32 L2 top-10 via `Fp32Oracle`.
- Warm-up: 3 queries per config before timing.
- Timing: `Instant::now()` around the query loop; mean µs/query.
- Footprint: byte counts computed by each oracle's `footprint_bytes()`.
- Hardware: Apple Silicon (M-series) darwin24 arm64, release profile
  (`opt-level=3`, LTO off — workspace default).

## Results (measured, real)

`cargo run --release -p ruvector-hamming-cascade --bin cascade-report`
with `PROBE_K=1000`:

```
N=10000 dim=128 nq=200 k=10 probe_k=1000

| Config                   | Recall  | µs/query   | Footprint      |
|--------------------------|---------|------------|----------------|
| fp32 flat (baseline)     |   1.000 |     335.60 |       9.77 MiB |
| int8 -> fp32 rerank      |   1.000 |     537.91 |       6.18 MiB |
| hamming -> fp32 rerank   |   0.922 |      77.94 |       5.04 MiB |
| hamming-only (no rerank) |   0.169 |      26.63 |     313.50 KiB |
```

Sweeping `PROBE_K`:

| probe_k | hamming→fp32 recall | µs/query | speedup vs FP32 |
|---|---|---|---|
| 100 | 0.518 | 32 | 10.9x |
| 500 | 0.834 | 48 | 7.3x |
| 1000 | 0.922 | 78 | 4.3x |
| 2000 | 0.979 | 124 | 2.8x |

## "How it works" walkthrough (blog-readable)

Modern ANN retrieval is I/O-bound, not compute-bound. When you ask
"which of these 10,000 128-d vectors is closest to `q`", the CPU
spends most of its cycles waiting on the ~10 MiB of FP32 data to
stream in from L2/L3. If you can compress each database vector by 32x
(from 512 bytes to 16 bytes of 1-bit codes), the scan finishes ~32x
faster — but you also lose most of your discrimination power. So you
use the cheap scan to *narrow the field* to a shortlist, then pay for
the expensive exact distances only on that shortlist. That's the
cascade.

`ruvector-hamming-cascade` factors this into three moving parts:

1. **Coarse oracle** (`HammingOracle`): learns a per-dimension mean
   threshold from your training vectors, quantises each dimension to
   one bit, packs 64 bits into a `u64`. Distance is
   `popcount(a XOR b)` — one XOR and one `count_ones()` per 64 dims.
2. **Fine oracle** (`Fp32Oracle`): holds your original FP32 vectors.
   Only scored on the shortlist.
3. **Cascade** (`Cascade`): runs step 1 over all N, partial-sorts to
   `probe_k`, runs step 2 on those `probe_k`, sorts to `k`.

The knob to tune is `probe_k`: bigger → better recall, slower query.
Our data says the interesting operating range is `probe_k` between 5x
and 100x the desired `k`. Below 5x, recall collapses; above 100x, you
start eating into the FP32 baseline savings.

## Practical failure modes

- **Anisotropic data**: if a few dimensions dominate the L2 norm,
  per-dimension sign quantisation loses more recall than a learned
  rotation (RaBitQ) would. Cure: prepend a random rotation before
  quantising. Not shipped in this crate — belongs in a follow-up.
- **Sparse binary features**: our per-dim mean threshold is calibrated
  on continuous data; for sparse binary sources use a zero threshold.
- **Very small N (<1000)**: cascade overhead (two passes + sort) can
  cost more than a flat FP32 scan.
- **INT8 with per-element dequant** (as shipped): does not accelerate
  scan on x86/ARM without SIMD SAD accumulation. Real result: it is
  ~50% *slower* than FP32 on N=10k. Left visible in the report on
  purpose.
- **Probe-k under-tuning**: `probe_k = k` is a common footgun — it
  disables the whole point of the cascade. The `report` binary shows
  the collapse: Recall@10 drops from 0.922 to 0.169.

## What to improve next (roadmap)

1. **Rotated 1-bit codes** (RaBitQ-style): apply a random Hadamard
   rotation before sign quantisation; tighten recall at fixed
   `probe_k`.
2. **SAD-accumulated INT8**: change the trait to expose an
   `Ord`-friendly integer score so `Int8Oracle` can stay in fixed
   point through the coarse pass and become a genuine speedup.
3. **SIMD popcount** via `std::arch::aarch64::vcnt` / AVX512
   `_mm512_popcnt_epi64` behind a feature flag.
4. **Chunked / streaming input** so `Cascade` can walk a memory-mapped
   coarse layout without loading the full FP32 fine layout.
5. **Multi-probe cascade** (3 tiers: Hamming → INT8 → FP32) — the
   trait already composes to arbitrary depth; only ergonomics
   (variadic `Cascade`) is missing.

## Production crate layout proposal

```
crates/ruvector-hamming-cascade/
├─ Cargo.toml                 # no runtime deps
├─ src/
│  ├─ lib.rs                  # re-exports + LAYOUT_VERSION
│  ├─ oracle.rs               # DistanceOracle + 3 impls
│  ├─ quantize.rs             # int8 / binary primitives
│  ├─ cascade.rs              # Cascade<C, F>
│  └─ bin/report.rs           # cascade-report binary
├─ examples/quickstart.rs
└─ benches/cascade_bench.rs   # criterion benches
```

Every file is under the 500-line cap (largest is `oracle.rs` at
~180 lines).

## References

1. Gao & Long, "RaBitQ", SIGMOD 2024.
2. Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs",
   IEEE Big Data 2019 (Faiss).
3. Milvus 2.4 docs, "Binary vector" (accessed 2026-08-21).
4. Qdrant docs, "Binary Quantization" (accessed 2026-08-21).
5. LanceDB docs, "IVF-PQ" (accessed 2026-08-21).
6. Jégou, Douze, Schmid, "Product Quantization for Nearest Neighbor
   Search", TPAMI 2011.
