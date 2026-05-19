# LVQ: Locally-Adaptive Vector Quantization for ruvector

**Date:** 2026-05-19  
**Branch:** `research/nightly/2026-05-19-lvq-quantization`  
**Crate:** `crates/ruvector-lvq`  
**ADR:** [ADR-195](../../../adr/ADR-195-lvq-locally-adaptive-quantization.md)

## Abstract

Most production vector indexes lean on scalar quantization (SQ8) or product
quantization (PQ) for compression. SQ8 fits a single per-dimension range over
the *whole* dataset, which wastes bits when individual vectors live in a
narrower local range; PQ is excellent for offline corpora but is awkward
under streaming inserts (codebooks drift). Intel's "Locally-Adaptive Quantization
for Streaming Vector Search" (Aguerrebere et al., NeurIPS 2023, arXiv:2402.02044)
solves both problems by fitting a (scale, bias) pair *per vector* after
centering against a global mean, then quantizing uniformly to B bits. This
research lands a clean Rust implementation (`ruvector-lvq`) with single-level
LVQ1 (4-bit and 8-bit) and two-level residual LVQ2, and reports real numbers
against an SQ8 baseline and fp32.

**Headline result (d=128, N=20 000, recall@10, Apple Silicon native, release build):**

| Quantizer  | Bits/comp | Bytes/vec | Index MB | Recall@10 | Scan throughput (vec/s) | vs SQ8 recall |
|-----------:|----------:|----------:|---------:|----------:|------------------------:|--------------:|
| f32        | 32        | 512       | 9.77     | 1.0000    | 30 650 020              | n/a           |
| SQ8        | 8         | 140       | 2.67     | 0.9745    |  8 795 532              | baseline      |
| **LVQ1-8** | **8**     | **140**   | **2.67** | **0.9885**| **13 457 928**          | **+1.4 pp**   |
| LVQ1-4     | 4         |  76       | 1.45     | 0.8320    | 11 305 267              | −14.2 pp      |
| LVQ2-8x4   | 12        | 212       | 4.04     | 0.9995    |  3 912 945              | +2.5 pp       |

**At identical bytes per vector, LVQ1-8 raises recall@10 from 0.9745 → 0.9885
while running scans ~53% faster than SQ8.** LVQ2-8x4 closes the recall gap to
0.9995 (within 0.0005 of fp32) at the cost of one extra 4-bit pass.

## SOTA survey

Quantization for ANN has converged on a small set of families. The relevant
2023–2025 work for streaming, large-scale, in-memory vector search:

- **SQ8 (scalar quantization)** — per-dimension affine map fit globally over
  the dataset. Cheap, but high-variance vectors get clipped or under-resolved.
- **PQ / OPQ / IVF-PQ** (Jégou et al. 2011; Ge et al. 2014) — split d-dim
  vectors into sub-vectors, learn k-means codebooks per sub-vector, store an
  index into each codebook. Excellent compression (8–32×), but codebooks must
  be retrained on data drift and ADC LUTs are awkward to stream.
- **LVQ** (Aguerrebere, Tepper, Bhattacharya, Hildebrand, Willke; NeurIPS
  2023; arXiv:[2402.02044](https://arxiv.org/abs/2402.02044)) — per-vector
  scale + bias, uniform B-bit quantization. No codebook, no retraining; insert
  is O(d) and decode is two FMAs per component. Used in Intel's
  [ScalableVectorSearch](https://github.com/intel/ScalableVectorSearch).
  US Patent Application 20240020308.
- **Turbo LVQ** (same group, 2024) — SIMD packing layout that reorders 4-bit
  codes so that 16 components hit a single AVX-512 lane, ~28% extra throughput.
- **RaBitQ** (Gao & Long, SIGMOD 2024) — random rotation + 1-bit code with
  theoretical guarantees. Excellent at extreme compression, weaker absolute
  recall at moderate budgets. Already in `crates/ruvector-rabitq`.
- **LeanVec** (Tepper et al., 2023) — dimensionality reduction before
  quantization; complements LVQ. Already in `crates/ruvector-leanvec`.
- **SymphonyQG** (Yang et al., SIGMOD 2025) — fuses graph search with
  per-block PQ LUTs. Future work (see roadmap).

Competitors:

- **Milvus** (Zilliz) — SQ8 + IVF-PQ, no LVQ.
- **Qdrant** — SQ8 + binary quantization; no per-vector scale+bias path.
- **Weaviate** — PQ + binary; no LVQ.
- **Pinecone** — proprietary; published designs use PQ-style codes.
- **LanceDB** — PQ on top of IVF; recently added scalar quant.
- **FAISS** — has `IndexScalarQuantizer` (SQ8) and `IndexPQ`. No LVQ.
- **Intel SVS** — the reference LVQ implementation, C++.

ruvector previously had SQ8 (in `ruvector-rairs` and several index crates),
RaBitQ (`ruvector-rabitq`), and LeanVec (`ruvector-leanvec`). LVQ closes the
streaming-quantization gap.

## Proposed design

A small, allocation-conscious crate exposing a `Quantizer` trait:

```rust
pub trait Quantizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn bits_per_component(&self) -> u8;
    fn code_bytes(&self) -> usize;

    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError>;
    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError>;
    fn decode(&self, e: &Encoded, out: &mut [f32]);
    fn distance(&self, q: &[f32], q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32;
}
```

Backends:

- `Sq8` — global per-dimension affine. Baseline.
- `LvqOne` — single-level LVQ at 4 or 8 bits. Per-vector `(Δ, lo, ‖x̂‖²)`.
- `LvqTwo` — primary 4/8-bit LVQ + residual 4/8-bit LVQ.

The encoding for LVQ1-B is:

```
c   = x - mean
lo  = min(c),  hi = max(c)
Δ   = (hi - lo) / (2^B - 1)
q_i = round((c_i - lo) / Δ)             ∈ [0, 2^B-1]
x̂_i = Δ q_i + lo + mean_i
```

Asymmetric L2 then expands to a single SIMD-friendly pass:

```
‖q - x̂‖² = ‖q‖² + ‖x̂‖² − 2⟨q, x̂⟩
        = ‖q‖² + ‖x̂‖² − 2 Δ ⟨q, c⟩ − 2 lo ·Σq − 2 ⟨q, mean⟩
```

`‖q‖²` and `Σq` and `⟨q, mean⟩` are hoisted once per query. The inner loop is
a single dot product against the raw code, which is what gives LVQ its
throughput advantage over SQ8 (SQ8 has a per-dim `delta[i]` to multiply,
LVQ has one shared `scale`).

## Implementation notes

- **Per-vector overhead** is exactly 12 bytes (scale + bias + ‖x̂‖² as
  f32 each). For d=128 and B=8 this is 9% overhead — cheaper than PQ.
- **Packing**: 8-bit codes are one byte per component, 4-bit codes are
  two-per-byte (low nibble = even index, high nibble = odd). No vector
  re-layout (Turbo-LVQ-style) yet — left as a follow-up.
- **Mean fitting** is a single linear pass over training. Streaming
  insertion can use an exponential moving mean (not implemented here; trivial
  follow-up).
- **No mocks**: every test exercises the actual encode/decode path against
  real Gaussian data and ground-truth top-K.
- **Files**: 7 source files, all under 200 lines (largest is `lvq1.rs`
  at 192). Project-rule compliant.

## Benchmark methodology

- **Hardware**: Apple Silicon (darwin 24.6.0), Rust release profile, single
  thread, no SIMD intrinsics (LLVM autovectorization only).
- **Dataset**: synthetic N=20 000 isotropic-Gaussian vectors at d=128.
  Standard normal distribution, seed `0xC0FFEE` for reproducibility.
- **Queries**: 200 independent Gaussian queries.
- **Metric**: L2.
- **Ground truth**: exact top-10 brute force over the f32 vectors.
- **Recall@10**: |truth ∩ approx| / 10, averaged across queries.
- **Scan throughput**: `Nq × N / wallclock`. Measured end-to-end including
  the sort-and-truncate top-K step.

Reproduce with:

```
cargo run -p ruvector-lvq --release --bin lvq-demo
cargo bench  -p ruvector-lvq
```

## Results

```
ruvector-lvq demo  |  N=20000  Nq=200  d=128  k=10

quantizer  bits/comp  bytes/vec  index_mb  fit_ms  encode_ms  scan_ms  scans/s     recall@10
f32        32         512        9.77      n/a     n/a        130.5    30 650 020  1.0000
SQ8        8          140        2.67      1.5     3.4        454.8     8 795 532  0.9745
LVQ1-8     8          140        2.67      0.2     7.9        297.2    13 457 928  0.9885
LVQ1-4     4          76         1.45      0.2     8.5        353.8    11 305 267  0.8320
LVQ2-8x4   12         212        4.04      0.3     15.4       1022.2    3 912 945  0.9995
```

Discussion:

1. **LVQ1-8 dominates SQ8 at identical footprint.** Same 140 bytes/vec, but
   recall climbs 0.9745 → 0.9885 (+1.4 pp absolute) *and* throughput goes
   from 8.8 M to 13.5 M scans/s (+53%). The throughput win comes from the
   single shared `scale` per vector vs SQ8's per-dimension `delta[i]` table.
2. **f32 brute force wins on this hardware at this size.** d=128 fits
   comfortably in L1 and LLVM autovectorizes `Σ(a-b)²` very well. The
   quantizer wins are about memory footprint (3.66×–6.74× smaller) and the
   fact that you can keep 4–7× more vectors resident before going to disk —
   not raw scan latency at this scale. The point of LVQ is to make graph
   indexes (HNSW, DiskANN, Vamana) cache-friendly, not to beat fp32 dense
   scans head-on.
3. **LVQ1-4 is a strong "fast filter" tier.** 6.74× compression and 0.832
   recall is exactly the regime where you'd use it as a coarse first pass
   before reranking with LVQ2 or fp32.
4. **LVQ2-8x4 essentially recovers fp32 recall** (0.9995 vs 1.0) at 2.4×
   compression. Use it where compactness *and* fidelity matter (reranking
   the top 100 after a coarse LVQ1-4 pass).

## How it works (blog-readable walkthrough)

Think of SQ8 as taking a snapshot of the *whole dataset's range* on each
axis. If most vectors live in a narrow corner of that range, you waste bits.

LVQ does the opposite: it asks each vector "what's *your* range?" and gives
it a personalised affine map into the 0..255 (or 0..15) integer grid. Two
floats of bookkeeping per vector pay for themselves immediately because the
inner loop becomes one dot product instead of one dot product *per
dimension's scale*.

The two-level variant is the classic "approximate the residual" trick: encode
once with 8-bit LVQ, look at what you missed, then encode the *miss* with
another 4-bit LVQ. The second pass costs +4 bits per component but knocks
recall from 0.9885 to 0.9995.

## Practical failure modes

- **Heavy-tailed coordinates.** A single outlier component stretches `(lo,
  hi)` and ruins resolution everywhere else. Mitigation: clip outliers above
  a fitted percentile before encoding (Turbo-LVQ does this).
- **Very low dimensions (d < 16).** Per-vector 12-byte overhead is 19%+ at
  d=16/B=8 — break-even with SQ8 erodes. LVQ is targeted at d ≥ 64.
- **Constant vectors.** If all components are equal, `hi == lo` and `Δ`
  collapses. We fall back to `Δ = 1`; recall is fine but it should never
  happen on real embedding data.
- **Streaming mean drift.** The current fit is one-shot. For long-running
  streams, swap to EMA + periodic re-fit; codes stay valid because per-vector
  scale+bias absorb global drift (this is the core LVQ insight).

## What to improve next (roadmap)

1. **Turbo-LVQ SIMD layout.** Re-pack 4-bit codes so 16 components hit one
   AVX2/NEON lane. The paper reports +28% scan throughput; on this PoC
   that's the cleanest next win.
2. **Anisotropic LVQ.** Use a learned diagonal whitening matrix before
   encoding (cheap variant of OPQ rotation). Should claw back another
   1–2 pp recall at 4-bit.
3. **HNSW + LVQ integration.** Wire `Quantizer` into `ruvector-graph` so
   HNSW neighbour scans use LVQ codes. Pair with reranking against an fp32
   tail (or LVQ2) at the top-K boundary.
4. **Disk tier.** LVQ1-4 codes (76 B/vec at d=128) fit ~13 vectors per
   page; integrate into `ruvector-diskann`.
5. **Streaming mean re-fit.** EMA mean + tombstone-aware periodic re-encode.

## Production crate layout proposal

```
ruvector-lvq/                      this crate (encoder + scan kernel)
ruvector-lvq-wasm/                 wasm-bindgen surface for browser indexes
ruvector-graph/lvq_integration.rs  HNSW neighbour scan using Quantizer trait
ruvector-diskann/                  page format = (graph_neighbours, lvq1_4_codes)
ruvector-bench/lvq_recall_curves   recall@k vs bits sweep on SIFT/GIST/DEEP
```

## References

- Aguerrebere, Bhattacharya, Hildebrand, Tepper, Willke. "Locally-Adaptive
  Quantization for Streaming Vector Search." arXiv:2402.02044, 2024.
  <https://arxiv.org/abs/2402.02044>
- Intel ScalableVectorSearch: <https://github.com/intel/ScalableVectorSearch>
- US Patent Application 20240020308 (LVQ for similarity search).
- Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor Search."
  PAMI 2011.
- Gao & Long. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  SIGMOD 2024.
- Tepper et al. "LeanVec: searching vectors faster by making them fit."
  2023.
- Yang et al. "SymphonyQG: Towards Symphonic Integration of Quantization
  and Graph for ANN Search." SIGMOD 2025.
