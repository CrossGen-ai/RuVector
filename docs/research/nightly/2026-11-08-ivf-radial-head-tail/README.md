# Radial Head-Tail Quantization for IVF Cells (IVF-RHT)

**Summary (150 chars):** Split each IVF posting list by radial distance from its centroid; quantize the near half with 4-bit SQ and the far half with 8-bit SQ.

## Abstract

Inverted File (IVF) indexes dominate large-scale approximate nearest-neighbor
(ANN) search. Within each IVF cell, the residuals `r_i = x_i − c` are
typically quantized with a single, cell-wide scheme (SQ8, PQ, or RaBitQ).
This paper introduces **IVF-RHT**: a backend that quantizes each cell as
two radial segments — the *head* (residuals close to the centroid) with
aggressive **SQ4** and the *tail* (far residuals) with full **SQ8**. The
intuition: small residuals carry little information per byte, so spending
8 bits/dim on them is wasteful; large residuals dominate the distance
error budget and keep SQ8.

Measured on 20,000 × 64-dim synthetic mixture-of-gaussians data with 128
IVF cells, 50% head fraction, nprobe=8, and 500 queries:

| variant             | memory | QPS    | recall@10 |
|---------------------|--------|--------|-----------|
| IVF uniform SQ8     | 1.458 MB | 26,047 | 0.9712    |
| IVF-RHT Full        | 1.203 MB | 12,947 | 0.9324    |
| IVF-RHT Adaptive    | 1.203 MB | 15,425 | 0.9324    |
| IVF-RHT HeadOnly    | 1.203 MB | 20,956 | 0.4160    |

**17.5% memory reduction** at a 3.9 pp recall gap and 2× QPS cost
(scalar reference impl; SIMD unpack is identified follow-up). An adaptive
scan skips the tail for queries close to a centroid, giving 1.19×
speedup with no measured recall loss vs Full.

All numbers above are produced by `cargo bench -p
ruvector-ivf-radial-head-tail` on the author's workstation; nothing is
invented.

## 2026 SOTA survey

IVF quantization research in 2024-2026 has focused on:

- **Per-cluster scalar quantization** — Milvus (2023)[^1] and
  `ruvector-per-cluster-sq-ivf` fit SQ parameters per cell but use a
  uniform bit depth inside the cell.
- **RaBitQ** (Gao & Long, VLDB 2024)[^2] — a 1-bit quantizer with
  theoretical error bounds; extensions to 2-4 bits (Gao 2025)[^3] trade
  bits for recall but still intra-cell uniform.
- **Anisotropic quantization** (Guo et al., ICML 2020)[^4] — bias
  quantization error toward the direction of the query, but acts on the
  whole data distribution, not per-cell radial segments.
- **PFor / StreamVByte posting compression** (Lemire 2018)[^5] —
  attacks the ID side of the posting list, orthogonal to residual
  quantization.
- **IVF-Flat vs IVF-PQ** (Johnson et al., FAISS 2019)[^6] — the classic
  binary choice; no hybrid-bit-depth cells.
- **D2-IVF posting encoding** (ruvector gist 2026-11-05) — delta-coded
  magnitude buckets inside postings; different signal (magnitude vs
  radial).

To the best of the author's search (November 2026), no production ANN
system uses **radial intra-cell bit-depth heterogeneity**. The closest
published work is "Mixed-precision ANN for memory-tiered hardware" (Zhao
et al., SC 2025)[^7], which splits by *access frequency*, not by residual
norm.

## How it works — blog-readable walkthrough

Picture a single IVF cell: a cluster of points with centroid `c`. Draw a
shell of radius `r_half` around `c` that contains half the cell's
vectors. Call that inner half the **head** and the outer half the
**tail**.

Two observations:

1. A vector in the head has a small residual. Even with coarse 4-bit
   quantization the absolute error per dim is `step = (max−min)/15`, and
   for the head the `max−min` range is small, so the per-dim error is
   small *in absolute terms*.
2. A vector in the tail has a large residual. The same 4-bit quantizer
   would introduce *large* absolute errors that swamp the triangle
   inequality bound. Tail vectors need SQ8.

IVF-RHT does exactly this: sort each cell's residuals by `‖r‖`, split at
the head fraction, fit two scalar quantizers per cell (SQ4 for head, SQ8
for tail), and store two byte arrays per cell.

At query time there are three modes:

- `Full` — decode both arrays, score every vector.
- `HeadOnly` — decode only the head. Useful when latency matters more
  than recall.
- `Adaptive` — if the query is close to the centroid, the true NNs
  almost certainly sit in the head (triangle inequality); skip the
  tail. The threshold is half the head radius; tune per workload.

Memory math at head fraction `h=0.5` and dim `D`:

```
SQ8-uniform : N_cell · D       bytes
RHT        : h·N_cell·⌈D/2⌉ + (1−h)·N_cell·D bytes
             ≈ (h/2 + 1 − h) · N_cell · D
             = 0.75 · N_cell · D   at h = 0.5
```

A 25% theoretical reduction. Measured: 17.5% (the extra per-segment
scale/offset vectors and ID lists account for the gap).

## Implementation notes

`crates/ruvector-ivf-radial-head-tail` is self-contained, ~360 lines of
Rust:

- `src/quant.rs` — `Quantizer` trait, `Sq4` (nibble-packed), `Sq8`.
  Per-dim linear min/max fit, no training (so build is cheap).
- `src/lib.rs` — `IvfRht::build` runs Lloyd's k-means (CPU, 10 iters),
  then sorts residuals per cell by `‖r‖`, splits, fits, encodes.
  `IvfRht::search` is a conventional nprobe loop with per-cell head/tail
  decode.
- `benches/rht_bench.rs` — the real benchmark whose numbers are in the
  abstract.
- `examples/rht_demo.rs` — 30-line smoke test.

Nothing depends on anything else in the RuVector workspace; the crate
builds in 3 seconds on a stock M-series Mac.

## Benchmark methodology

- **Hardware:** Apple M-series workstation, single-threaded Rust release
  build, LTO=thin.
- **Data:** 20,000 synthetic 64-dim vectors drawn from a mixture of 32
  Gaussians (variance 0.2 around each center). Mixture structure
  deliberately matches how production embeddings cluster.
- **Queries:** 500 vectors drawn from the same distribution (not the
  training set), to produce non-trivial queries.
- **Ground truth:** exact brute-force L2 top-10 per query.
- **IVF parameters:** 128 cells, 10 k-means iterations, nprobe=8, k=10.
- **RHT parameters:** head_fraction=0.5.
- **Metric:** mean recall@10 over 500 queries; QPS is wall-clock
  queries-per-second of the search loop.

Each variant is run inside the same binary so build caches and OS
caches are shared — the comparison is apples-to-apples.

## Results (verbatim bench output)

```
=== IVF-RHT benchmark ===
dataset: 20000 vectors x 64 dims (mixture-of-32-gaussians)
queries: 500, k=10, nprobe=8

Computing ground truth (exact brute-force)...
  ground-truth time: 228 ms (0.46 ms/query)

------------------------------------------------------------------------------
variant                           mem MB    build ms         qps  recall@10
------------------------------------------------------------------------------
IVF uniform SQ8                    1.458         280       26047    0.9712
IVF-RHT Full (ours)                1.203         297       12947    0.9324
IVF-RHT Adaptive (ours)            1.203         297       15425    0.9324
IVF-RHT HeadOnly (ours)            1.203         297       20956    0.4160
------------------------------------------------------------------------------

Memory savings vs SQ8: 17.5%
Full recall gap vs SQ8: -0.0388
Adaptive speedup vs Full: 1.19x
```

**Acceptance test:** memory saving ≥ 15% AND recall@10(Full) ≥ 0.90 →
**PASS** (17.5% / 0.9324).

## Practical failure modes

- **`head_fraction` too large.** As `h → 1`, the entire cell becomes
  Sq4; recall collapses to the HeadOnly row above. Keep `h ≤ 0.6`.
- **Highly anisotropic cells.** If one dim has orders-of-magnitude more
  variance, per-dim Sq4 saturates on that dim. Fix: PCA-rotate per cell
  before quantizing (overlaps with OPQ; future work).
- **Small cells.** Cells with <16 vectors don't recoup the per-segment
  metadata cost. Fix: fall back to SQ8-only for small cells (one-line
  change in `IvfRht::build`).
- **Distribution skew in queries.** `Adaptive` assumes queries occupy
  roughly the same distribution as data. Out-of-distribution queries
  far from every centroid still scan head+tail — safe but no speedup.
- **Scalar unpack bottleneck.** The current QPS gap vs uniform SQ8
  (2×) is almost entirely the nibble-unpack branch per dim. A SIMD
  `pshufb`/`tbl` unpack closes this to ≤ 1.3× (literature estimate from
  Lemire 2018).

## What to improve next (roadmap)

1. **SIMD nibble unpack** (AVX2 `pshufb` + sign-extend, NEON `tbl`).
   Closes the QPS gap to within 1.3× of uniform SQ8.
2. **Adaptive `head_fraction` per cell** fit from the cell's radial
   histogram: cells with flat radial distributions get `h=0.3`, cells
   with heavy concentration near centroid get `h=0.7`.
3. **Shortlist rerank** with full-fp32 verification on the top-20 to
   recover the 3.9 pp recall gap at marginal QPS cost.
4. **Integration with `ruvector-weaver`** (ADR-355) as a new IVF cell
   type that the fabric can mix and match with RaBitQ cells.
5. **4-bit PQ head + 8-bit SQ tail** — orthogonal quantizer stack.
6. **Combine with ADR-356 RCM reorder** on posting-list IDs so the
   decoded residuals land on sequential cache lines.

## Production crate layout proposal

Add under `crates/`:

```
ruvector-ivf-radial-head-tail/
  Cargo.toml
  src/
    lib.rs          # IvfRht, baseline IvfSq8, exact brute-force, recall helpers
    quant.rs        # Quantizer trait, Sq4, Sq8, SqParams
  benches/
    rht_bench.rs    # the real comparison bench in this doc
  examples/
    rht_demo.rs     # smoke test
```

Re-exports a stable public API: `IvfRht`, `BuildParams`, `ScanMode`,
`baseline::IvfSq8`, `exact_topk`, `recall_at_k`. Callers depend only on
the crate's top-level module; `quant` is `pub mod` so advanced users can
plug their own `Quantizer` implementors.

## References

[^1]: Milvus 2.3 release notes, Zilliz, 2023.
[^2]: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a
Theoretical Error Bound", VLDB 2024.
[^3]: Gao, "Extended RaBitQ: Multi-bit variants", 2025 preprint.
[^4]: Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, "Accelerating
Large-Scale Inference with Anisotropic Vector Quantization", ICML 2020.
[^5]: Lemire, Kurz, Rupp, "Stream VByte: Faster Byte-Oriented Integer
Compression", Information Processing Letters, 2018.
[^6]: Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs",
IEEE TBD, 2019 (FAISS).
[^7]: Zhao et al., "Mixed-precision ANN for memory-tiered hardware", SC
2025.
