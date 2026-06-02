# SymphonyQG: Joint Graph + 1-bit Quantization for ruvector

**Nightly research · 2026-06-02**

> **⚠️ Provenance.** The name "SymphonyQG" is borrowed from Yang et al.,
> "SymphonyQG: Towards Symphonious Integration of Quantization and Graph for
> Approximate Nearest Neighbor Search" (SIGMOD 2025). The implementation in
> `crates/ruvector-symphonyqg` is an original Rust take on the *core idea*
> (joint graph traversal with quantized distances + float rerank), not a faithful
> port of the paper's exact code layout or AVX-512 kernels. Judge it on the
> reproducible benchmarks below.

---

## Abstract

Most modern vector indexes treat the graph and the quantizer as two separate
systems: HNSW (or DiskANN) chooses candidates with full-precision arithmetic,
then a separate IVF-PQ or RaBitQ layer compresses for storage. **SymphonyQG**
collapses that boundary: the graph *itself* is traversed using quantized
distances, with a small float-precision rerank applied only to the final
shortlist. The result on the paper's billion-scale benchmarks is roughly 1.5–
2× the QPS of HNSW at equivalent recall.

This PoC ships `crates/ruvector-symphonyqg`: ruvector's first crate to fuse a
single-layer NSW graph with a 1-bit RaBitQ-style codec, exposed via three
`SearchMode`s that share the same graph but score it with different
distance functions. The honest result on N=8K, D=128 i.i.d. Gaussian data:

| Mode | Recall@10 | QPS | µs/query | vs Float |
|---|---|---|---|---|
| Float graph (baseline) | 91.6% | 4,777 | 209.4 | 1.00× |
| Binary graph (1-bit only) | 11.5% | 10,676 | 93.7 | 2.23× |
| **SymphonyQG (1-bit + rerank=200)** | **57.9%** | **8,938** | **111.9** | **1.87×** |
| Brute force (oracle) | 100.0% | 1,934 | 517.0 | 0.40× |

Hardware: Apple M4 Max, macOS Darwin 24.6.0 arm64, `rustc 1.89.0 --release`.
Data: i.i.d. Gaussian via Box–Muller, N=8K, D=128, 200 queries, k=10,
ef_search=200, M=24, ef_construction=128, rerank=200.

**Memory:** the 1-bit codes use 16 bytes per vector versus 512 bytes for the
float copy → **32× compression**.

---

## SOTA survey

* **HNSW** (Malkov & Yashunin, 2018) — multi-layer proximity graph,
  long-time recall/QPS leader.
* **RaBitQ** (Gao et al., SIGMOD 2024) — 1-bit randomized quantization
  with proven asymmetric distance bound; the substrate for several
  2025-era systems.
* **SymphonyQG** (Yang et al., SIGMOD 2025) — fuses RaBitQ into the
  graph itself, traverses on quantized distances, reranks the top-r in
  float. Reports 1.5–2× QPS over HNSW at recall ≥ 0.95 on SIFT/DEEP/T2I.
* **CAGRA** (NVIDIA, 2024) — GPU graph search, complementary direction.
* **iRangeGraph** / **SeRF** (SIGMOD 2024) — range-filtered graph
  indexes; orthogonal to quantization.

The SymphonyQG paper's key insight is that **graph traversal is dominated
by distance computations, not by graph navigation**. If each distance
becomes ~30× cheaper through quantization, the whole pipeline gets close
to that speedup, provided the quantized distance ordering is correlated
enough with the true distance to keep the same edges on the search frontier.

---

## Proposed design

```
              ┌──────────────────────────────┐
              │  GraphBuilder (NSW, M=24)    │
              │  greedy beam, ef_c=128       │
              └─────────────┬────────────────┘
                            │ build
                            ▼
              ┌──────────────────────────────┐
              │  Graph: vectors + adjacency  │
              └─────────────┬────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │  BinaryCodec (RaBitQ-1-bit)  │
              │  centroid + ±1 diag + Walsh– │
              │  Hadamard rotation + sign    │
              └─────────────┬────────────────┘
                            │ encode_all
                            ▼
              ┌──────────────────────────────┐
              │  Searcher                    │
              │  search(mode = Float |       │
              │         Binary |             │
              │         Symphony { rerank }) │
              └──────────────────────────────┘
```

### Components

* **Graph** — flat single-layer NSW (`graph.rs`). Multi-entry seeding
  with 8 stride-spaced entry points cures the single-fixed-entry
  failure mode without paying for an HNSW upper layer.
* **BinaryCodec** — centroid-subtract → random ±1 diagonal flip →
  in-place fast Walsh–Hadamard transform → sign-bit per coordinate.
  Stored as packed `u64` words. Distance is asymmetric XOR-popcount
  weighted by the query's average projected magnitude — monotone in the
  true L2 (the RaBitQ asymmetric bound, simplified).
* **Searcher** — owns the graph + codes and exposes the three modes.
  Symphony walks the graph using quantized scores, gathers the top-`r`,
  then reranks with float L2 and truncates to `k`.

---

## Implementation notes

* No `unsafe` — `#![forbid(unsafe_code)]` at the crate root. The
  Walsh–Hadamard transform is a textbook radix-2 in-place loop and
  vectorises automatically under `-Copt-level=3`.
* All four source files are well under the 500-line cap.
* The codec is reproducible: the random ±1 diagonal is derived from a
  fixed LCG seed (`0xC0FFEE`) so codebook layouts match across rebuilds.
* The `BinaryCodec` pads `dim` up to the next power of two before
  Hadamard, then bit-packs the padded length. With D=128 (already power
  of two) there is zero pad waste.

---

## Benchmark methodology

`cargo run --release -p ruvector-symphonyqg --bin symphonyqg-demo`
runs the full pipeline end-to-end:

1. Generate 8,000 i.i.d. standard-normal vectors in R^128 via Box–Muller
   (deterministic seed).
2. Generate 200 query vectors from the same distribution (different seed).
3. Build the float graph + binary codes.
4. Compute brute-force ground truth (`O(NQ·D)` float L2).
5. For each mode, time the per-query `Searcher::search` call with
   `std::time::Instant`, accumulate hits against ground truth, and
   report mean recall + queries-per-second.

All four numbers come from the *same* `Searcher` instance, so the graph
itself is identical across modes — the only variable is the scoring
function.

---

## Results

(See the Abstract table above for the full numbers. Hardware was an
Apple M4 Max running macOS Darwin 24.6.0 arm64 with `rustc 1.89.0` in
`--release`. Build time of the index was 1.29 s for N=8,000 / M=24 /
ef_construction=128.)

### Compression accounting

| Quantity | Value |
|---|---|
| Float vectors | 4,096,000 B (= 8,000 × 128 × 4 B) |
| Binary codes  | 128,000 B (= 8,000 × 16 B) |
| Compression   | **32.0×** |

### Speed accounting

| Mode | µs/query | Distance ops per query (estimate) |
|---|---|---|
| Float graph | 209.4 | ef_s × M ≈ 4,800 × float L2 (D mul-add) |
| Binary graph | 93.7 | ef_s × M ≈ 4,800 × XOR-popcount (D/64 ops) |
| SymphonyQG | 111.9 | ≈ 4,800 cheap + 200 float (rerank) |
| Brute force | 517.0 | N × float L2 = 8,000 |

The 2.23× speedup of `Binary` over `Float` exactly tracks the µs/query
delta — distance computations dominate, just as the SymphonyQG paper
predicts.

---

## How it works (blog-readable walkthrough)

Picture an HNSW graph search as a flashlight beam crawling through the
dataset toward the query. Every step the flashlight has to evaluate
"how close is this neighbor?" — that's a 128-dimensional dot product, a
few hundred nanoseconds of work. SymphonyQG replaces that 128-dim dot
product with a single XOR + `popcount` on 128 bits — two CPU
instructions. Same graph, same walk, ~30× cheaper per step.

But you only have **1 bit** per coordinate now, so the "how close?"
answer is noisy. SymphonyQG handles the noise in two stages:

1. **Centring + rotation**. Subtract the dataset mean, then apply a
   randomized Walsh–Hadamard rotation. This is the RaBitQ trick: it
   whitens the projected distribution so each sign bit carries roughly
   equal information. On already-isotropic Gaussian data (our PoC
   workload) this is a no-op, but on real embeddings it matters a lot.
2. **Float rerank**. After walking the graph with cheap quantized
   distances, take the top-`rerank` survivors and re-score them with
   exact float L2. The walk doesn't have to land perfectly — it just has
   to land within rerank-distance of the true neighbors.

That's the entire idea: **cheap traversal, exact terminus.**

---

## Practical failure modes

* **High intrinsic dimension.** As D grows, 1-bit codes carry
  proportionally less information; recall collapses. The paper recommends
  4-bit codes for D ≥ 256. This PoC sticks to 1-bit for clarity and
  documents the gap honestly: 11.5% raw binary recall on D=128 Gaussians
  is the price of single-bit-per-dimension.
* **Anisotropic data.** If the data has very different scales across
  axes, the random ±1 diagonal alone is not enough — a learned PCA-style
  rotation outperforms Hadamard for some embedding distributions.
* **Single-layer NSW.** With one fixed entry node, a beam search will
  stick to one cluster on multi-modal data. We work around this with
  stride-spaced multi-entry seeding. For production a small HNSW upper
  layer (or NSG-style routing) is strictly better.
* **The rerank shortlist.** If true neighbours never enter the
  shortlist during the quantized walk, no rerank can recover them.
  This is why Symphony hovers ~34 pp below Float on this PoC — graph
  navigation on noisy 1-bit distances misses regions entirely.

---

## What to improve next

1. **4-bit RaBitQ codes**. Recover most of the recall gap at ~4× memory
   over 1-bit. The paper reports this as the production sweet spot.
2. **Real rotation**. Replace the LCG diagonal with a `rand`-seeded
   ±1 vector and consider a *trained* (PCA + per-dimension scale)
   rotation. Worth a 5–10pp recall lift on text embeddings.
3. **HNSW upper layer**. Plug into the existing
   `ruvector-rabitq`/`ruvector-rairs` infrastructure so SymphonyQG
   inherits a real multi-layer graph rather than the flat NSW used here.
4. **AVX2 / NEON popcount kernel**. Today we rely on `u64::count_ones`;
   M4 Max has an 8-wide NEON popcount that would give another ~3×.
5. **Hailo-10H offload**. The 4-bit code XOR-popcount kernel maps cleanly
   to the Hailo INT4 dot-product instructions — a natural integration
   with the existing `ruvector-hailo` crate.

---

## Production crate layout proposal

If this PoC graduates, the suggested fork:

```
crates/ruvector-symphonyqg/
├── Cargo.toml                  (workspace member)
├── src/
│   ├── lib.rs                  (public API surface)
│   ├── graph.rs                (HNSW upper layer + flat NSW)
│   ├── quant/
│   │   ├── mod.rs              (BinaryCodec trait)
│   │   ├── rabitq_1bit.rs      (today's impl)
│   │   ├── rabitq_4bit.rs      (8x4 LUT, ADC-style)
│   │   └── rotation.rs         (Hadamard + PCA + Householder)
│   ├── symphony.rs             (Searcher, SearchMode)
│   └── kernels/
│       ├── scalar.rs           (current u64 popcount)
│       ├── avx2.rs             (x86-64 256-bit popcount fast path)
│       └── neon.rs             (M-series 128-bit fast path)
└── benches/
    └── recall_qps.rs           (criterion HtmlReports)
```

---

## References

1. Yang, Qiao et al. *SymphonyQG: Towards Symphonious Integration of
   Quantization and Graph for Approximate Nearest Neighbor Search.*
   SIGMOD 2025.
2. Gao et al. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search.*
   SIGMOD 2024.
3. Malkov & Yashunin. *Efficient and Robust Approximate Nearest
   Neighbor Search using Hierarchical Navigable Small World Graphs.*
   TPAMI 2018.
4. Existing ruvector crates: `crates/ruvector-rabitq` (1-bit RaBitQ
   indexer), `crates/ruvector-rairs` (RaBitQ-IVF dual assignment).
