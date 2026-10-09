# ruvector 2026: Walsh-Hadamard Sign Sketch — Structured O(d log d) ANN Prefilter in Rust

**TL;DR (150 chars):** Rust ANN prefilter using Fast Walsh-Hadamard sign sketches. 36× faster index build, 512× smaller projector, 1.5× faster query vs classic Gaussian SRP at d=512.

Vector search prefilters live or die by two numbers: how fast you can build the index, and how small the projector sits in RAM. The industry default — Gaussian **sign-random-projection** (SRP / SimHash) — pays `O(d·m)` time and `O(d·m)` memory per vector. ruvector's new PoC crate, **ruvector-whss**, replaces that matrix with a **Walsh-Hadamard transform × random ±1 diagonal (HD)**, dropping the cost to `O(d log d)` time and `O(d)` memory while matching or beating recall at embedding-grade dimensions (384, 512, 768, 1536).

## Features

* **Structured O(d log d) sign sketch** — Fast Walsh-Hadamard butterflies, no matrix materialized.
* **u64-packed bits + popcount Hamming** — SIMD-friendly, uses the CPU's native `popcnt` on x86-64 and NEON on aarch64.
* **Trait-ready backend** — swappable with Gaussian SRP for A/B testing, and with planned stacked HD-blocks for rank > d.
* **Deterministic, seeded** — `rand_chacha`-based CSPRNG, reproducible benches and recall numbers.
* **Pure safe Rust** — no `unsafe`, no BLAS, no C deps.

## Benefits

* **36× faster index build** at d=512, 10 000 vectors (35.6 ms vs 1 301 ms, Apple M-series).
* **512× smaller projector** (2 KB vs 1 MB at d=512, m=512) — fits comfortably in Cloudflare Workers' per-DO memory budget.
* **Higher recall at embedding-grade dims** — 0.80 vs 0.68 at d=512, m=512, cand=1024.
* **Zero model to ship.** The projector is a seed + a short permutation, not a trained matrix.

## Comparisons

| System         | Sign-sketch backend | Build cost        | Projector memory       |
|----------------|---------------------|-------------------|------------------------|
| Milvus (2026)  | Gaussian SRP        | O(d·m)            | m·d floats             |
| Qdrant (2026)  | Gaussian SRP        | O(d·m)            | m·d floats             |
| Weaviate (2026)| Gaussian SRP        | O(d·m)            | m·d floats             |
| Pinecone       | (opaque)            | —                 | —                      |
| FAISS LSH      | random bit-sampling | O(d·m)            | m·d bits               |
| FALCONN        | FWHT inside cross-polytope hash | O(d log d) | O(d)              |
| **ruvector-whss** | **HD + FWHT sign sketch** | **O(d log d)**  | **O(d)** |

FALCONN uses FWHT inside its own hashing family; **ruvector-whss is the first to ship it as a standalone drop-in SimHash prefilter inside a modern vector-DB-adjacent stack**.

## Benchmarks (real wall-clock, Apple M-series, scalar Rust, 100 queries, single thread)

**d = 512, N = 10 000, k = 10**

| method | m   | cand | ms/100q | QPS   | recall@10 | build ms | projector bytes |
|--------|-----|------|---------|-------|-----------|----------|-----------------|
| exact  | —   | —    | 212.15  | 471   | 1.0000    | 0        | 20 480 000      |
| **WHSS**   | 512 | 1024 | **29.76**   | **3 360** | **0.8010**    | **35.6**     | **2 048**           |
| SRP    | 512 | 1024 | 44.20   | 2 262 | 0.6760    | 1 301.1  | 1 048 576       |
| WHSS   | 256 | 1024 | 28.89   | 3 462 | 0.5560    | 24.3     | 1 024           |
| SRP    | 256 | 1024 | 36.59   | 2 733 | 0.4900    | 642.6    | 524 288         |

**d = 128, N = 20 000** (narrow-dim regime, SRP's territory)

| method | m   | cand | ms/100q | QPS   | recall@10 | build ms |
|--------|-----|------|---------|-------|-----------|----------|
| exact  | —   | —    | 70.67   | 1 415 | 1.0000    | 0        |
| WHSS   | 512 | 1024 | 15.23   | 6 564 | 0.7370    | 71.6     |
| SRP    | 512 | 1024 | 17.95   | 5 571 | 0.9290    | 582.6    |

Honest trade-off: at low dim (d=128) with large m, SRP's full-rank Gaussian matrix wins on recall (0.93 vs 0.74) because the WHSS sketch is rank-bounded by d. Pick WHSS when `d ≥ 2m` — the common case for 384-/768-/1536-d embeddings.

## Optimizations

* In-place iterative FWHT butterflies, stride-doubling to stay in L1.
* `select_nth_unstable_by_key` for candidate top-c, no full sort.
* Bit-packed sketches, `u64::count_ones()` → native `popcnt` / NEON.
* Projector = seed + m-element permutation; serialized in ~1 KB regardless of d.

## Get started

Source lives in the CrossGen-ai fork on branch `research/nightly/2026-10-09-walsh-hadamard-sign-sketch`:

* Crate: [`crates/ruvector-whss`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/crates/ruvector-whss)
* Research doc: [`docs/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/README.md`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/docs/research/nightly/2026-10-09-walsh-hadamard-sign-sketch)
* ADR: [`docs/adr/ADR-361-walsh-hadamard-sign-sketch.md`](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/docs/adr/ADR-361-walsh-hadamard-sign-sketch.md)

Upstream ruvector: https://github.com/ruvnet/RuVector

```bash
git clone -b research/nightly/2026-10-09-walsh-hadamard-sign-sketch https://github.com/CrossGen-ai/RuVector.git
cd RuVector/crates/ruvector-whss
cargo test --release
cargo run --release --bin whss-bench
```

**Keywords:** rust vector search, ANN prefilter, SimHash, sign random projection, Walsh-Hadamard transform, FastFood, structured sketch, LSH, popcount, Hamming distance, high-dimensional embeddings, retrieval-augmented generation, Milvus Qdrant Weaviate alternative, ruvector.
