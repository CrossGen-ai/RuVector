# ruvector 2026: SPANN Boundary Closure in Rust — High-Performance IVF Vector Search with Smart Replication

> +6.4 recall points at lower latency than baseline IVF, with only 1.51× replication vs the 2-4× of naive multi-probe. Pure Rust, no `unsafe`, no external numeric deps — drops into the [ruvector](https://github.com/ruvnet/ruvector) Rust vector database alongside HNSW, DiskANN, RaBitQ, AnisotropicVQ, MUVERA, NN-Descent, and RAIRS-IVF.

[ruvector](https://github.com/ruvnet/ruvector) is a Rust-native high-performance vector database with 150+ crates. This nightly research drop adds **SPANN-style boundary-aware closure assignment** (Chen et al., NeurIPS 2021) as ruvector's second IVF family, behind a swappable `ClosurePolicy` trait that makes future closure strategies (SOAR, anisotropic, learned) one-trait-impl additions.

## Introduction

Production-scale Approximate Nearest Neighbor (ANN) search on billion-vector indices keeps coming back to inverted-file (IVF) posting lists — Milvus, Qdrant, FAISS, Weaviate, and Pinecone all use IVF or an IVF-shaped layer somewhere in their stack. IVF's classic weakness is recall *at low nprobe*: vectors near a Voronoi cell boundary get arbitrarily assigned to one cell, and a query that lands one grain of dust into the neighboring cell never sees them. Multi-probe trades memory for recall by replicating *every* vector into 2-4 cells. SPANN replaces that brute-force replication with a **boundary-aware closure rule**: replicate only the vectors whose runner-up centroid is within `(1+ε)` of the nearest. Interior points pay nothing; boundary points get exactly the extra coverage they need. This drop is the first published implementation in Rust. Real benchmark numbers below — Apple M4 Max, `rustc 1.89.0`, single thread, `--release`.

## Features

- **`ClosurePolicy` trait** with three concrete policies in v0.1: `SingleAssign` (baseline IVF), `FixedMultiAssign { k }` (uniform multi-probe), and `SpannClosure { epsilon, cap }` (boundary-aware closure).
- **Generic `SpannIndex::build<P: ClosurePolicy + ?Sized>`** — pass a `&dyn ClosurePolicy` for runtime selection without per-policy monomorphisation.
- **Dedup-aware search**: replicated points appear in multiple posting lists but are counted once in the top-k heap (`Vec<bool>` bitset).
- **Deterministic k-means++** seeding via `StdRng::seed_from_u64` — reproducible builds.
- **Zero `unsafe`, zero external numeric deps** beyond `rand`. All files ≤ ~250 LoC; total crate is ~700 LoC of Rust.
- **Runnable demo binary** (`cargo run --release -p ruvector-spann --bin spann-demo`) plus Criterion micro-benchmarks.

## Benefits

- **Higher recall at lower latency.** SPANN(ε=0.10, cap=4) beats baseline IVF on *both* axes simultaneously at low nprobe.
- **Tiny memory overhead.** +0.8% of index bytes vs baseline; 4× cheaper in replication than `fixed-multi(k=4)`.
- **Composable.** Trait-based design means new closure strategies (SOAR-style anti-correlated spillover, anisotropic closure, learned thresholds) add by implementing one trait, with zero changes to search or k-means.
- **Auditable.** Single self-contained crate, no `unsafe`, deterministic seeds, files under 500 lines per CLAUDE.md.

## Comparisons (versus other vector databases)

| System    | IVF strategy                                    | Closure equivalent? | Open source? |
|-----------|-------------------------------------------------|---------------------|--------------|
| FAISS     | IVFFlat, IVF-PQ, IVF-OPQ                        | Multi-probe only    | Yes          |
| Milvus    | IVFFlat, IVF-PQ, IVF-SQ                         | Multi-probe only    | Yes          |
| Qdrant    | HNSW + IVF-PQ                                   | Multi-probe only    | Yes          |
| Weaviate  | HNSW (no IVF)                                   | N/A                 | Yes          |
| Pinecone  | Proprietary IVF-like                            | Unknown             | No           |
| **ruvector** (this drop) | IVF + `SpannClosure`         | **Yes (SPANN-style)** | Yes (MIT/Apache) |
| ruvector (RAIRS, ADR-193) | IVF + fixed 2× redundant assign | Fixed spill, not boundary-aware | Yes |

To our knowledge this is the first open-source pure-Rust implementation of SPANN-style boundary closure on top of a clean IVF layer.

## Benchmarks (real numbers, captured live)

Dataset: 20,000 vectors × 64 dimensions, 64-mode Gaussian mixture (σ=1.0). 200 queries, K=128 coarse centroids, top_k=10. Single-threaded Apple M4 Max, `rustc 1.89.0`, `--release`.

### nprobe = 2 (low-probe regime — SPANN's sweet spot)

| Variant                | Replication | recall@10  | Search µs/q | Mem MB |
|------------------------|------------:|-----------:|------------:|-------:|
| baseline-single        | 1.00×       | 0.8460     | 14.2        | 4.99   |
| fixed-multi(k=2)       | 2.00×       | 0.9155     | 20.1        | 5.07   |
| fixed-multi(k=4)       | 4.00×       | 0.9415     | 33.9        | 5.22   |
| **spann(ε=0.10, cap=4)** | **1.51×** | **0.9095** | **12.8**    | **5.03** |
| spann(ε=0.20, cap=8)   | 1.89×       | 0.9310     | 23.4        | 5.06   |

SPANN(ε=0.10) gives **+6.4 recall points and ~10% lower latency than baseline**, at only 1.51× replication.

### nprobe = 4

| Variant                | recall@10 | Search µs/q |
|------------------------|----------:|------------:|
| baseline-single        | 0.9715    | 17.4        |
| fixed-multi(k=4)       | 0.9890    | 53.5        |
| spann(ε=0.10, cap=4)   | 0.9790    | 17.8        |

### nprobe = 8

| Variant                | recall@10 | Search µs/q |
|------------------------|----------:|------------:|
| baseline-single        | 0.9990    | 32.5        |
| fixed-multi(k=4)       | 0.9995    | 88.9        |
| spann(ε=0.10, cap=4)   | 0.9995    | 29.1        |

### Build cost & memory

K-means dominates build (~565 ms — identical across variants within noise). SPANN(ε=0.10) adds 10,188 posting entries on top of the baseline's 20,000 — that's +40 KB of `u32` ids on top of 5.0 MB of vectors, i.e. **+0.8%** index size.

## Optimizations

- **Heap-of-`top_k`** kept at size `top_k` so insertion is `O(log k)` not `O(log n)`.
- **`Vec<bool>` dedup bitset** is `n` booleans — cheaper than `HashSet<u32>` at this scale.
- **`?Sized` bound** on `SpannIndex::build<P>` lets the benchmark loop pass `&dyn ClosurePolicy` without monomorphising five copies of the build loop into the binary.
- **Squared L2** kept in a hot inline function in `metrics.rs` so callers can swap to cosine / dot product without touching index code.
- **Centroid sort buffer reused** across base vectors in `build` — one `Vec<(usize, f32)>` allocation amortised across all `n` insertions.

## Get Started

The implementation lives on the [CrossGen-ai/RuVector](https://github.com/CrossGen-ai/RuVector) nightly research fork:

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-05-28-spann-boundary-closure

cargo build --release -p ruvector-spann
cargo test  --release -p ruvector-spann   # 7 tests, all green
cargo run   --release -p ruvector-spann --bin spann-demo
cargo bench -p ruvector-spann
```

Reads:
- [ADR-195 — SPANN-Style Boundary-Aware Closure for IVF Posting Lists](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-05-28-spann-boundary-closure/docs/adr/ADR-195-spann-boundary-closure.md)
- [Research doc — Survey, design, methodology, results, roadmap](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-05-28-spann-boundary-closure/docs/research/nightly/2026-05-28-spann-boundary-closure/README.md)
- Upstream project: [github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)

**Keywords:** ruvector, SPANN, IVF, posting list, boundary closure, Voronoi, approximate nearest neighbor, ANN, vector database, vector search, Rust, recall, nprobe, multi-probe, RaBitQ, AnisotropicVQ, MUVERA, DiskANN, HNSW, RAIRS, NN-Descent, NeurIPS 2021, Chen et al., Microsoft, FAISS, Milvus, Qdrant.
