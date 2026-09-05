# ruvector 2026: Cascade-ADC — High-Performance Rust Vector Search with Coherent Two-Level PQ

**Summary (150 chars):** Cascade-ADC is a coherent 4-bit prune + 8-bit refine scanner for Rust PQ vector search that preserves recall and beats plain 8-bit throughput.

Cascade Asymmetric Distance Computation (Cascade-ADC) is a two-stage progressive-precision Product Quantisation scanner for the [ruvector](https://github.com/ruvnet/RuVector) Rust vector search engine. It sweeps every code at 4-bit precision using a tiny L1-resident lookup table, then refines only the top ρ·N survivors at 8-bit precision. Because the 4-bit codebook is a *coherent coarsening* of the 8-bit codebook (k-means over its own centroids), Stage-1 rankings correlate strongly with Stage-2 rankings — so pruning is safe and recall is preserved.

## Features

- **Coherent two-level codebook.** K8 = 256 fine centroids per subspace and K4 = 16 coarse centroids per subspace, derived from a second Lloyd pass over the fine ones. A `parent[j][fine] → coarse` table makes the mapping explicit.
- **Progressive-precision scan.** Stage-1 sweeps 4-bit codes over all N; Stage-2 refines top ρ·N at 8-bit.
- **Zero unsafe, zero SIMD intrinsics.** Pure scalar Rust that auto-vectorises. SIMD is a future orthogonal improvement.
- **Composable trait.** `Scanner` trait with three shipping impls: `FullEightBitScanner`, `FullFourBitScanner`, `CascadeScanner`.
- **Recall-floor guarantee.** `CascadeScanner::with_floor(rho, top_t)` never drops below `top_t` survivors even for tiny ρ.
- **O(N) partition, not O(N log T) heap.** Stage-1 uses `select_nth_unstable_by` so the hot loop stays branchless.

## Benefits

- **Recall preservation.** Cascade at ρ = 0.05 lands within 0.3 pp of the 8-bit ceiling; ρ ≥ 0.10 matches it exactly.
- **Throughput win.** Measured +13 % over plain 8-bit at ρ = 0.05.
- **Composable.** Plug into any existing scan path (`ruvector-pq-search`, `ruvector-rabitq`, `ruvector-fused-rabitq-residual`) via the `Scanner` trait.
- **Deterministic.** Seeded training and scan; benchmarks and tests are reproducible bit-for-bit.
- **Portable.** No SIMD intrinsics, no OS-specific code — runs on ARM (M-series), x86_64, WebAssembly with cargo alone.

## Comparisons

| System | Codec | Coherent coarse pre-filter? | Recall-preserving prune? | Language |
|--------|-------|-----------------------------|--------------------------|----------|
| **ruvector Cascade-ADC** | **PQ8 + coherent PQ4** | **yes** | **yes (≥ ρ=0.05)** | **Rust** |
| Milvus 2.4 PQFastScan | PQ4 SIMD | no (standalone) | drop-in replace, not cascade | C++ |
| Qdrant multi-tier quant | scalar / PQ | no coherence between tiers | tier chosen per collection | Rust |
| Weaviate PQ | PQ8 flat | no | no | Go |
| Pinecone (managed) | proprietary | undisclosed | undisclosed | closed |
| FAISS IVFPQ + refine | PQ8 + float refine | no | float refine, expensive | C++ |
| LanceDB PQ-refine | PQ8 + float refine | no | float refine, expensive | Rust |

Cascade-ADC is the only in-tree pattern combining a *coherent* companion codebook with a partition-based Stage-1 selection.

## Benchmarks

Hardware: **Apple M4 Max**, macOS 15.6, rustc 1.89.0, release profile, **single thread, scalar (no SIMD)**.

Corpus: n = 200 000 Gaussian vectors, d = 128, m = 32 subspaces, k = 10.

```
  full-8bit         qps=411.9   recall@10=0.4625   32 B/vec
  full-4bit         qps=497.3   recall@10=0.1685   16 B/vec
  cascade-rho0.05   qps=464.3   recall@10=0.4595   48 B/vec   ← +13% vs 8-bit
  cascade-rho0.10   qps=428.3   recall@10=0.4625   48 B/vec
  cascade-rho0.20   qps=398.5   recall@10=0.4625   48 B/vec
```

Full 4-bit alone destroys recall (0.169). Cascade at ρ = 0.05 preserves it (0.4595 vs 0.4625) *and* beats plain 8-bit throughput.

## Optimisations

- **L1-resident coarse LUT.** m × 16 × 4 B = 2 kB at m=32 — trivially fits in every modern L1.
- **Unrolled 4-bit unpack.** Two codes per byte handled per iteration; parity branch eliminated from the hot loop.
- **Partition, not heap.** `select_nth_unstable_by` is O(N) and — critically — keeps Stage-1 branchless. A size-T heap costs O(N log T) push overhead when T = ρ·N is large.
- **Query-local LUT reuse.** LUT is built once per query and reused across all N scans; zero allocation in the inner loop after warm-up.
- **No unsafe, no intrinsics.** Compiler auto-vectorisation is enough today; SIMD kernels (NEON / AVX2) are a planned Stage-1 backend for a 4–8× headroom win.

## Get Started

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-09-05-cascade-adc-scan
cargo test  --release -p ruvector-cascade-adc
cargo run   --release -p ruvector-cascade-adc --bin cascade-bench
```

- **Crate**: [`crates/ruvector-cascade-adc/`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-09-05-cascade-adc-scan/crates/ruvector-cascade-adc)
- **ADR**: [`docs/adr/0001-cascade-...md`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-09-05-cascade-adc-scan/docs/adr)
- **Research doc**: [`docs/research/nightly/2026-09-05-cascade-adc-scan/README.md`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-09-05-cascade-adc-scan/docs/research/nightly/2026-09-05-cascade-adc-scan)
- **Upstream**: [github.com/ruvnet/RuVector](https://github.com/ruvnet/RuVector)

*Tags: rust vector search, product quantization, PQ ADC, cascade scan, ANN, approximate nearest neighbor, ruvector, high-performance retrieval, IVFPQ, RaBitQ, PQFastScan, DiskANN.*
