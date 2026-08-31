# ruvector 2026: Prefetch-Guided IVF Scan — High-Performance Rust Vector Search

**tl;dr — RuVector ships portable software prefetch (x86_64 + aarch64, stable Rust) for IVF ANN scans. Up to 3.9× faster on strided gather workloads; correctly regresses to no-op on contiguous scans where the CPU already wins.**

Software prefetch in ANN (approximate nearest neighbor) search is one of those "everyone says it helps" perf myths. This crate measures the real answer on modern silicon (Apple M4 Max, arm64, `rustc 1.89.0`), gives you a portable one-line abstraction, and honestly documents where it wins and where it doesn't.

Part of the [ruvector](https://github.com/ruvnet/RuVector) high-performance Rust vector search stack. Contributed via the CrossGen-AI fork.

## Features

- **Portable software prefetch** on stable Rust, no `nightly`, no C compiler dep. One call, one instruction:
  - `_mm_prefetch(ptr, _MM_HINT_T0)` on x86_64
  - `prfm pldl1keep, [ptr]` (stable inline asm) on aarch64 (Apple M-series, Graviton, Ampere)
  - No-op fallback on other targets
- **Adaptive lookahead** — picks a per-vector prefetch distance `K` from vector byte size, clamped `[1, 32]`, so the same code auto-tunes across dim ∈ [16, 4096].
- **Five scan kernels** — three contiguous variants (baseline / fixed-lookahead / adaptive) and two strided-gather variants (baseline / prefetched), all sharing the same top-k structure so results are byte-identical.
- **Correctness-preserving by construction** — prefetch is architecturally side-effect-free; unit + integration tests enforce identical top-k across every variant on every configuration.
- **No dependencies.** Not even `rand` — internal xorshift64 for reproducible benches.
- **Fits in every RuVector storage tier** — `PostingList = (dim, Vec<f32>)` is a trivial newtype the existing IVF-flat / IVF-PQ / rabitq crates can adopt without a refactor.

## Benefits

- **Real numbers, honestly reported.** No cherry-picked graphs. This ships with the raw benchmark output (`raw-runs.txt`) and reports the *regression* on contiguous scans as loudly as the 3.9× win on strided scans.
- **Gate at the call site, not build time.** `ScanMode::{Off, Contiguous, Strided}` — decide per query, not per binary.
- **No unsafe leaks.** The one `unsafe` block wraps only the architecturally-guaranteed no-fault property of PRFM / PREFETCH. Safe wrappers everywhere else.
- **Small.** Under 500 lines per file, workspace-clean.

## Comparisons

| Library | Portable x86 + ARM SW prefetch | Adaptive lookahead | Stable Rust | Gated on access pattern | Open-source |
|---|---|---|---|---|---|
| **ruvector-prefetch-guided-ivf-scan** | Yes (`_mm_prefetch` + `PRFM`) | Yes | Yes | Yes | Yes (MIT/Apache-2.0) |
| FAISS 1.10 (Meta) | x86 only | No (fixed la=1) | N/A (C++) | No | Yes |
| DiskANN (Microsoft) | Page-level `madvise` only | N/A | N/A (C++) | Sector boundary | Yes |
| HNSW.rs (rust-cv) | None | No | Yes | N/A | Yes |
| Milvus 2.5 / Knowhere 2.7 | x86-tuned; ARM path silent | Per-list knob | N/A (C++) | Manual | Yes |
| Qdrant 1.14 | `__builtin_prefetch` (may no-op on ARM) | No | Yes | No | Yes |
| Pinecone | Closed source | ? | N/A | ? | No |
| Weaviate | None | No | N/A (Go) | No | Yes |
| LanceDB | None | No | Yes | No | Yes |

## Benchmarks

**Hardware**: Apple M4 Max, Darwin 24.6.0 arm64. `rustc 1.89.0 (29483883e 2025-08-04)`. Release profile: `opt-level=3 lto=thin codegen-units=1`.

### Strided scan (the case that matters for real IVF-PQ workloads)

Prime-stride gather (stride coprime with `n`), which mimics cross-cluster fanout that the hardware stream prefetcher cannot help with.

| dim | n vectors | Baseline ns/query | Best prefetch ns/query | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 10 000 | 423 843 | 394 093 (la=16) | **1.075×** |
| 128 | 50 000 | 3 916 221 | 1 908 428 (la=16) | **2.052×** |
| 128 | 200 000 | 30 189 803 | 7 736 608 (la=8) | **3.902×** |
| 256 | 20 000 | 1 999 681 | 1 469 128 (la=8) | **1.361×** |

Speedup grows with `n` because larger `n` overflows M4 Max's ~16 MB L2 and every gather becomes a cold miss. At dim=128 n=200 000, prefetched drops to ~38 ns/dot — the compute floor.

### Contiguous scan (where the hardware prefetcher already wins)

| dim | n vectors | Baseline ns/query | Best SW prefetch | vs baseline |
|---:|---:|---:|---:|---:|
| 64 | 10 000 | 199 542 | 199 087 | 1.002× |
| 128 | 10 000 | 375 435 | 374 335 | 1.003× |
| 256 | 10 000 | 699 738 | 736 093 | 0.951× |
| 512 | 10 000 | 1 411 866 | 1 543 700 | 0.915× |
| 128 | 200 000 | 7 161 917 | 7 577 735 | 0.945× |

**Contiguous verdict**: neutral-to-harmful. Apple Silicon's stream prefetcher already saturates. Extra `PRFM` instructions consume issue slots without hiding any additional latency. This is why the crate ships a `ScanMode` gate rather than compiling prefetch in unconditionally.

## Optimizations

- Manual 4-way unrolled L2² kernel that lowers to NEON `FMLA` on Apple Silicon and AVX/AVX2 with `target-cpu=native` on x86_64.
- Prefetch every cache line of the target vector (8 lines for a 512-byte dim=128 vector) — one `PRFM` per line, cheap vs. the ~200 cycle miss it hides.
- Adaptive lookahead formula: `K = ceil(budget_lines * 64 / bytes_per_vec)` clamped `[1, 32]`. Empirically within 2 % of hand-picked optimum at dim=128 strided.
- Locality hints exposed (`L1` / `L2`) so callers with deeper reorder buffers can pull L2 hints when their loop body has more compute slack.
- Bounded linear-scan `TopK` (no binary heap) — the branch predictor eats k ≤ 32 for lunch.
- No allocations in the hot path (top-k struct is preallocated).

## Get Started

Fork branch: <https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-31-prefetch-guided-ivf-scan>

```bash
git clone https://github.com/CrossGen-ai/RuVector
cd RuVector
git checkout research/nightly/2026-08-31-prefetch-guided-ivf-scan

# Build and test
cargo test --release -p ruvector-prefetch-guided-ivf-scan

# Run the benchmark yourself
cargo run --release --example bench_scan -p ruvector-prefetch-guided-ivf-scan
```

Design notes and full research write-up live at `docs/research/nightly/2026-08-31-prefetch-guided-ivf-scan/`. Architecture Decision Record: `docs/adr/ADR-342-prefetch-guided-ivf-scan.md`.

Upstream: <https://github.com/ruvnet/RuVector>. This is a research contribution to the ruvector high-performance Rust vector-search project.
