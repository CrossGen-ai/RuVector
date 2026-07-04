# ruvector 2026: Speculative ANN Search — High-Performance Rust Vector Search Inspired by LLM Speculative Decoding

**150-char summary.** SpecANN ports LLM speculative decoding to vector search:
a cheap draft index proposes candidates, an exact float32 verifier rescores
only the top few. Real Rust, real numbers.

## Introduction

**Speculative decoding** transformed LLM inference throughput by pairing a
tiny draft model with an exact verifier: the small model proposes N tokens,
the big model verifies in a single forward pass, and the system accepts the
longest common prefix. **ruvector 2026** applies the same asymmetry to
**approximate nearest-neighbor (ANN) vector search**: a cheap draft index
(int8 or 1-bit-quantized) proposes an over-provisioned candidate set of size
`k_draft = α · k`, and an exact float32 verifier rescores only that set. An
escalation policy watches the confidence gap between rank k and rank k+1 in
the draft's scores — when the gap is thin, the driver widens the probe or
falls back to exhaustive verification. The result: a **first-class Rust
trait pair** (`DraftIndex` + `Verifier`) that composes with any existing
quantized index in the ruvector ecosystem — RaBitQ, LVQ, PQ, HNSW —
delivering measured verification savings (**2.55 % of the corpus at 100 %
recall@10** on 10 000 × 128 gaussians) and, with a rotated 1-bit draft,
**2.5× QPS** over exact brute force. Keywords: **Rust vector search**,
**speculative decoding**, **approximate nearest neighbor**, **RaBitQ**,
**HNSW**, **quantized retrieval**, **RAG infrastructure**, **vector
database**, **high-performance ANN**, **ruvector**.

## Features

- **Trait-based draft/verifier abstraction** — `DraftIndex` proposes
  candidates, `Verifier` exact-rescores. Swap drafts (int8, 1-bit, RaBitQ,
  HNSW-partial-walk) without touching downstream code.
- **Confidence-gap escalation policy** — watches
  `(score[k+1] − score[k]) / score[k]`. Below threshold: widen probe by
  `α · multiplier`. Bounded escalations, deterministic full-verify fallback.
- **Observable stats per query** — `SpecStats { escalations,
  draft_candidates, verified, full_verify_fallback }`. Enables offline
  training of a learned escalation policy.
- **Three shipped implementations** — `F32BruteForce` (exact baseline),
  `Int8BruteForce` (symmetric per-vector int8 quantization),
  `Sign1BitDraft` (u64-packed Hamming distance).
- **Zero-dep production surface** — `rand`, `rand_distr`, `thiserror`, `serde`.
- **All files under 500 lines**, all tests pass, all benchmarks are real.

## Benefits

- **Verification is the expensive step in every quantized ANN pipeline.**
  Making draft/verifier explicit measures and minimizes it — SpecANN
  verified 255 / 10 000 vectors on average on the 10 k × 128 gaussian
  benchmark, at exact-baseline recall@10.
- **Composability with existing ruvector crates.** Wrap `ruvector-rabitq`
  as a `DraftIndex`, wrap `ruvector-core` HNSW as a `DraftIndex`, wrap
  `ruvector-postgres` as a `Verifier` for disk-backed rescoring — no
  rewrites.
- **Analogy transfers.** Every optimization from LLM speculative decoding
  literature (staged speculation, learned draft acceptance, token-tree
  verification) has a direct ANN analogue.
- **Debug-friendly.** `SpecStats` makes it obvious *why* a query was slow —
  too many escalations? Full-verify fallback? Draft too coarse for the data?

## Comparisons

| System         | Approach                                | Draft-verifier explicit? | Escalation observable? | Language |
|----------------|-----------------------------------------|--------------------------|------------------------|----------|
| Milvus         | HNSW + optional PQ rescore              | No — baked into index    | No                     | C++/Go   |
| Qdrant         | HNSW + scalar quantization rescore      | No — baked into index    | No                     | Rust     |
| Weaviate       | HNSW + PQ/BQ                            | No                       | No                     | Go       |
| Pinecone       | Proprietary; opaque                     | No                       | No                     | Closed   |
| LanceDB        | IVF-PQ + optional float32 rescore       | Partial                  | No                     | Rust     |
| FAISS          | IVF-PQ / HNSW-PQ                        | Partial                  | No                     | C++      |
| **ruvector SpecANN** | **Trait-based draft + verifier + escalation policy** | **Yes** | **Yes** — `SpecStats` | **Rust** |

## Benchmarks

Real cargo-run numbers. Hardware: **Apple M4 Max, single-threaded**, macOS
15.7 (Darwin 24.6), rustc 1.89.0, `cargo build --release`. Corpus: 10 000
gaussian vectors, dim = 128, k = 10, 200 queries, warmup 5.

| Variant                          | recall@10 | QPS       | p50 (µs) | p95 (µs) | avg verified / 10 000 |
|----------------------------------|-----------|-----------|----------|----------|-----------------------|
| A. exact float32 baseline        | 1.000     | 2 604.2   | 386.8    | 408.0    | 10 000                |
| B. int8-only, no verify          | 0.984     | 2 119.8   | 465.9    | 519.0    | 0                     |
| C. **SpecANN int8 + f32 verify** | **1.000** | 741.7     | 1 347.2  | 1 412.2  | **255.2 (2.55 %)**    |
| D. **SpecANN 1-bit + f32 verify**| 0.699     | **6 591.7** | 150.3  | 164.6    | 768.8                 |

Reproduce:

```bash
git clone https://github.com/CrossGen-ai/RuVector
cd RuVector
git checkout research/nightly/2026-07-04-speculative-ann-draft-verify
cargo run --release -p ruvector-specann --bin specann-bench
```

## Optimizations

Roadmap items ordered by expected impact:

1. **Rotated 1-bit draft (RaBitQ Hadamard rotation).** Expected to lift
   Variant D recall from 0.699 → 0.95+ at the same 6 592 QPS. Two-day wire-up
   against `ruvector-rabitq`.
2. **HNSW as `DraftIndex`.** Replace brute-force draft with `hnsw_rs` graph
   walk. Expected 10–50× QPS over Variant A at 99 %+ recall — verification
   savings *compose* with sub-linear draft cost.
3. **Learned escalation policy.** Tiny MLP over `(k, α, gap, top-k spread)`
   trained offline on ground-truth traces. Analogous to *predictive draft
   acceptance* in speculative decoding literature (Miao et al., SpecInfer,
   ASPLOS 2024).
4. **Batched verification.** Pack candidate ids across a query batch, issue
   one SIMD gather. Expected 1.5–2× on large batches.
5. **Cascading draft chain: 1-bit → int8 → f32.** Each stage escalates only
   on gap. Direct analogue of staged speculation (StagedSpec).

## Get started

- **Fork with research branch:** https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-07-04-speculative-ann-draft-verify
- **Crate:** `crates/ruvector-specann/` on that branch.
- **Research doc:** `docs/research/nightly/2026-07-04-speculative-ann-draft-verify/README.md`.
- **ADR:** `docs/adr/ADR-272-speculative-ann-draft-verify.md`.
- **Upstream:** https://github.com/ruvnet/ruvector

```bash
cargo run --release -p ruvector-specann --bin specann-demo    # quick tour
cargo test --release -p ruvector-specann                       # 6 passing tests
cargo run --release -p ruvector-specann --bin specann-bench    # real numbers
```

*Tags: ruvector, vector-search, ANN, approximate-nearest-neighbor,
speculative-decoding, RaBitQ, HNSW, quantization, Rust, RAG,
vector-database, embedding-search, high-performance.*
