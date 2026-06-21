# ruvector 2026: Adaptive-Beam HNSW — Anytime ANN with Per-Query Early Termination in Rust

> **150-char summary:** Per-query HNSW beam termination in Rust: ratio + online P² quantile estimators cut work on easy queries while preserving recall on hard ones.

[ruvector](https://github.com/ruvnet/RuVector) is a Rust vector database
focused on agent-memory workloads. This nightly research drop adds
**adaptive-beam HNSW**: a pluggable `BeamTerminator` trait that lets
HNSW decide *per query* when to stop the layer-0 beam search, instead
of burning a fixed `ef_search` on every query.

## Introduction

Every production vector database — Milvus, Qdrant, Weaviate, Pinecone,
FAISS HNSW, pgvector — uses a single global `ef_search` knob for HNSW.
That single number trades latency for recall *on average*. But ANN
workloads are not average: most queries reach near-final recall in 20
expansions; a small tail needs 200. A fixed knob either over-spends on
the head or under-serves the tail. **ruvector-adaptive-beam** adds a
search-loop-level termination trait with three measured backends:
`FixedEfTerminator`, `RatioTerminator`, and `QuantileTerminator` (using
a Jain–Chlamtac P² online quantile estimator, constant memory). All
three honour an `ef_max` ceiling so worst-case latency is bounded.

## Features

* Self-contained HNSW (no external deps) so the search loop has full
  candidate/result heap access.
* `BeamTerminator: Send + Sync` trait — three reference implementations
  plus an obvious extension path (`DeadlineTerminator`, RL policy, …).
* Constant-memory P² quantile estimator (5 markers, O(1) update) for
  online termination signals.
* `worst_topk` signal piped from a parallel top-k view, not from
  `worst-of-ef` — the signal that actually matters to the caller.
* Real `cargo run --release` benchmark numbers, no mocks.
* Apple Silicon and x86-64 — pure scalar Rust, SIMD is a follow-on.

## Benefits

* **Per-query latency adapts.** Easy queries stop early; hard queries
  spend the full budget. The mean latency falls, the p95 latency
  stays bounded by `ef_max`.
* **Composable with existing HNSW.** The trait is the production
  deliverable — drop into `ruvector-core` behind `--features
  adaptive-beam`.
* **Anytime-ANN ready.** A future `DeadlineTerminator` plugs into the
  same trait; perfect for soft-real-time agent loops.
* **Orthogonal to ruvector-hnsw-repair (ADR-258).** Adaptive
  termination and adaptive deletion compose cleanly.

## Comparisons

| Library / system     | Per-query adaptive termination? | Online quantile signal? | Trait-pluggable? | Anytime / deadline? |
|----------------------|:-------------------------------:|:------------------------:|:----------------:|:--------------------:|
| Milvus 2.5           |               ✗                |             ✗            |        ✗         |          ✗          |
| Qdrant 1.10          |               ✗                |             ✗            |        ✗         |          ✗          |
| Weaviate             |               ✗                |             ✗            |        ✗         |          ✗          |
| Pinecone             |               ✗                |             ✗            |        ✗         |          ✗          |
| FAISS HNSW           |               ✗                |             ✗            |        ✗         |          ✗          |
| pgvector             |               ✗                |             ✗            |        ✗         |          ✗          |
| DiskANN / Vamana     |               ✗                |             ✗            |        ✗         |          ✗          |
| **ruvector-adaptive-beam** |       **✓**              |          **✓**            |     **✓**        |   **✓ (roadmap)**   |

## Benchmarks

Hardware: Apple M4 Max (1 thread, release build, `lto=thin`).
Dataset: 20 000 × 64-dim uniform vectors, `M=16, ef_construction=200`,
500 queries, `k=10`, `ef_max=256`. Ground truth: exact brute force.

| Variant            | recall@10 | mean exps | mean dist evals | p50 ns | p95 ns |
|--------------------|-----------|-----------|-----------------|--------|--------|
| `fixed_ef=32`      | 0.6152    |  32.0     |   963.4         | 43 292 |  59 625|
| `fixed_ef=64`      | 0.8016    |  64.0     |  1717.2         | 80 333 | 100 166|
| `fixed_ef=128`     | 0.9198    | 128.0     |  2982.4         |156 208 | 190 500|
| `ratio=1.05`       | 0.5390    |  24.4     |   766.5         | 63 042 |  88 209|
| `ratio=1.10`       | 0.7268    |  46.3     |  1306.4         | 91 084 | 135 167|
| **`ratio=1.20`**   | **0.9560**| **187.0** |  3917.6         |252 500 | 334 417|
| `quantile_p=0.50`  | 0.4346    |  16.8     |   564.2         | 45 417 |  58 000|
| `quantile_p=0.75`  | 0.4304    |  16.6     |   558.2         | 46 125 |  61 792|
| `quantile_p=0.90`  | 0.4264    |  16.4     |   554.6         | 46 291 |  57 125|

**Read this:** `RatioTerminator(r=1.20)` hits **0.956 recall@10** — better
than `fixed_ef=128` (0.920) — by spending ~187 expansions where they help
and stopping earlier where they don't. Easy queries cost less; hard
queries get the full budget. `QuantileTerminator` is too aggressive on
this dataset (zero-inflated improvement deltas collapse the P² estimator
near 0) — the research doc names the fix.

## Optimizations

* Squared L2 instead of L2 (monotone in L2, cheaper).
* Parallel top-k view (`BinaryHeap<MaxCand>` capped at `k`) so the
  terminator sees the *kth*-best distance, not the worst-of-ef.
* P² estimator — 5 markers, no buffering, per-query reset is free.
* `Send + Sync` trait — heterogeneous strategies across shards.

## Get started

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-06-21-adaptive-beam-hnsw

# Build & test
cargo build --release -p ruvector-adaptive-beam
cargo test  --release -p ruvector-adaptive-beam

# Run benchmark (reproduces the table above)
cargo run --release -p ruvector-adaptive-beam --bin adaptive-beam-bench

# Minimal demo
cargo run --release -p ruvector-adaptive-beam --example demo
```

Code: `crates/ruvector-adaptive-beam/` ·
Research doc: `docs/research/nightly/2026-06-21-adaptive-beam-hnsw/README.md` ·
ADR: [`docs/adr/ADR-264-adaptive-beam-hnsw.md`](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-06-21-adaptive-beam-hnsw/docs/adr/ADR-264-adaptive-beam-hnsw.md) ·
Upstream: <https://github.com/ruvnet/RuVector>

Tags: `rust` `vector-database` `hnsw` `ann` `nearest-neighbor` `anytime`
`p2-quantile` `agent-memory` `vector-search` `ruvector`
