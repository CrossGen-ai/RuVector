# ADR-328: Hamming-Cascade ANN Retrieval Path

- **Status**: Proposed
- **Date**: 2026-08-21
- **Deciders**: RuVector Nightly Research (autonomous)
- **Related**: `ruvector-rabitq` (1-bit codes with rotation), `ruvector-pq-search` (product quantization), `ruvector-hnsw-repair` (graph indexes), `ruvector-entropy-ann` (adaptive termination)
- **Tags**: quantization, cascade, ann, hamming, bandwidth, rerank

## Context

RuVector already ships several quantization crates (`ruvector-rabitq`,
`ruvector-pq-search`, `ruvector-turboquant`) and several ANN index crates
(`ruvector-coherence-hnsw`, `ruvector-spann`, `ruvector-lsm-ann`,
`ruvector-diskann`, `ruvector-entropy-ann`). What is missing is a **stable,
trait-based composition surface** that lets callers mix a coarse, lossy
distance oracle with a fine, exact one *without* pulling in a specific
index topology.

Prior art (Faiss binary indexes, Milvus `BIN_FLAT + FLOAT_RERANK`, Qdrant
Binary Quantization + oversampling, LanceDB IVF-PQ with FP32 rerank) all
implement a cascade internally, but each hardcodes its coarse quantizer to
its index. In practice most modern QPS/recall wins come from *shrinking
the bytes touched during the flat-ish coarse scan* and paying the FP32
price only on a small shortlist.

## Decision

Introduce `ruvector-hamming-cascade`: a small, safe-Rust crate that:

1. Defines a `DistanceOracle` trait (`prime`, `score`, `footprint_bytes`,
   `name`) — the only extension point.
2. Ships three concrete oracles (`Fp32Oracle`, `Int8Oracle`,
   `HammingOracle`) that all share the same identity space (`i in 0..N`).
3. Ships a `Cascade<C, F>` composition that runs the coarse oracle over
   the full population, keeps `probe_k` best via `select_nth_unstable`,
   then re-scores that shortlist with the fine oracle.
4. Ships a real report binary (`cascade-report`) that produces measured
   numbers on a reproducible synthetic workload.

The crate is deliberately **not** an index — it is the composable
`{coarse, fine}` primitive that existing indexes (HNSW, IVF, SPANN,
LSM-ANN) can plug in as their scan/rerank layer once the trait proves
stable.

### Acceptance (measured, `PROBE_K=1000 cargo run --release -p ruvector-hamming-cascade --bin cascade-report`)

| Config | Recall@10 | µs/query | Footprint | Notes |
|---|---|---|---|---|
| fp32 flat (baseline) | 1.000 | 336 | 9.77 MiB | reference |
| int8 → fp32 rerank | 1.000 | 538 | 6.18 MiB | **loses** — dequant-per-element is not amortised |
| hamming → fp32 rerank | 0.922 | 78 | 5.04 MiB | **4.3x faster @ 92% recall** |
| hamming-only (no rerank) | 0.169 | 27 | 314 KiB | shows what the rerank is buying |

The Hamming-cascade acceptance target (recall ≥ 0.90 at <30% of FP32
latency) is met at `probe_k=1000` on N=10k, dim=128. Sweeping
`probe_k → 2000` lifts recall to 0.979 at 124 µs (still 2.8x faster
than FP32).

## Consequences

**Positive**
- New composable surface (`DistanceOracle`) that other RuVector indexes can
  adopt without rewriting their storage layer.
- Honest, reproducible numbers (no mocked scans, no aspirational recalls).
- Zero unsafe code, zero external runtime dependencies.

**Negative / open**
- `Int8Oracle` is currently a loss vs FP32 because scoring dequantises per
  element rather than accumulating in fixed-point. Left in the report on
  purpose — it is a **real failure mode** callers should see.
- The Hamming threshold is a per-dimension mean; a learned rotation
  (à la RaBitQ / ITQ) would tighten recall at fixed `probe_k`. That is
  the natural follow-up.
- No SIMD. Portable scalar Rust already delivers >4x cascade speedup;
  a SIMD Hamming intrinsic would widen the gap but is deferred.

## Alternatives Considered

1. **Extend `ruvector-rabitq`** with a cascade helper. Rejected because
   RaBitQ's scoring couples to its rotation matrix; forcing every caller
   through that gate is heavier than the composition-only story here.
2. **Bake the cascade into `ruvector-coherence-hnsw`.** Rejected — the
   cascade primitive is orthogonal to graph topology and belongs in a
   crate that indexes can *use*, not one they *are*.
3. **INT8 SAD (integer accumulator)** for the coarse pass. Real win, but
   would not compose with the `f32`-typed `score` on the trait. Marked
   as a next-iteration extension of `Int8Oracle` (add a
   `score_i32(&self, i) -> i32` variant).

## References

- Faiss binary indexes docs — https://github.com/facebookresearch/faiss/wiki/Binary-indexes
- Milvus binary vector + rerank — https://milvus.io/docs/binary_vector.md
- Qdrant Binary Quantization — https://qdrant.tech/documentation/guides/quantization/#binary-quantization
- LanceDB IVF-PQ rerank — https://lancedb.github.io/lancedb/concepts/index_ivfpq/
- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search",
  SIGMOD 2024.
