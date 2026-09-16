# ADR-347 — Additive Quantization with Beam-Search Encoding

- **Status**: Proposed (nightly research 2026-09-16)
- **Date**: 2026-09-16
- **Related**: ADR-329 (RaBitQ), ADR-346 (bidirectional-densify), `crates/ruvector-rabitq`, `crates/ruvector-fused-rabitq-residual`, `crates/ruvector-cascade-adc`

## Context

RuVector currently ships several compressed-code ANN backends:

- Product Quantization (`ruvector-pq-search`) — the standard subspace quantizer, per-subspace k-means with 256 centroids.
- RaBitQ and RaBitQ-fused-residual — 1-bit rotation-based quantization with theoretical error bounds.
- Cascade ADC and Anisotropic PQ — refinements to the ADC scan.
- LVQ, ternary, Matryoshka coarse/fine — alternative bitrate/accuracy trade-offs.

**PQ is fundamentally limited on data whose intrinsic dimension does not align with axis-parallel subspaces.** Its subspaces are disjoint slices `R^{d/M}`, and correlations between coordinates in different subspaces are not representable at all. Empirically PQ leaves substantial recall on the table for low-rank or highly-correlated embeddings (LLM last-hidden-state vectors, CLIP image embeddings, RAG document embeddings after L2-normalization).

**Additive Quantization** (Babenko & Lempitsky, CVPR 2014; refined by Martinez et al. LSQ, 2016; Ge et al. OPQ, 2013) approximates a vector as the *sum* of `M` centroids drawn one from each of `M` full-dimensional codebooks:

```
x ≈ Σ_{m=1..M} C_m[i_m],   i_m ∈ [0, K)
```

At the same bit budget as PQ (`M · log2(K)` bits per code), AQ can represent correlations across all coordinates and typically achieves 2–4× lower reconstruction MSE. The catch: **encoding is NP-hard**. Greedy residual encoding is fast but leaves quality on the table; the standard remedy is beam search over the codebooks.

The project has no AQ crate. This ADR proposes adding one as a research-quality PoC with a real benchmark against the PQ baseline.

## Decision

Add crate `crates/ruvector-aq-beam` implementing:

1. **AQ codebook training** via successive residual k-means (`M` passes of Lloyd's, each fitting on the residual of the prior pass). Deterministic under a seed.
2. **Beam-search encoding** with configurable beam width `B`. `B=1` reproduces the greedy residual encoder; `B ∈ {8, 32}` are the practically useful settings.
3. **ADC-style L2 search** using precomputed per-query linear tables (`q · c_m[k]`, M·K values) and static self-norm + cross-codebook inner-product tables (M(M−1)/2 · K² values) built at train time.
4. A **PQ baseline** in the same crate (identical bit-rate, identical k-means init) so comparisons are apples-to-apples.
5. A **`Quantizer` trait** so subsequent crates (`ruvector-lsq`, `ruvector-opq-aq`) can slot in without touching the benchmark harness.

Codes are `M` bytes (u8) per vector; K is fixed at 256 for this PoC (a byte per code fits the ADC-scan cache-line story used elsewhere in the project).

## Consequences

**Positive**

- Bridges an obvious quality gap vs PQ on correlated embeddings.
- Sets up a shared `Quantizer` trait that future quantizer research can implement without duplicating the benchmark harness.
- Complements the RaBitQ and Cascade-ADC efforts: AQ is the *codebook-side* improvement, RaBitQ / Cascade-ADC are the *distance-side* improvements — they compose.

**Negative**

- Encoding is 30–1100× slower than PQ depending on beam width. For online insertion into HNSW-over-codes indexes this is significant and requires either batching or a background reencode pipeline.
- Cross-codebook tables cost `M(M−1)/2 · K² · 4 bytes` — for `M=4, K=256` that is ~1.5 MB per index. Acceptable; scales quadratically in K and quadratically in M.
- Search-time cost per code is `O(M² )` for the cross terms vs `O(M)` for PQ. On `M=4` this is 6 extra table lookups per candidate — measured overhead below is 60–70%.

**Measured (Apple M4 Max, 128 GB RAM, rustc 1.89.0, release, single-threaded, `n=8000` d=128 rank-24 + noise 0.15, 200 queries, k=10)**

| Backend          | Code bytes | Encode 8k vecs | Reconstruction MSE | Search 200×8k | recall@10 |
| ---------------- | ---------: | -------------: | -----------------: | ------------: | --------: |
| PQ M=4 K=256     |          4 |         43.6ms |           11.28156 |        24.8ms |     0.098 |
| AQ M=4 beam=1    |          4 |       1473.6ms |            3.44474 |        42.0ms |     0.343 |
| AQ M=4 beam=8    |          4 |       9927.4ms |            3.35406 |        40.7ms |     0.360 |
| AQ M=4 beam=32   |          4 |      48140.1ms |            3.33212 |        40.8ms |     0.357 |

Highlights: AQ cuts MSE 3.4× and lifts recall@10 from 0.098 → 0.360 at identical 32-bit code footprint. Beam=8 is the sweet spot; beam=32 is 5× slower for a fraction of a percent.

## Alternatives considered

- **OPQ (Optimized PQ)** — learn a rotation before PQ. Cheaper than AQ but strictly weaker: it still uses disjoint subspaces. Consider as a stackable preprocessing step on top of PQ; does not obsolete this ADR.
- **LSQ (Local Search Quantization, Martinez et al. 2016)** — supersedes beam-search AQ via ICM (Iterated Conditional Modes) refinement of assignments plus periodic codebook re-fits. **Follow-up path**: implement as `ruvector-lsq` reusing the codebook/self-norm/cross-table code from this crate.
- **Composite Quantization (Zhang et al. 2014)** — enforces near-zero cross-codebook inner products so the O(M²) search term vanishes. Follow-up ADR candidate; requires a constrained k-means solver.
- **RaBitQ / Ternary** — already in-tree; orthogonal (distance-side compression rather than codebook-side).
- **Do nothing** — leaves the correlated-embedding regime under-served.

## Follow-up

- Implement **LSQ** refinement (ICM + codebook re-fit).
- Wire AQ codes into the `ruvector-pq-search` scan for a drop-in replacement path.
- Parallelize encoding with rayon (embarrassingly parallel across dataset rows).
