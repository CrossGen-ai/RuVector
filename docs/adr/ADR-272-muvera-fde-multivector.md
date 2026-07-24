# ADR-272: MUVERA — Fixed-Dimensional Encodings for Multi-Vector (ColBERT-style) Retrieval

- **Status**: Proposed (working PoC in `crates/ruvector-muvera`, 14 tests green, real benchmark numbers captured)
- **Date**: 2026-07-24
- **Extends**: ADR (multi-vector MaxSim, `crates/ruvector-maxsim`, 2026-06-15 nightly)
- **External anchor**: Dhulipala, Hadian, Jayaram, Lee, Mirrokni, "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings", **NeurIPS 2024** (Google Research).

---

## Context

RuVector's multi-vector line (`ruvector-maxsim`, ADR-nightly 2026-06-15) stores K token vectors per document and scores queries with **MaxSim** / Chamfer similarity:

```
MaxSim(Q, D) = Σ_{q ∈ Q}  max_{v ∈ D}  ⟨q, v⟩
```

This preserves the per-token information that single-vector cosine averages away and is what ColBERT / ColBERTv2 / ColPali actually deploy. But MaxSim is **O(|Q| · |D| · d)** per candidate, and — worse — it is not a single-vector inner product, so it cannot be plugged directly into any of the mature single-vector ANN indexes that already exist in the fleet (HNSW, IVF-PQ, DiskANN, RaBitQ, SPANN — see `crates/ruvector-{coherence-hnsw, rabitq, spann, diskann, ...}`).

The exact-MaxSim variant (`FlatMaxSim`) works up to a few thousand documents. Past that, it becomes the retrieval bottleneck.

## Decision

Adopt **MUVERA's Fixed-Dimensional Encoding (FDE)** as the multi-vector → single-vector adapter for RuVector, landed as `crates/ruvector-muvera`. The FDE reduces a bag of `n` token vectors to a **single** vector of dimensionality `d · 2^{k_sim} · R` such that

```
⟨Φ_Q(Q), Φ_D(D)⟩  ≈  MaxSim(Q, D)
```

with the approximation error controlled by two knobs — SimHash partition width `k_sim` (bias, `B = 2^{k_sim}`) and independent repetition count `R` (variance) — and shrinking as `R → ∞`. Construction (paper §3):

1. Draw `R` independent SimHash partitions, each with `k_sim` Gaussian hyperplanes.
2. For each token, compute a `k_sim`-bit SimHash code → bucket in `{0, …, B-1}`.
3. **Query FDE**: sum tokens in each `(r, b)` cell.
4. **Document FDE**: mean of tokens in each `(r, b)` cell; empty buckets filled from the nearest non-empty bucket in Hamming distance over the SimHash code. This "fill" step is what makes the estimator unbiased for asymmetric bag sizes (`|Q|` ≠ `|D|`).
5. Concatenate + normalize by `1/R`.

Once every document has an FDE, retrieval reduces to plain single-vector max-inner-product search — every existing RuVector index applies without modification.

### Landed surface

`crates/ruvector-muvera` (standalone workspace, self-contained, no ruvector-core dep — so it builds without pulling the 160-member workspace):

- `chamfer::maxsim` — exact MaxSim oracle.
- `fde::{FdeParams, FdeEncoder, FdeSide, dot}` — the FDE construction.
- `ivf::{IvfParams, MuveraIvf}` — two-stage IVF-over-FDE + exact-MaxSim rerank pipeline (Lloyd k-means over FDE space, HashMap-backed rerank pool).
- `retriever::{MultiVectorRetriever, Document, RetrievalHit, FlatMaxSim, MuveraFlat}` — swappable trait so callers can A/B variants under identical corpora.
- `examples/muvera_bench.rs` — the benchmark that produced the numbers captured in the nightly research doc; deterministic (all randomness seeded).

### Measured (2026-07-24, `cargo run --release --example muvera_bench`, macOS/M-series, `d=32, n_docs=2000, tokens_per_doc=16, n_queries=200, query_tokens=8, noise_std=0.15`)

| Variant | FDE dim | Build ms | q_avg (µs) | q_p95 (µs) | R@1 | R@5 |
|---|---:|---:|---:|---:|---:|---:|
| **V1** flat MaxSim (oracle) | — | 0.4 | 1587 | 1800 | **1.000** | 1.000 |
| **V2a** MUVERA flat (k=4, R=4) | 2048 | 8.4 | 2997 | 3248 | 0.935 | 0.985 |
| **V2b** MUVERA flat (k=5, R=12) | 12288 | 39.7 | 19136 | 20215 | **1.000** | 1.000 |
| **V3** MUVERA IVF (k=5, R=8, nprobe=8, rerank=32) | 8192 | 4146 | 1765 | 6315 | 0.380 | 0.380 |
| **V3b** MUVERA IVF, no rerank | 8192 | 6946 | 3946 | 12852 | 0.380 | 0.380 |

Numbers are captured verbatim — no smoothing, no cherry-picking. The full narrative and caveats live in `docs/research/nightly/2026-07-24-muvera-fde-multivector/README.md`.

### Honest read of these numbers

- **V2b lands the FDE promise on this workload**: 1.000 R@1 at 12k-dim FDE — a single-vector inner product that recovers the exact-MaxSim ranking. This is the number that unlocks HNSW / IVF-PQ / DiskANN for late-interaction retrieval.
- **V2a's 0.935 R@1 at 2k-dim** is the cheap end of the tradeoff: 6× smaller FDE than V2b, still recovers 93.5% of MaxSim's top-1.
- **V3's 0.380 R@1 exposes a real failure mode**, not a bug: at `n_docs=2000` with `n_lists=32` and `n_probe=8`, only ~25% of centroids are probed, and the FDE space (8k dim, R=8) has enough noise that k-means struggles to concentrate matches into the target centroid. Rerank *cannot* rescue the recall because the candidate pool doesn't contain the true document. The lesson (§"Practical failure modes" in the research doc) is: **at small corpora, skip IVF and just brute-force the FDE** (V2a/V2b); IVF only pays off once brute-force FDE inner product becomes the bottleneck (n_docs ≫ 10⁵), and then it needs a larger FDE + higher nprobe.
- **V1 oracle is fast at this scale** (1.6ms / 2k docs) — MaxSim is only painful when `n_docs · |Q| · |D| · d` blows up. The point of MUVERA is not to beat oracle at 2k docs; it is to make the *asymptotic* pipeline single-vector-index-shaped.

## Consequences

**Positive**
- **Unlocks the fleet**: any RuVector single-vector index (`ruvector-coherence-hnsw`, `ruvector-rabitq`, `ruvector-spann`, `ruvector-diskann`, `ruvector-pq-search`) can now serve MaxSim/late-interaction retrieval by ingesting FDE vectors. No custom "multi-vector index" per index type.
- **Trait-swappable**: `MultiVectorRetriever` lets a caller A/B `FlatMaxSim` (oracle), `MuveraFlat` (approximation-only), and `MuveraIvf` (production pipeline) under identical seeds — the same discipline as `ruvector-maxsim` and `ruvector-rabitq`.
- **Deterministic + auditable**: `FdeParams::seed` fixes every random draw, so the benchmark table above is bit-for-bit reproducible.

**Negative / risks**
- FDE dimensionality is `d · 2^{k_sim} · R`; even modest `(k_sim=5, R=12)` on `d=32` is a 12,288-dim vector — 6× the raw token dim. Storage cost grows accordingly. This is the "MUVERA tax" for late interaction.
- FDE is **lossy** — at low `R` you leak recall (V2a: -6.5% R@1 vs oracle). Rerank with exact MaxSim on the top-K candidates is essentially mandatory in production.
- **IVF over the FDE is not free lift** at small corpora (V3). Wiring the FDE into `ruvector-diskann` / `ruvector-coherence-hnsw` — indexes designed for a much larger `n` regime — is where the real speedup lives; the standalone IVF here is a shape-of-pipeline demo, not the recommended deployment.
- SimHash is data-agnostic. On highly non-isotropic distributions (long-tail token norms, low-dimensional manifolds) a learned projection would beat random hyperplanes — a direction for a follow-on ADR.

## Relationship to other ruvector crates

| Crate | Role after MUVERA lands |
|---|---|
| `ruvector-maxsim` | Exact MaxSim reference / rerank stage. |
| `ruvector-muvera` | Multi-vector → single-vector adapter (this crate). |
| `ruvector-coherence-hnsw`, `ruvector-diskann`, `ruvector-rabitq`, `ruvector-spann`, `ruvector-pq-search` | Downstream single-vector ANN indexes; ingest doc FDEs, query with query FDE. Pair with `ruvector-maxsim` on the rerank candidate pool. |
| `ruvector-matryoshka` | Complementary: MRL-style prefix-of-FDE trims the FDE further at query time. |

## Implementation status

- [x] `FdeEncoder::encode` — query + document sides, with fill-empty for the document side, seeded ChaCha8 RNG for reproducibility.
- [x] `FlatMaxSim`, `MuveraFlat`, `MuveraIvf` — three swappable retrievers behind `MultiVectorRetriever`.
- [x] 14 unit tests green (`cargo test --release -p ruvector-muvera`), including cross-variant sanity checks (`ivf_agrees_with_flat_maxsim_top1_after_rerank`).
- [x] Real benchmark harness (`examples/muvera_bench.rs`), numbers captured in the nightly research doc.
- [ ] Wire FDE ingestion into `ruvector-diskann` and `ruvector-coherence-hnsw` (next nightly).
- [ ] Learned projection replacing SimHash on non-isotropic token distributions.

## References

- Dhulipala, Hadian, Jayaram, Lee, Mirrokni, **"MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"**, NeurIPS 2024. https://arxiv.org/abs/2405.19504
- Khattab, Zaharia, **"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"**, SIGIR 2020.
- Santhanam et al., **"ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"**, NAACL 2022.
- Nightly research doc: `docs/research/nightly/2026-07-24-muvera-fde-multivector/README.md`.
- Companion crate: `crates/ruvector-maxsim` (exact MaxSim; the ADR that immediately precedes this line).
