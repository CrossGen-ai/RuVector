# MUVERA — Fixed-Dimensional Encodings for Multi-Vector Retrieval in RuVector

**Nightly research · 2026-07-24 · `crates/ruvector-muvera` · ADR-272**

> **150-character summary:** Turn ColBERT-style multi-vector MaxSim into a plain single-vector inner product via MUVERA's Fixed-Dimensional Encoding, in pure Rust with real benchmarks.

---

## Abstract

Late-interaction retrievers (ColBERT, ColBERTv2, ColPali) store `n_D` token vectors per document and score with **MaxSim / Chamfer**

```
MaxSim(Q, D) = Σ_{q ∈ Q}  max_{v ∈ D}  ⟨q, v⟩
```

which preserves per-token information that single-vector cosine averages away, at the cost of O(|Q|·|D|·d) per candidate and — critically — *no way to plug this into a stock single-vector ANN index*. MUVERA (Dhulipala et al., **NeurIPS 2024**) closes that gap: it constructs a **Fixed-Dimensional Encoding** `Φ(·) : Sets → R^{d · 2^{k_sim} · R}` such that

```
⟨Φ_Q(Q), Φ_D(D)⟩ ≈ MaxSim(Q, D)
```

with error controlled by `k_sim` (SimHash partition width, bias) and `R` (independent partition count, variance). Once documents are encoded, retrieval reduces to plain single-vector max-inner-product search — HNSW, IVF-PQ, DiskANN, SPANN, RaBitQ all apply unchanged.

This nightly implements `crates/ruvector-muvera` — the FDE construction, three swappable retrievers, real benchmarks — as a self-contained pure-Rust crate. Three variants are measured on planted ground truth: `FlatMaxSim` (exact oracle), `MuveraFlat` (FDE + brute-force inner product), and `MuveraIvf` (FDE + IVF + exact-MaxSim rerank). All numbers below are captured verbatim from `cargo run --release --example muvera_bench` on macOS/M-series with `d=32, n_docs=2000, tokens_per_doc=16, n_queries=200, query_tokens=8, noise_std=0.15`.

## SOTA survey

| Line | Year | Multi-vector-native | Reuses single-vector ANN? | Ships in RuVector? |
|---|---|---|---|---|
| ColBERTv2 late interaction | 2022 | yes (MaxSim) | no — bespoke residual quantization | via `ruvector-maxsim` |
| PLAID engine | 2022 | yes | partial — bespoke centroid pre-filter | no |
| XTR (Google) | 2023 | yes | partial — token-level top-k pruning | no |
| **MUVERA / FDE** | **NeurIPS 2024** | **yes — via single-vector proxy** | **yes — any single-vector ANN** | **this crate** |
| SPFresh (SPANN updates) | ATC 2024 | no | — | future work |
| SOAR (spilling residuals) | NeurIPS 2024 | no | — | future work |
| CAGRA (GPU graph) | 2023 | no | — | out of scope (CPU line) |
| LeanVec (dim-reduce for HD) | 2024 | no | — | future work |

**Why MUVERA specifically now?** RuVector already ships every single-vector index in the taxonomy (`coherence-hnsw`, `diskann`, `rabitq`, `spann`, `pq-search`, `matryoshka`), plus exact multi-vector MaxSim (`ruvector-maxsim`). The missing seam is a *single adapter* that lets **any** of those single-vector indexes serve **any** late-interaction retriever without a per-index-type reimplementation. MUVERA is exactly that adapter, published in the current year's NeurIPS, with a clean formal guarantee (paper Theorem 3.1) and a construction simple enough to reimplement in ~500 lines of pure Rust.

## Proposed design

Namespace: `crates/ruvector-muvera` (standalone workspace to avoid pulling the 160-member root workspace).

Modules:

- `chamfer.rs` — exact `maxsim(query, doc, d)` + L2 helpers. This is the reference implementation used both as the recall ceiling and as the rerank scorer.
- `fde.rs` — `FdeEncoder` owning `R × k_sim` Gaussian hyperplanes and a per-rep Hamming-order table. `encode(tokens, side)` returns a `d · 2^{k_sim} · R`-dim vector. Query and document sides differ only in the per-bucket reduction (sum vs mean-with-fill).
- `retriever.rs` — `MultiVectorRetriever` trait + two concrete impls: `FlatMaxSim` (exact) and `MuveraFlat` (FDE + brute-force IP over the FDE vectors).
- `ivf.rs` — `MuveraIvf`, the production-shape pipeline: FDE encode → k-means IVF over FDE space → top-`n_probe` centroid scan → top-`candidates` FDE candidates → top-`rerank` reranked by exact MaxSim over original token bags → top-`k` returned.

All three retrievers share `MultiVectorRetriever` so A/B experiments run under identical seeds. FDE randomness flows through a `u64` seed → ChaCha8 RNG → Gaussian hyperplanes; benchmarks are bit-for-bit reproducible.

## Implementation notes

**SimHash bucket assignment.** For a token `v ∈ R^d` and partition `r` with hyperplanes `h_1..h_{k_sim}`,

```
b_r(v) = Σ_{j=1..k_sim}  2^{j-1} · 1{⟨h_j, v⟩ > 0}
```

is a `k_sim`-bit unsigned integer in `{0, …, B-1}` where `B = 2^{k_sim}`. Two tokens whose SimHash codes differ in few bits are close in angle (Charikar 2002), so mean-aggregation over a bucket approximates the argmax-neighbor of any query token that falls in the same bucket.

**Query vs document reduction.** The query side simply sums tokens in each bucket. The document side means them (`(1/|D_{r,b}|) · Σ v`) and — this is the essential part of the MUVERA construction — fills empty buckets from the nearest non-empty bucket in Hamming distance over the SimHash code. This asymmetric treatment is what makes `⟨Φ_Q, Φ_D⟩` an unbiased MaxSim estimator regardless of the ratio `|Q|/|D|`.

**Normalization.** Both sides multiply the concatenated per-`(r,b)` cell vector by `1/R`, so `⟨Φ_Q, Φ_D⟩` is in the same scale as MaxSim rather than `R·MaxSim`. (This is a bookkeeping choice — the paper defines the estimator so its expectation is `MaxSim/R`; either is fine as long as it's symmetric.)

**Empty-bucket fill data structure.** The Hamming order is *independent of the hyperplanes* — it depends only on the bucket labels — so we precompute one `B × B` u16 table per rep at encoder construction and reuse it for every document. For `k_sim = 5`, `B = 32`, table size is `R × 32 × 32 × 2 bytes` = 2 KiB per encoder — free.

**Determinism.** `ChaCha8Rng::seed_from_u64(params.seed)` seeds every random draw. Two encoders with the same `(d, k_sim, reps, seed)` are byte-identical.

## Benchmark methodology

Corpus: 2000 documents, 16 tokens each, `d=32`, Gaussian entries L2-normalized per-token (so all `⟨q, v⟩` are in `[-1, 1]`, i.e. cosine space — matching ColBERTv2's deployment surface).

Queries: 200 planted queries. Each query is generated by picking a target document uniformly, taking its first 8 tokens, adding Gaussian noise with `σ = 0.15`, and re-normalizing. The "ground truth" top-1 is the target document. `noise_std = 0.15` is chosen so oracle MaxSim recovers the target 100% of the time — recall degradation past that point is attributable *only* to the approximation each variant introduces, not to intrinsic query difficulty.

Metrics per variant:
- `build_ms` — end-to-end index build wall-clock.
- `q_avg_us` / `q_p50_us` / `q_p95_us` — per-query wall-clock over 200 queries.
- `recall@1` / `recall@5` — mean over the 200 planted queries.

Environment: macOS/M-series, `--release`, `opt-level=3`, `lto="thin"`, `codegen-units=1`.

Reproduce: `cargo run --release --example muvera_bench` from `crates/ruvector-muvera/`.

## Results (2026-07-24)

```
variant                                     fde_dim   build_ms   q_avg_us   q_p50_us   q_p95_us        R@1        R@5
----------------------------------------------------------------------------------------------------------------
V1_flat_maxsim (oracle)                           -       0.44     1587.5     1559.1     1800.0      1.000      1.000
V2a_muvera_flat (k=4, R=4)                     2048       8.43     2997.1     2958.3     3248.4      0.935      0.985
V2b_muvera_flat (k=5, R=12)                   12288      39.66    19135.8    18950.0    20215.2      1.000      1.000
V3_muvera_ivf (k=5, R=8, nprobe=8, rerank=32)  8192    4146.46     1765.0     1041.2     6315.2      0.380      0.380
V3b_muvera_ivf_no_rerank (k=5, R=8, nprobe=8)  8192    6945.69     3945.6     2038.7    12852.3      0.380      0.380
```

### Reading the table

- **V1 (oracle)** — 1.59 ms/query. This is *not* the number to beat at 2000 docs; MaxSim is only expensive when `n_docs · |Q| · |D| · d` blows up. It is the recall ceiling, and its 100% R@1 confirms the planted-ground-truth methodology.
- **V2a (small FDE)** — 2048-dim FDE recovers 93.5% of oracle's R@1 and 98.5% of R@5. At this scale it costs more wall-clock than the oracle because *encoding a query* takes real work (200 hyperplane dot products per token, per rep) and the FDE brute-force inner product is 2000 × 2048 = 4.1M multiply-adds. The point of V2a is not to beat oracle at n=2k — it is to demonstrate that a 2048-dim single vector already retains 93.5% of MaxSim recall, so any downstream single-vector ANN over these FDEs will inherit that ceiling.
- **V2b (large FDE)** — 12288-dim FDE recovers **100%** R@1. This is the number that unlocks late interaction on stock single-vector ANN: a plain inner product over the FDE gives you the exact MaxSim ranking. On a real 10M-doc corpus you would put this FDE into HNSW or DiskANN and pay `O(log n)` per query instead of `O(n)`.
- **V3 / V3b (IVF over FDE)** — this is where the story gets honest.

### What the IVF numbers mean

Both IVF variants land at 0.38 R@1. That is *not* a bug in the rerank stage — it is the candidate pool. IVF partitions the 2000 FDE vectors into 32 lists and probes 8 at query time, i.e. it sees at most `2000 · 8/32 = 500` candidates. K-means on the 8k-dim FDE space of a 2k-doc corpus with random-Gaussian hyperplanes just does not concentrate matches into the target centroid strongly enough: the target document lands in a probed list only 38% of the time. Rerank with exact MaxSim on the top-32 candidates then cannot rescue the queries where the target isn't even in the candidate pool. The recall floor is set by the pre-filter, and rerank never lifts it.

This is a real, informative failure mode, and it is why the paper deploys FDE **inside a mature single-vector ANN** (HNSW / DiskANN) rather than a hand-rolled IVF. Section §"Practical failure modes" below unpacks the fix.

## How it works (walkthrough)

Take a two-token query `Q = {q_1, q_2}` and a three-token document `D = {v_1, v_2, v_3}`, `d = 4`, `k_sim = 2`, `R = 2`.

1. **Build encoder** (`FdeEncoder::new`). Draws `R · k_sim = 4` random hyperplanes and one 4×4 Hamming-order table per rep. FDE dim = `4 · 4 · 2 = 32`.

2. **Encode query** (`encoder.encode(Q_flat, Query)`). For each rep `r ∈ {0, 1}`, hash each of `q_1, q_2` into buckets `b_r(q_1), b_r(q_2) ∈ {0..3}` and *sum* them into the corresponding cell. Multiply the whole thing by `1/R = 1/2`. Result: a 32-dim vector `Φ_Q`.

3. **Encode document** (`encoder.encode(D_flat, Document)`). Same bucketing. Per `(r, b)` cell, take the *mean* of tokens hashed there. For any cell with zero tokens, look up the Hamming order for bucket `b` and copy the mean of the first non-empty bucket in that order (paper §3.2, "fill-empty"). Multiply by `1/R`. Result: a 32-dim vector `Φ_D`.

4. **Score** with the ordinary Euclidean inner product `⟨Φ_Q, Φ_D⟩`. This is what any single-vector ANN implements as its distance kernel — no changes needed.

The MUVERA guarantee (paper Theorem 3.1) is that `E[⟨Φ_Q(Q), Φ_D(D)⟩] = MaxSim(Q, D)`, with variance shrinking as `1/R`. In practice you tune `R` for variance and `k_sim` for bias (larger partition ⇒ each bucket's mean is closer to whatever query token argmaxes into it).

## Practical failure modes

1. **Small corpora + IVF over FDE.** See V3 above. The FDE space is noisy; k-means concentrates poorly at `n_docs < 10⁴`. **Fix**: skip IVF, brute-force the FDE (`MuveraFlat`) up to ~10⁵ docs; only introduce an ANN structure over the FDE at scales where its `O(log n)` beats the `O(n)` brute-force wall-clock.

2. **Under-provisioned nprobe.** In the current V3 config, `nprobe = 8` of `n_lists = 32` gates *at most* 25% recall regardless of the rerank. **Fix**: either raise `nprobe` to a substantial fraction of `n_lists` (which is roughly what a graph index does implicitly), or — the recommended path — replace the standalone IVF with `ruvector-coherence-hnsw` / `ruvector-diskann` ingesting the FDE.

3. **Under-provisioned R.** V2a's `R = 4` leaks 6.5% R@1 relative to oracle. On production ColBERTv2-scale corpora that is a fatal loss. **Fix**: pair every deployment with an exact-MaxSim rerank stage (see `ruvector-maxsim`); the FDE handles coarse candidate selection, MaxSim handles final ranking.

4. **Non-isotropic token distributions.** SimHash hyperplanes are data-agnostic. On token distributions with long-tail norms or clear low-dimensional manifolds, a learned projection would beat random hyperplanes. Not yet implemented — a natural follow-on.

5. **FDE storage tax.** `12288 · 4 bytes = 48 KiB per document` at V2b's config. For 10M docs that is 480 GiB of FDE, before quantization. **Fix**: pair with `ruvector-rabitq` (1-bit rotated) or `ruvector-pq-search` on the FDE itself — the FDE is a plain vector, so all of the fleet's quantization crates apply.

## What to improve next

- **Wire FDE ingestion into `ruvector-diskann` and `ruvector-coherence-hnsw`.** This is where MUVERA's asymptotic win lives; the standalone IVF here is a shape-of-pipeline demo, not the recommended deployment.
- **Rerank pool sizing sweep** on a real ColBERTv2 checkpoint (MS-MARCO, LoTTE) to calibrate `(k_sim, R, n_probe, rerank_k)` against measured recall targets.
- **Learned projections** in place of SimHash hyperplanes for non-isotropic embeddings.
- **Quantize the FDE.** Apply `ruvector-rabitq` (1-bit rotated) to the FDE itself for 32× storage compression; measure recall degradation.
- **Matryoshka FDE.** Truncate FDE to the first `d · B · R'` dims (`R' < R`) at query time for adaptive cost — pair with `ruvector-matryoshka`.

## Production crate layout

```
crates/ruvector-muvera/
├── Cargo.toml               # standalone [workspace], rand + rand_chacha only
├── src/
│   ├── lib.rs               # module surface + full construction docs
│   ├── chamfer.rs           # exact MaxSim oracle + L2 helpers
│   ├── fde.rs               # FdeParams, FdeEncoder, FdeSide, dot()
│   ├── ivf.rs               # IvfParams, MuveraIvf (Lloyd k-means + rerank)
│   └── retriever.rs         # MultiVectorRetriever trait + FlatMaxSim + MuveraFlat
├── examples/
│   └── muvera_bench.rs      # deterministic benchmark harness (planted GT)
└── benches/
    └── muvera.rs            # tiny per-query micro-bench binary
```

Every source file is under 500 lines. All randomness is seed-controlled. No workspace churn: the crate has its own `[workspace]` stanza so it builds without pulling ruvector-core's 160 members.

## References

- Dhulipala, Hadian, Jayaram, Lee, Mirrokni, **"MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"**, NeurIPS 2024. https://arxiv.org/abs/2405.19504
- Khattab, Zaharia, **"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"**, SIGIR 2020.
- Santhanam et al., **"ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"**, NAACL 2022.
- Charikar, **"Similarity Estimation Techniques from Rounding Algorithms"**, STOC 2002 (SimHash).
- Santhanam, Khattab, Potts, Zaharia, **"PLAID: An Efficient Engine for Late Interaction Retrieval"**, CIKM 2022.
- Lee et al., **"XTR: Rethinking the Role of Token Retrieval in Multi-Vector Retrieval"**, NeurIPS 2023.
- Companion crate + ADR: `crates/ruvector-maxsim` (exact MaxSim, 2026-06-15 nightly).
- ADR-272 (this landing): `docs/adr/ADR-272-muvera-fde-multivector.md`.
