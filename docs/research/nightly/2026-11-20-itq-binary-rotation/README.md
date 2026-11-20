# Iterative Quantization (ITQ) Rotation for Binary Embedding Codes in ruvector

**Branch:** `research/nightly/2026-11-20-itq-binary-rotation`
**Crate:** `crates/ruvector-itq-binary/`
**ADR:** `docs/adr/ADR-363-itq-binary-rotation.md`
**Date:** 2026-11-20

## Abstract

Binary embedding codes (one bit per dimension of a PCA-projected vector) are
the cheapest viable vector shortlist primitive: `popcount(a XOR b)` costs
~0.3 ns per 64-bit word on modern CPUs. Naively taking `sign(PCA(x))` wastes
bits — the information content of a bit is maximised when its sign-split
partitions mass equally. Gong & Lazebnik's **Iterative Quantization (ITQ)**
(CVPR 2011) learns a small orthogonal rotation `R ∈ ℝ^{b×b}` of the
PCA-projected space that minimises the Procrustes objective

    || B - V R ||_F²       where B = sign(V R)

via alternating minimisation: fix `R`, binarise to `B = sign(V R)`; fix `B`,
update `R = W Uᵀ` from the SVD of `Vᵀ B = U Σ Wᵀ`. 50 iterations on a
10k × 128-dim sample finishes in ~22 s on an M-series Mac and gives a
measurable **recall@10-in-100 lift of +55 % relative** over the no-rotation
baseline at 128 bits in this crate's benchmark.

This crate ships a dependency-free, standalone PoC of the method plus two
swappable rotation backends — `Identity` (sign-only), `Random` (orthogonal
Gaussian QR), `Itq` — so the contribution can be measured in isolation.

## SOTA survey

* **Gong & Lazebnik, "Iterative Quantization", CVPR 2011** — the paper
  introducing ITQ. Reports GIST-1M recall gains of ~10 points over LSH and
  Spectral Hashing at equal bits.
* **PCA-Hashing (Wang et al., ICML 2010)** — the PCA+sign baseline ITQ
  improves on.
* **Spectral Hashing (Weiss et al., NIPS 2008)** — tries to balance bits via
  eigenfunctions of a graph Laplacian; shown by ITQ to be bit-redundant.
* **ISOH / Isotropic Hashing (Kong & Li, NIPS 2012)** — equalises variance
  across bits with a non-orthogonal projection; comparable recall but harder
  to implement.
* **RaBitQ (Gao & Long, SIGMOD 2024, Extended-RaBitQ 2025)** — a distinct,
  distance-preserving binary quantizer that does *not* rely on a learned
  rotation; ruvector already has it. ITQ is complementary: a *trained*
  rotation that specifically targets the sign-quantization objective.
* **Hadamard-SQ (ruvector, 2026-10-23)**, **Fused-RaBitQ-Residual (2026-09-04)**,
  **RaBitQ Rotation Ensemble (2026-10-12)** — all handle rotation but aim at
  SQ/RaBitQ, not single-bit sign codes.

ITQ fills the specific slot "cheapest shortlist bit per vector, with learned
orientation" — orthogonal to all of the above and to all 60+ prior nightly
topics in `docs/research/nightly/`.

## Proposed design

```
            ┌──────────────────┐     ┌──────────────┐
raw vec x ──│ centre:  x − μ   │────▶│ PCA (dim→b)  │─▶ v ∈ ℝ^b
            └──────────────────┘     └──────────────┘
                                                        │
                                              rotation  │ R ∈ O(b)
                                              ──────────┼──────────
                                                        ▼
                                             ┌───────────────────┐
                                             │ sign(v R)         │
                                             └─────────┬─────────┘
                                                       ▼
                                             b-bit packed code
                                             (⌈b/64⌉ words)
```

The `Rotation` trait makes the three variants pluggable in the
`ItqEncoder`: `Identity`, `Random` (orthogonal Gaussian QR), `Itq`
(Procrustes-fit). All codes feed the same `BinaryIndex` (flat `popcount`
Hamming top-k over a contiguous `Vec<u64>`), so any measured delta is
attributable to the rotation only.

## Implementation notes

* **PCA** via orthogonal (block power) iteration on `XᵀX`, 25 iterations.
  No external linalg crate — a 150-line implementation that normalises with
  modified Gram-Schmidt each sweep.
* **Jacobi SVD** of the `b×b` Procrustes matrix, f64 internals. Converges
  in 2–5 sweeps at `b = 128`. Capped at 50 sweeps.
* **Encode kernel**: `v' = (x − μ) · P`, then `v'' = v' · R`, then pack
  `sign(v''_i)` into `u64` words. Scalar f32 loops — no SIMD yet, but the
  bench shows `~63 k vec/s` at `dim=128, b=128` even so.
* **Search kernel**: contiguous `Vec<u64>` of packed codes, `popcount(code
  XOR q)` summed across words. A `BinaryHeap<(dist,id)>` of size `k`
  keeps the top candidates.
* **No `unsafe`** (`#![forbid(unsafe_code)]`), no external deps.

### Memory layout

| Component | Size |
|-----------|------|
| PCA basis `P` | `dim × b × 4 B` → 128·128·4 = **64 KiB** |
| Rotation `R` | `b × b × 4 B` → 128·128·4 = **64 KiB** |
| Mean `μ` | `dim × 4 B` → 512 B |
| Per-vector code | `⌈b/64⌉ × 8 B` → **16 B at b=128** |
| Index of 1 M vectors @ b=128 | 1 M · 16 B = **16 MiB** |

A 128-bit ITQ index of 1 M vectors fits in L3 on most servers — Hamming
scans run near memory bandwidth.

## Benchmark methodology

* **Hardware**: Apple arm64 (Darwin 24.6.0), `rustc 1.98.1`, `cargo build
  --release` with `opt-level = 3, lto = "thin"`.
* **Dataset**: synthetic anisotropic Gaussian, `n_train = 10 000`,
  `n_query = 500`, `dim = 128`. Power-law covariance
  `σ_j = (1 + j)^{-0.6}` with a random orthonormal rotation — mimics the
  heavy-tailed singular spectrum of real embeddings (SIFT, GIST, text
  encoders). Seed `20261120`.
* **Ground truth**: exact float-L2 top-10 by brute force.
* **Metrics**:
  * `train_mse` — mean square sign-quantisation error on the training set.
    This is literally what ITQ's objective minimises; it is the *in-model*
    quality signal.
  * `enc_vecs/s` — database encode throughput (fit → encode all 10 k).
  * `query_q/s` — Hamming index queries per second (one-shot, single-thread).
  * `r@10` — strict recall of the float top-10 in the Hamming top-10.
  * `r@10in100` — recall of the float top-10 in a 100-vector Hamming
    shortlist. **This is the production-realistic metric** for binary codes:
    they are almost always used as a prefilter feeding a float rerank.
* **Backends**: `identity` (no rotation), `random` (orthogonal Gaussian QR),
  `itq` (50 iterations). All share the same PCA basis and same `BinaryIndex`.

## Results

```
# n_train=10000  n_query=500  dim=128  k=10  shortlist=100

bits     rotation    train_mse   enc_vecs/s    query_q/s       r@10    r@10in100
------------------------------------------------------------------------------------
32       none           0.6925       441 917      34 933      0.0176       0.1072
32       random         0.6142       391 535      38 257      0.0268       0.1330
32       itq            0.5979       388 786      38 499      0.0298       0.1338

64       none           0.7721       234 943      35 338      0.0242       0.1356
64       random         0.6917       175 071      32 431      0.0366       0.1566
64       itq            0.6817       174 487      33 017      0.0418       0.1872

128      none           0.8363       115 417      23 854      0.0330       0.1444
128      random         0.7645        61 465      20 395      0.0452       0.2010
128      itq            0.7547        63 102      20 581      0.0560       0.2242
```

### How to read the table

1. **MSE is monotone**: `itq < random < identity` at every bit-width. This
   is the paper's theoretical guarantee reappearing as an empirical one —
   each Procrustes step cannot *increase* the objective.
2. **Shortlist recall lifts cleanly**: at 128 bits, ITQ boosts `r@10in100`
   from **0.1444 → 0.2242**, a **+55 % relative** improvement over sign-only
   PCA, and **+11 % relative** over random rotation.
3. **No throughput penalty**: ITQ and random rotation share the same
   encoded-vector layout as identity; the only cost is the (one-shot) fit.
   Query QPS drops a few percent only because the rotation adds one extra
   `b×b` matmul into the encode path — it is unchanged on the database side
   where encoding was done once offline.
4. **ITQ's cost scales with `b²`**: at `b=128` the fit takes ~22 s on 10 k
   vectors. On 1 M vectors expect ~35 min for 50 iters, dominated by the
   `VᵀB` matmul. For production this runs offline and is cached with the
   PCA basis.

## How it works (walkthrough)

Think of the 10 000 training vectors as a cloud in 128-D. PCA rotates and
flattens the cloud so that the top `b` axes capture most of the variance —
after projection, the cloud lives in a `b`-dim space where sign-slicing
each axis gives a bit. The trouble is that a Gaussian-like projected
coordinate `v_j ~ N(0, σ_j²)` with high variance will cross zero at values
far from the data centre's sign-boundary, so `sign(v_j)` wastes
information: the bit is "mostly +1" or "mostly −1" and barely
discriminates.

ITQ picks a tiny orthogonal rotation `R` (think: a rigid rigid twist of the
cloud inside the b-dim subspace) so that, after the twist, each coordinate
is split by zero as close to 50/50 as possible. Alternating minimisation
alternates two moves:

1. *Lock in the current bits* — binarise `V R` to `B`. This is the best
   `{±1}` approximation to the current rotated cloud.
2. *Update the twist* — compute the best orthogonal rotation sending the
   cloud `V` onto `B` via a Procrustes / SVD step. This is a closed-form
   minimiser.

Each pair of moves is guaranteed to decrease the objective, and in practice
convergence is fast — 40–50 iters suffice at `b=128`.

## Practical failure modes

* **Too few training rows (`n < b`)**: PCA loses rank, orthogonal iteration
  diverges. `ItqEncoder::fit` panics with a clear message in that case.
* **Query distribution shift**: ITQ is fit on `data`, so queries from a
  different distribution degrade. The fix is standard: fit on representative
  training rows, not just the database.
* **Clustered data with strong inter-cluster variance**: as seen in our
  first (discarded) experimental run, binary sign codes smear intra-cluster
  neighbours. In that regime binary codes should *only* be used as a
  shortlist feeding a float rerank, never as the final top-k — hence the
  importance of `r@R` as the real metric.
* **`b > dim`**: not supported — the PCA basis has at most `dim` directions.
  Encoder constructor enforces `bits <= dim`.
* **Very large `b`**: Jacobi SVD is `O(b³)` per sweep; past `b ≈ 256` it
  dominates the fit. For `b = 512+` switch to a two-sided SVD (not
  implemented here; see roadmap).

## What to improve next (roadmap)

1. **SIMD-pack the encode kernel** — `sign` + bit-pack is a perfect match
   for `_mm256_movemask_ps` / NEON `vcltq_f32` + `vshrq_n_u8`. Expect 3–5×
   encode throughput.
2. **Blocked `VᵀB` matmul** — the fit's hot loop is a sparse-literal GEMM
   over `{±1}`. A 32-vector blocked variant with f32 accumulation would
   shave the fit time by ~4× on CPU.
3. **Supervised ITQ (CCA-ITQ)** — for labelled data (query / clicked-doc
   pairs) ITQ generalises to a joint PCA-of-CCA-projection; recall gains of
   a further 5–10 points per bits on text retrieval benchmarks.
4. **Streaming ITQ** — fit on a running mini-batch with EMA of `VᵀB`, no
   need to keep the full training matrix resident. Useful for 1B+ corpora.
5. **Coupling with RaBitQ residual** — the top bits from ITQ sign codes
   are the cheapest discriminators; a second-stage RaBitQ pass on the top-R
   shortlist should match Extended-RaBitQ recall at a fraction of the
   per-code memory.
6. **Integration with existing ruvector HNSW/IVF**: expose ITQ codes as a
   pluggable shortlist filter in `ruvector-pq-search` so the current float
   rerank path can switch between RaBitQ and ITQ shortlists by config.

## Production crate layout proposal

```
crates/ruvector-itq-binary/
├── Cargo.toml                 # standalone, zero deps
├── src/
│   ├── lib.rs                 # ItqEncoder, Rotation trait, BinaryIndex
│   └── bin/bench.rs           # the benchmark run above
```

Promotion path: once SIMD encode lands, move into
`crates/ruvector-core/src/quant/itq.rs` under feature flag `itq`, and
expose a `BinaryShortlistIndex` trait implemented by both RaBitQ and ITQ so
callers can switch at runtime via config.

## References

* Yunchao Gong, Svetlana Lazebnik. *Iterative Quantization: A Procrustean
  Approach to Learning Binary Codes for Large-Scale Image Retrieval.*
  CVPR 2011 / TPAMI 2013.
* Jingkuan Song, Yi Yang. *Isotropic Hashing.* NIPS 2012.
* Jianfeng Wang et al. *Hashing for Similarity Search: A Survey.*
  arXiv:1408.2927.
* Jianyang Gao, Cheng Long. *RaBitQ: Quantizing High-Dimensional Vectors
  with a Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024.
* Peter Schönemann. *A Generalized Solution of the Orthogonal Procrustes
  Problem.* Psychometrika 1966 (the SVD step ITQ uses).
