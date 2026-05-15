# ADR-195: MUVERA Fixed Dimensional Encodings for multi-vector retrieval

- **Status:** Proposed (PoC landed in `crates/ruvector-muvera`)
- **Date:** 2026-05-15
- **Related:** ADR-189 (sparse attention KV cache), ADR-194 (anisotropic VQ),
  ADR on RaBitQ (1-bit quantization), ADR on Acorn (filtered HNSW)

## Context

Late-interaction retrievers (ColBERT, ColBERT-v2, ColPali, JinaColBERT) score
each query–document pair by

```
Chamfer(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} <q, d>
```

with token-level vectors. They are the strongest dense retrievers on BEIR
and ViDoRe but are operationally expensive: scoring is `O(|Q|·|D|·d)` per
pair, indices are large (one vector per token), and existing single-vector
ANN infra (HNSW, IVF, RaBitQ, AnisotropicVQ — all already in this workspace)
cannot be reused as-is.

Dhulipala et al. (NeurIPS 2024, *MUVERA*) show that a randomized,
data-oblivious encoding can transform any multi-vector representation into
a *single* fixed-dimensional vector whose inner product is an unbiased
estimator of asymmetric Chamfer similarity. This makes ColBERT-style
retrieval drop directly onto our existing single-vector indices.

We have no MUVERA implementation today. RuVector ships ~150 crates but
none target multi-vector retrieval; the "ColBERT problem" was previously
out of scope. With ColPali / ColQwen visual document retrievers driving
demand for multi-vector ANN, we need this primitive.

## Decision

Land `crates/ruvector-muvera` as the canonical FDE encoder for
ruvector. Crate exposes a single `FdeEncoder` keyed on a `FdeConfig`
(SimHash bits `k_sim`, repetitions `R`, fill strategy, optional Gaussian
projection). Encoded vectors are plain `Vec<f32>` so they plug into any
existing index without further glue.

Asymmetric encoding follows the paper:

- **Document:** per-bucket centroid (mean of doc tokens in that SimHash
  bucket); empty buckets are filled by Hamming-nearest non-empty bucket.
- **Query:** per-bucket sum of query tokens; no fill, no normalisation.

Reps are averaged (instead of concatenated) to keep magnitudes
independent of `R` while preserving ranking — strictly equivalent for
top-k retrieval.

The crate is intentionally *backend-agnostic*: it does not own an index.
Downstream crates wire FDE output into `ruvector-rabitq`,
`ruvector-anisotropic-vq`, or HNSW.

## Consequences

### Positive

- Reuses ~150 crates of existing single-vector infra for multi-vector
  retrieval. No new index needed.
- Trivially parallelisable (Rayon-friendly), no learned components, no
  training data required.
- Pluggable into the workspace's compression stack: FDE → RaBitQ gives a
  ~256× total compression vs raw multi-vector storage on a typical
  ColBERT-128 corpus.

### Negative / open

- **FDE dim can balloon** without projection: `R · 2^k_sim · d`. With
  `k_sim=5, R=20, d=64` we get `40 960` floats ≈ 160 KB per doc, *larger*
  than the raw multi-vector. Production deployments must combine FDE
  with random projection or PQ.
- **Projection cost is quadratic in raw dim.** Our PoC's Gaussian
  projection from 40 960 → 1 024 is ~24 ms per doc on Apple M4 Max.
  Index-build time is the bottleneck, not query time. Mitigation:
  count-sketch / FastFood projection (future ADR).
- **Variance vs Chamfer** is bounded by `O(1/√R)`; small `R` (≤ 4) is
  fast but unreliable on hard rankings. The crate ships sane defaults
  (`k_sim=5, R=20`) consistent with the paper.
- We need a *reranker* in production: FDE retrieves a candidate set,
  exact Chamfer reranks the top ~100. Already feasible with existing
  infra.

## Alternatives considered

1. **PLAID / EMVB-style native multi-vector indices.** Powerful but
   requires bespoke index structures. MUVERA wins on operational
   simplicity: it is a pure encoding step.
2. **DESSERT (Engels et al., 2023).** Earlier randomized estimator for
   Chamfer; has higher per-rep variance. MUVERA strictly dominates with
   the centroid + fill scheme.
3. **Train a single-vector reducer (e.g. ColBERT pooling).** Loses the
   theoretical guarantee and requires per-corpus training. MUVERA is
   data-oblivious.
4. **Do nothing; rely on cross-encoder reranking.** Doesn't solve the
   ANN problem — you still need an initial retriever.

## Validation

- `cargo test -p ruvector-muvera`: 4/4 pass (recall@20 ≥ 0.7 vs exact
  Chamfer on a planted-signal corpus).
- `cargo bench -p ruvector-muvera`: see research doc for numbers.
- `cargo run -p ruvector-muvera --release --bin muvera-demo`: prints a
  full 3-variant comparison.
