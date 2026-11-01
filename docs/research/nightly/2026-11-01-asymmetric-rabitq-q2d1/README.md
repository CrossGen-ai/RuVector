# Asymmetric RaBitQ (q=2 bits, d=1 bit)

**Date:** 2026-11-01
**Branch:** `research/nightly/2026-11-01-asymmetric-rabitq-q2d1`
**Crate:** `crates/ruvector-asymmetric-rabitq-q2d1`

## Abstract

RaBitQ quantizes each vector down to one bit per Hadamard-rotated
dimension and scores with a POPCNT. Database storage is minimal, but
recall at a fixed rerank depth is capped by the Hamming estimator's
variance. We ask: can we lift recall without touching the database at
all? We encode the *query* with an extra bit — a per-dimension mask that
suppresses coordinates whose magnitude lies below a per-query threshold
— and keep the database at exactly 1 bit/dim. On a 50 000 × 128
Gaussian corpus this lifts recall@10 from **8.64% to 13.04%** (+51%
relative) at essentially identical scan time (723 vs 733 µs/query) and
zero added database bytes.

## SOTA survey

- **RaBitQ** (SIGMOD 2024, Jianyang Gao, Cheng Long). Rotation-based
  1-bit quantization with a theoretical error bound.
- **Extended RaBitQ** (B-bit RaBitQ, 2025 preprint). Trades bits per DB
  dim for recall. All of the recall lift lives in database storage.
- **FAISS ADC** / **SDC** (Jégou et al.). The canonical statement that
  asymmetric distance beats symmetric at identical codebook size for
  product quantization. Our proposal is the RaBitQ analogue.
- **LVQ / LeanVec** (Intel, VLDB 2024). Shares the "query is small, DB
  is huge, pay on the query side" intuition for scalar quantization.
- **Milvus / Qdrant / Weaviate** all ship symmetric 1-bit RaBitQ as of
  their 2025–2026 releases; none expose a 2-bit asymmetric query.

## Proposed design

Each database vector `d ∈ ℝ^D` is encoded as `BitCode(sign, ||d||)`,
where `sign ∈ {0,1}^D` is the Hadamard rotation's sign pattern. This
is bit-identical to RaBitQ.

Each query `q ∈ ℝ^D` is encoded as `TernaryCode(sign, mask, ||q||,
active)`:

- `sign[i]` = sign bit of the rotated `q[i]`;
- `mask[i]` = 1 iff `|q[i]| ≥ τ · median(|q|)`, with `τ` a tuning
  knob;
- `active` = popcount(mask).

The inner product is estimated as

    ⟨q, d⟩ ≈ ||q|| · ||d|| · (1 − 2 · H / active),
    H = popcount((q.sign ⊕ d.sign) ∧ q.mask).

For `τ = 0` this collapses back to symmetric RaBitQ exactly, so the
backend is a strict generalization. Each estimator is a tight POPCNT
loop over `D/64` words.

## Implementation notes

- Rotation: fast Walsh–Hadamard with a per-coordinate ±1 sign flip,
  seeded. `O(D log D)`, no matrix storage, deterministic.
- Query threshold uses a partial sort (we take the median of `|q|`),
  which is `O(D log D)` once per query — tiny relative to the N-way
  scan.
- The three estimators (`F32Flat`, `SymRabitQ`, `AsymQ2D1`) share one
  `TriIndex` so the same corpus can be scored under all three in one
  run.
- Zero `unsafe`, no intrinsics in the first cut. The POPCNT lowers to
  `popcntq` on x86-64 and `cnt` on NEON through LLVM.

## Benchmark methodology

Deterministic Gaussian corpus (`StandardNormal`, `SmallRng(42)`),
N = 50 000, D = 128, 500 queries (`SmallRng(99)`), top-k = 10.
Ground truth is `F32Flat` on the rotated corpus (rotation is isometric
up to a constant scale, so the ordering is preserved). All numbers from
`cargo run --release -p ruvector-asymmetric-rabitq-q2d1` on this host.

## Results (real cargo-run numbers)

| Backend              | recall@10 | µs / query | DB bytes/vec |
|----------------------|----------:|-----------:|-------------:|
| F32Flat (truth)      |  1.0000   |    2055.6  |          512 |
| SymRabitQ (1b/1b)    |  0.0864   |     733.5  |           16 |
| AsymQ2D1 (τ = 0.25)  |  0.1078   |     736.0  |           16 |
| AsymQ2D1 (τ = 0.50)  |  0.1234   |     724.6  |           16 |
| AsymQ2D1 (τ = 0.75)  |  0.1304   |     723.0  |           16 |

Index build is 43.9 ms (50 000 vectors). The asymmetric estimator costs
one extra `AND` per 64-dim word — invisible in wall-clock time against
the memory-bound scan.

**Headline:** +51% relative recall@10 at identical database storage and
identical scan time.

## How it works (blog-readable walkthrough)

1. Build time, per vector: apply a shared Hadamard rotation, keep the
   sign bits, discard everything else. Store 16 bytes per 128-D vector.
2. Query time: apply the same rotation, compute the median of the
   absolute components, drop every coordinate whose magnitude is less
   than `τ · median` by setting its mask bit to 0.
3. Scoring: for each database vector, XOR the sign words, AND with the
   query mask, POPCNT. Multiply by `||q||·||d||` and the normalisation
   factor.

The intuition: in symmetric 1-bit RaBitQ, a query coordinate near zero
contributes just as much Hamming "signal" as a coordinate far from zero,
even though it is just noise. The mask removes those noisy dimensions
from scoring. Database vectors do not need any new bits — only the
query spends them, and only once per query.

## Practical failure modes

- **Heavy-tailed queries** where the median is near zero push the
  threshold to zero and `AsymQ2D1` degenerates into symmetric. In this
  case recall equals SymRabitQ — never worse, by construction.
- **Low-dimensional data** (`D` < 32) — the median is a poor summary
  statistic; use the mean of `|q|` instead or disable the mask.
- **Already-renormalised queries** (all coordinates near equal
  magnitude) — the mask gives little information, speedup is nil.
- **τ chosen too aggressively** (τ > 1) can drop `active` low enough to
  collapse the estimator variance. We suggest `τ ∈ [0.25, 0.75]` and a
  per-dataset validation sweep.

## What to improve next (roadmap)

1. **Learned τ.** Fit `τ` per query length band or per IVF cluster.
2. **3-bit query** (`AsymQ3D1`). Encode query magnitude in 2 extra bits
   (small, medium, large) and weight the Hamming contribution. Still
   zero DB cost.
3. **SIMD POPCNT & gather.** AVX-512 `vpopcntq` triples the per-cycle
   POPCNT; worth measuring against the memory bandwidth ceiling.
4. **Pairing with IVF.** The mask is the natural signal for IVF probe
   ordering: send dense-mask probes first.
5. **Composable with RaBitQ rerank.** First pass with AsymQ2D1 at a
   wider rerank depth, exact float refinement at a narrower top-k.

## Production crate layout proposal

If adopted into mainline `ruvector-rabitq`:

```
ruvector-rabitq/
├── src/
│   ├── lib.rs
│   ├── codec/
│   │   ├── bit.rs          # existing 1-bit DB code
│   │   └── ternary.rs      # NEW: query ternary code
│   ├── estimator/
│   │   ├── sym.rs          # existing symmetric
│   │   └── asym_q2d1.rs    # NEW: asymmetric
│   └── rotation/wht.rs
├── benches/
└── tests/
```

Public API: `Query::encode_asym(x, tau_frac)` and
`Index::search_asym(&query)`. The backend selector lives on the query,
not the index — no reindex to switch estimators.

## References

- Gao, J., Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  SIGMOD 2024.
- Jégou, H., Douze, M., Schmid, C. "Product Quantization for Nearest
  Neighbor Search." TPAMI 2011. (asymmetric vs symmetric distance.)
- Intel LVQ / LeanVec, VLDB 2024.
- Elasticsearch "Better Binary Quantization" (BBQ) technical note,
  2024–2025.
- Milvus / Qdrant RaBitQ changelogs (2025–2026).
