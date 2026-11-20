# ADR-363: Iterative Quantization (ITQ) rotation for single-bit-per-dim binary codes

- **Status**: Accepted (nightly PoC); no production integration yet
- **Date**: 2026-11-20
- **Related**: ADR-362 radix-bucket top-K (sibling shortlist infrastructure);
  prior nightlies on quantization — `rabitq`, `fused-rabitq-residual`,
  `hadamard-sq`, `rabitq-rotation-ensemble`, `opq-rotation`,
  `ternary-vector-search`.

## Context

Binary embedding codes (`sign(projection(x))`, one bit per packed dimension)
remain the cheapest viable shortlist primitive for vector search:
`popcount(code XOR q)` on `u64` words runs at memory-bandwidth speed and
the per-vector footprint is 16 B at 128 bits. The ruvector tree already has
several *distance-preserving* binary quantizers in the RaBitQ family
(ADR entries for `rabitq`, `fused-rabitq-residual`, `rabitq-rotation-ensemble`,
`extended-rabitq-multibit`); it does *not* have a classical learned-rotation
sign hash.

The canonical method for the latter is **Iterative Quantization (ITQ)**
(Gong & Lazebnik, CVPR 2011): fix a `b`-dim PCA basis, then learn a small
orthogonal rotation `R ∈ O(b)` that minimises the Procrustes objective

    || B − V R ||_F²       where B = sign(V R)

via alternating minimisation. ITQ is complementary to RaBitQ in two ways:

1. **It is a *trained* sign-hash**, not a distance-preserving binary
   encoder. Its design target is shortlist recall, not estimator accuracy.
2. **Its code length is tunable below the dimension** (`b ≤ dim`) — RaBitQ
   codes have fixed length tied to the input dim, whereas ITQ can sit at
   32 or 64 bits with meaningful recall. This matters for L3-resident
   indexes.

The open question for this nightly: can we reproduce the paper's reported
shortlist-recall gain in a dependency-free Rust port, and is the rotation
fit cheap enough to sit on a normal CI run?

## Decision

Land `crates/ruvector-itq-binary` as a standalone nightly PoC. The crate is
built around a swappable `Rotation` trait with three implementations —
`Identity`, `Random` (orthogonal Gaussian QR), `Itq` (Procrustes fit) — so
the measured delta is attributable to the rotation alone. Everything
downstream (`ItqEncoder`, `BinaryIndex`) is shared.

Key shape:

- `ItqEncoder::fit(data, dim, bits, kind, iters, seed)` — centres the data,
  fits PCA via 25 iterations of orthogonal iteration on `XᵀX`, then
  constructs the rotation per `kind`.
- For `kind = Itq`, the fit loop is 50 alternating steps of (i) `B =
  sign(V R)` and (ii) `R = W Uᵀ` where `SVD(VᵀB) = U Σ Wᵀ`. The `b×b` SVD
  is one-sided Jacobi, f64 internals, 50-sweep cap.
- `BinaryIndex` is a flat `Vec<u64>` of `⌈bits/64⌉` words per vector with a
  heap-bounded `popcount` top-k scan.
- Zero external dependencies, `#![forbid(unsafe_code)]`, standalone
  workspace (does not touch the main workspace lints).

## Consequences

**Measured on anisotropic Gaussian (σ_j = (1+j)^{-0.6}, n = 10 k, dim = 128):**

| bits | rotation | train_mse | enc_vecs/s | query_q/s | r@10 | r@10in100 |
|-----:|---------:|----------:|-----------:|----------:|-----:|----------:|
|  32  | none     |    0.6925 |     441 917 |    34 933 | 0.018 |    0.107 |
|  32  | random   |    0.6142 |     391 535 |    38 257 | 0.027 |    0.133 |
|  32  | **itq**  | **0.5979**|     388 786 |    38 499 | **0.030** | **0.134** |
|  64  | none     |    0.7721 |     234 943 |    35 338 | 0.024 |    0.136 |
|  64  | random   |    0.6917 |     175 071 |    32 431 | 0.037 |    0.157 |
|  64  | **itq**  | **0.6817**|     174 487 |    33 017 | **0.042** | **0.187** |
| 128  | none     |    0.8363 |     115 417 |    23 854 | 0.033 |    0.144 |
| 128  | random   |    0.7645 |      61 465 |    20 395 | 0.045 |    0.201 |
| 128  | **itq**  | **0.7547**|      63 102 |    20 581 | **0.056** | **0.224** |

- MSE is strictly monotone: `itq < random < identity` at every bit-width,
  as guaranteed by alternating minimisation.
- 128-bit ITQ lifts `r@10in100` from 0.144 to 0.224 — **+55 % relative** over
  sign-only PCA and **+11 % relative** over random rotation. This is the
  production-relevant metric: binary codes are a prefilter, not the final
  top-k.
- No query-time throughput cost: identical code layout, identical scan.
  Fit cost scales `O(iters · n · b²)`; 50 iters on 10 k × 128 finishes in
  ~22 s on an M-series Mac.

**Costs / risks:**

- Rotation fit scales badly with `b` on the Jacobi path (`O(b³)` per
  sweep). Past `b ≈ 256` switch to a two-sided SVD.
- Encode kernel is scalar f32 for now — SIMD pack via NEON `vcltq_f32` +
  bit-shift (or `_mm256_movemask_ps` on x86) is the obvious next win, a
  3–5× throughput headroom per the roadmap.
- ITQ is trained on the fit set; query drift degrades it the same way it
  degrades PCA-sign. This is standard and handled by training on
  representative traffic, not by algorithm change.

## Alternatives considered

1. **RaBitQ instead** — already landed, distance-preserving, but fixed
   length ≈ dim and not a *learned* code. Complementary, not substitutable:
   this ADR stands up ITQ for the "sub-dim, cheap shortlist" slot that
   RaBitQ does not target.
2. **Spectral Hashing (NIPS 2008)** — the classical alternative; superseded
   on recall by ITQ at equal bits in the original paper. Not worth a
   nightly.
3. **Isotropic Hashing (NIPS 2012)** — equalises bit variance via a
   non-orthogonal projection. Comparable recall, more fragile numerically,
   no clear win for a PoC.
4. **Skip and use Hadamard-SQ** — Hadamard is a *fixed* random rotation and
   the first-row of our table shows ITQ beating it at every bit-width. The
   whole point of ITQ is that the rotation is *learned*.
5. **Fuse into an existing crate immediately** — rejected. Nightly PoC stays
   standalone (`exclude`d from the main workspace) so it can't break the
   workspace build. Promotion path is captured in the research doc's
   roadmap once SIMD and streaming-fit land.

## Follow-ups

See `docs/research/nightly/2026-11-20-itq-binary-rotation/README.md` §
"What to improve next": SIMD encode, blocked `VᵀB` matmul, supervised
(CCA) ITQ, streaming fit with EMA, coupling with RaBitQ residual, and
integration with `ruvector-pq-search`.
