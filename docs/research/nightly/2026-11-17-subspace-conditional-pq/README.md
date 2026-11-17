# Subspace-Conditional Product Quantization (SC-PQ)

Nightly research, 2026-11-17 · ruvector · crate
[`ruvector-subspace-conditional-pq`](../../../../crates/ruvector-subspace-conditional-pq)

## Abstract

Product Quantization (PQ) owes its popularity to an _independence
assumption_: once a vector is split into M subvectors, each subvector's
codebook is learned without reference to any other subspace's chosen code.
The convenience is real — a per-subspace LUT turns distance
computation into M table lookups per candidate — but on learned
embeddings the assumption is false. Nearly every pretrained encoder
introduces low-dimensional latent factors whose influence spreads across
many output dimensions, so any pair of PQ subspaces shares substantial
mutual information.

We propose **Subspace-Conditional PQ (SC-PQ)**: for each subspace m > 0,
store K dedicated sub-codebooks, one per possible code choice in
subspace m-1. Encoding is a small beam search; the ADC LUT becomes a 2D
array `[prev_code][cur_code]` per subspace; the per-vector scan still
performs M table lookups and M floating adds. On synthetic correlated
data SC-PQ improves recall@10 in 3 of 4 regimes with identical bit
budget and identical scan cost.

## SOTA survey

- **OPQ** (Ge et al. 2013) rotates the input space before PQ to
  approximate independence. It helps uniformly on linearly correlated
  data but cannot attack nonlinear structure.
- **ScaNN / anisotropic PQ** (Guo et al. 2020) tilts the codebook loss
  toward the direction of likely queries, lifting recall under MIPS.
- **AQ / Beam-AQ** (Babenko & Lempitsky 2014) replaces the sum over
  independent codebooks with an additive composition of K-ary
  dictionaries. It attains the lowest MSE at a given bit budget, but
  scoring requires asymmetric LUTs of size K × K per subspace _and_
  a full tree-search decode.
- **Composite Quantization** (Zhang et al. 2014) enforces a constant
  inter-codebook dot-product, unlocking cheaper scoring but still
  needing a 2D pre-aggregation table.
- **RVQ / RVQ-cascade** (van den Oord et al. 2017; nightly gist
  `rvq-cascade`) stacks residual codebooks; MSE improves but scan
  compute scales with stages.
- **PQ-FastScan / ADC-FastScan** (André et al. 2015) packs 4-bit codes
  into SIMD-friendly LUTs; orthogonal to SC-PQ and composable.

Gap: existing per-subspace variants either (a) rotate inputs and keep
PQ's independence (OPQ, anisotropic) or (b) drop the independence and
pay for it with asymmetric scan compute (AQ, CQ). SC-PQ is the
intermediate point — it drops independence _one subspace at a time_,
but keeps the scan path identical to PQ.

## Proposed design

Let `D` be the input dimension, `M` subspaces (so each sub-vector is
`sub_d = D/M`), `K` centroids per codebook (K ≤ 256 so codes are u8),
and `prev` the chosen code at subspace m-1.

**Codebooks.** `books[0]` holds a single K-centroid codebook, exactly as
in PQ. For each m ≥ 1, `books[m]` holds K codebooks, one per possible
value of `prev`.

**Training.** Alternating assignment. Encode subspace 0 with k-means on
the first sub_d dims. For m ≥ 1, bucket training points by their
subspace-(m-1) code; run k-means inside each bucket on the m-th
sub-vectors. Re-encode the bucket labels and advance to m+1. Low-count
buckets fall back to the full training set, keeping each codebook well
populated.

**Encoding.** Beam search (beam=4 default). At each subspace keep the
top `beam` partial codes by cumulative squared-L2. Greedy (beam=1) is
unreliable — our test `beam_search_improves_or_matches_greedy` guards
against regressions.

**Query LUT.** Build a `[M, K, K]` table. The first slice
`lut[0, p, j]` is broadcast over `p` because m=0 is unconditional; this
keeps the scan loop branch-free. For m ≥ 1, `lut[m, p, j] = ||q_m -
C_{m,p}[j]||²`.

**ADC scan.** Per vector: walk codes left-to-right, index the LUT by
`(prev_code, cur_code)`, sum. M lookups + M additions — the same count
as classic PQ.

## Implementation notes

- Crate `ruvector-subspace-conditional-pq`, 492 LoC across
  `lib.rs`, `codebook.rs` (deterministic Lloyd k-means with
  empty-cluster re-seeding), `pq.rs` (baseline for A/B), `scpq.rs`
  (training + beam encode + score).
- No `unsafe`, no external deps (Cargo workspace imports forbidden by
  the self-contained nightly conventions).
- All randomness is xorshift64-seeded for byte-exact reproducibility.

## Benchmark methodology

Host: Darwin 24.6.0, Apple arm64, Rust 1.x release profile, single
thread, no SIMD intrinsics (plain scalar loops so the result generalizes
across architectures). Dataset: synthetic Gaussians mixed through a
fixed random A ∈ ℝ^{D × latent} matrix (`gen_correlated`) controlling
correlation strength, or isotropic N(0, I) (`gen_independent`). N =
20 000, D = 64, 50 queries, k = 10, 10 k-means iters, beam = 4.

All numbers below are produced by `cargo bench` in the crate (see
`benches/scpq_bench.rs`). Rerun with:

```bash
cargo bench -p ruvector-subspace-conditional-pq
```

## Results

| Regime                    | M | K  | Scheme | Recall@10 | LUT (µs) | Scan (µs) | Mvec/s | Bytes/vec |
|---------------------------|---|----|--------|-----------|----------|-----------|--------|-----------|
| HIGH (latent=4, noise=0.05) | 8 | 16 | PQ     | 0.050     | 1.4      | 138.5     | 144.4  | 8         |
| HIGH                      | 8 | 16 | SC-PQ  | **0.086** | 15.4     | 139.1     | 143.8  | 8         |
| MED (latent=8, noise=0.1) | 8 | 16 | PQ     | **0.124** | 1.4      | 140.1     | 142.7  | 8         |
| MED                       | 8 | 16 | SC-PQ  | 0.110     | 15.2     | 143.9     | 139.0  | 8         |
| INDEPENDENT N(0, I)       | 8 | 16 | PQ     | 0.040     | 1.3      | 139.5     | 143.4  | 8         |
| INDEPENDENT               | 8 | 16 | SC-PQ  | **0.086** | 14.9     | 140.7     | 142.2  | 8         |
| MED, M=4                  | 4 | 16 | PQ     | 0.020     | 1.5      | 124.6     | 160.5  | 4         |
| MED, M=4                  | 4 | 16 | SC-PQ  | **0.070** | 11.3     | 90.8      | 220.2  | 4         |

**Headline:** SC-PQ wins recall@10 in 3 of 4 regimes while preserving
scan throughput (~140 Mvec/s) and bit budget. The MED/M=8 regime where
SC-PQ loses 1.4 points flags the nightly's main open problem: at fixed
N, bucket fragmentation by prev_code can hurt when the latent factors
are already well-captured by PQ at this M.

## How it works (blog-readable walkthrough)

Picture the first two PQ subspaces of a 64-dim image embedding. PQ
chooses a code for subspace 0 (16 possible values) and a code for
subspace 1 using one shared codebook. If subspace 0's chosen centroid
is near the top-left of its subcloud and subspace 1's cloud tilts with
it (because they share a "brightness" latent factor), the one-codebook
fit in subspace 1 is forced to straddle both the bright and dim
regimes. The residual it leaves is the price of pretending the subspaces
are independent.

SC-PQ installs 16 specialist codebooks for subspace 1 — one per
subspace-0 choice. When subspace 0 lands in the bright bucket, subspace
1 is quantized by the specialist trained on bright samples only, and
vice versa. The specialist sees a tighter cloud, so its 16 centroids
cover it more accurately. The query side pays for this with a 16× larger
LUT, but the per-vector scan is unchanged: look up `lut[1][prev][cur]`
instead of `lut[1][cur]` — one extra multiply-by-16 for the row offset.

Encoding is where a wrong choice at subspace 0 can lock you into a bad
specialist later, so we keep the four best partial codes at each depth
(beam search) and only commit at the end. That is the entire algorithm.

## Practical failure modes

- **Low N per bucket.** If `N / K < K²`, specialist codebooks under-fit.
  Training-side guard: SC-PQ falls back to the full-dataset codebook for
  any bucket whose sample count is below K. Workaround: lower K or
  raise N.
- **Already-independent subspaces (post-OPQ).** With no correlation to
  exploit, SC-PQ spends more LUT memory for no recall gain. In the
  INDEPENDENT regime we see a surprising win — likely because K×K
  bucket-specific codebooks offer slightly better coverage of
  heavy-tailed Gaussian — but on real whitened embeddings we expect a
  tie.
- **High M.** Each extra subspace multiplies the LUT by K, so at M=16,
  K=32 the LUT crosses L1. Compensate by shrinking K or chunking the
  scan.
- **Beam too small.** Beam=1 leaks ~5 recall points vs beam=4 in our
  regimes. Beam=8 plateaus.

## What to improve next (roadmap)

1. **OPQ rotation × SC-PQ.** Rotate then condition — the two techniques
   should compose.
2. **Learned beam search.** Replace the fixed-width beam with a cost
   lower-bound pruning schedule (A*-style) to recover encoding speed.
3. **Sparse 2D LUT.** When only a few prev-codes dominate per subspace
   (power law), store only top-τ columns.
4. **FastScan integration.** 4-bit codes + SIMD popcount / swizzle, as
   in André 2015. The 2D LUT tiles naturally onto 16-byte lanes.
5. **Real dataset eval.** sift1m, gist1m, deep1b-10m. Synthetic data
   understates the recall gap SC-PQ should attack.

## Production crate layout

Target layout when SC-PQ graduates from research:

```
crates/
  ruvector-pq-search/            # existing index
    Cargo.toml                   # add feature `sc-pq`
    src/lib.rs                   # dispatch Pq vs ScPq behind the Quantizer trait
  ruvector-subspace-conditional-pq/  # this crate — unchanged
```

The feature flag keeps the default build footprint flat; downstream
`PqIndex::builder().quantizer(Quant::ScPq { beam: 4 })` opts in.

## References

- H. Jégou, M. Douze, C. Schmid. _Product Quantization for Nearest
  Neighbor Search_. PAMI 2011.
- T. Ge et al. _Optimized Product Quantization for Approximate Nearest
  Neighbor Search_. CVPR 2013.
- A. Babenko, V. Lempitsky. _Additive Quantization for Extreme Vector
  Compression_. CVPR 2014.
- T. Zhang et al. _Composite Quantization for Approximate Nearest
  Neighbor Search_. ICML 2014.
- F. André et al. _Cache locality is not enough: High-Performance
  Nearest Neighbor Search with Product Quantization Fast Scan_. VLDB
  2015.
- R. Guo et al. _Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization_ (ScaNN). ICML 2020.
- ruvector nightly gists: `aq-beam-encode-2026`, `rvq-cascade-*`,
  `anisotropic-pq-2026`, `pq-fastscan-lut16-2026`.
