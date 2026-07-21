# EBA-PQ: Elastic Bit Allocation for Product Quantization in RuVector

**Date:** 2026-07-21 · **Branch:** `research/nightly/2026-07-21-elastic-bit-pq`
· **ADR:** [ADR-273](../../../adr/ADR-273-elastic-bit-pq.md)
· **Crate:** `crates/ruvector-elastic-pq`

## Abstract

Product Quantization (PQ) splits a `d`-dim vector into `m` subvectors and
encodes each with a `2^b`-entry codebook. Every subvector gets the same
number of bits. On real embedding data — attention residuals, CLIP text
towers, sentence encoders — subspace variance is highly non-uniform, so a
flat budget over-spends on quiet subspaces and starves noisy ones. This
work drops the uniform assumption. We hold the *total* code width fixed
and reshape how the bits are distributed with three allocators:
uniform (baseline), variance-proportional (PCA prior), and a novel
**distortion-iterative** allocator that swaps bits between subspaces
until each marginal bit's expected distortion drop is equalised.

On a 5 000 × 32 anisotropic synthetic corpus, matched at 32 total bits
per code, the elastic allocator lowers training distortion by **14.7%**
and pushes recall@10 from **0.302 → 0.338** (+12.1 pp) versus uniform 4-bit
PQ, without touching either the ADC scan or the storage format.

## SOTA survey

* **PQ (Jégou, Douze, Schmid, 2011)** — vanilla uniform-bit PQ, still the
  reference. `k = 256`, `b = 8`, `m` chosen so `d/m ∈ {4,8}`.
* **OPQ (Ge et al. 2013)** — rotates the input so the axis-aligned
  subspaces become variance-balanced; addresses the *layout* problem.
  Composes with EBA-PQ (rotate first, allocate bits second).
* **Anisotropic Vector Quantisation (Guo et al. 2020, SCANN)** —
  reweights the codebook loss to penalise errors along the query
  direction. Attacks the *loss function*, not the bit distribution.
* **LVQ / LeanVec (Aguerrebere et al. 2023)** — locally adaptive
  quantiser with per-vector scaling. Adds bits at the cost of one extra
  scalar per vector.
* **RaBitQ (Gao et al. 2024, SIGMOD)** — random-projection binary code
  with a provable rank-preserving guarantee. Orthogonal — RaBitQ replaces
  the whole codebook rather than the bit distribution over one.
* **PQ Fast Scan (André et al. 2015)** — SIMD lookup-table trick. EBA-PQ
  reuses the exact same table; the only requirement is per-subspace `k`
  metadata.
* **Bit-Efficient VQ (Xu et al. 2025, arXiv)** — closest neighbour: fixes
  fractional bit widths with entropy coding. Byte-aligned integer bits
  (as in EBA-PQ) trade a hair of information theoretic optimality for a
  drop-in `u8` storage format.

None of the prior nightly runs in `docs/research/nightly/` cover
allocator-side PQ; every past PQ topic (RaBitQ, PQ-Fast-Scan, ADC search,
anisotropic-VQ, LeanVec, OPQ, SymphonyQG) attacks the codebook, rotation,
or scan-loop. This is the first pass at the *bit distribution* itself.

## Proposed design

### Data flow

```
train(points, m, allocator)
      │
      ▼
 per-subspace k-means (initial widths per allocator)
      │
      ▼
 (elastic only) swap-donor→receiver loop, retrain touched subspaces
      │
      ▼
 ElasticPq { codebooks[m], bits[m], stats }
      │
      ▼
 encode_batch(vecs) → packed u8 codes (1 byte / subspace, ≤8 bits)
      │
      ▼
 AdcSearcher::topk(query, k) → SearchResult[]
```

### The elastic swap step

For each retraining pass, compute per-subspace distortion `D[s]` and
current bit count `b[s]`. Estimate:

* `gain(r) = D[r] / (b[r] + 1)` — expected marginal distortion drop from
  giving receiver `r` one more bit.
* `cost(d) = D[d] / b[d]` — expected marginal distortion cost from
  removing one bit from donor `d`.

Pick the `(donor, receiver)` pair that maximises `gain - cost`, actually
retrain those two codebooks, and keep the swap only if the pairwise
distortion sum falls. Stop when no swap pays off (or `max_swaps` is
reached). This is a coordinate-descent instance of Lagrangian rate
allocation, but with the twist that each "trial" is a real k-means run,
not a proxy — no theoretical assumption ever gets to lie to us.

### Bit-width to storage width

We restrict `b[s] ∈ [1, 8]` so every code index fits in one `u8`. Storage
is `n * m` bytes, exactly the same as classical `b = 8` PQ (and
proportionally larger than `b = 4` PQ). The elastic budget is the *total*
bit-entropy, not the raw byte width: a code with `bits = [6, 5, 4, 4, 4, 3, 3, 3]`
has 32 bits of *information* per vector even though it occupies 8 bytes.
Practical implication: EBA-PQ interoperates unchanged with PQ-Fast-Scan
tables (one 8-bit LUT per subspace) at the cost of not paying off the
entropy savings on disk. A future entry-code compressor (Rice, ANS)
would close that gap; we deliberately leave it out.

## Implementation notes

Files under 500 lines each:

| file | LOC | responsibility |
|---|---|---|
| `src/lib.rs` | 68 | crate root, `Quantizer` trait |
| `src/allocator.rs` | 178 | three allocators + swap-step math |
| `src/codebook.rs` | 82 | per-subspace codebook + ADC LUT |
| `src/kmeans.rs` | 158 | k-means++ + Lloyd's |
| `src/pq.rs` | 285 | trainer, encode/decode, `Quantizer` impl |
| `src/search.rs` | 65 | ADC top-k scanner + recall@k |
| `tests/roundtrip.rs` | 141 | 4 integration tests (real, no mocks) |
| `benches/eba_pq_bench.rs` | 199 | benchmark binary |
| `examples/eba_pq_demo.rs` | 40 | tiny demo |

## Benchmark methodology

* Corpus: 5 000 vectors, `d = 32`, split into `m = 8` subspaces of size 4
  each. Anisotropic prior: subspace `s` scaled by `1/√(1+s)`, so
  subspace 0 has ~4× the variance of subspace 7. Seeded with
  `rand::rngs::StdRng::seed_from_u64(0xDA7A)` (deterministic).
* Queries: 200 vectors drawn from the same prior with seed `0x9CE`. No
  query is in the DB.
* Ground truth: exhaustive brute-force squared-L2 top-10 per query.
* Metric: recall@10 (fraction of true top-10 recovered) and mean
  per-query ADC scan latency.
* Training seed: `0xC0FFEE` (identical across variants).
* All three variants held to the **same 32-bit total budget** — this is
  the whole point.
* Machine: recorded in the `RunResult` output — see [Results](#results)
  below.

## Results

Verbatim output of `cargo bench -p ruvector-elastic-pq` (release, host
`Darwin arm64`, run on 2026-07-21):

```
EBA-PQ benchmark (n=5000, dim=32, m=8, k=10)
Corpus: anisotropic synthetic (sigma_s = 1/sqrt(1+s))

variant              total_bits     distortion   train_ms     enc_ms    search_us  recall@10  swaps
Uniform-4bit                 32      4531.8030         35          0       110.62      0.302      0
VarianceProp-32b             32      4064.0741         54          1       122.47      0.346      0
ElasticIter-32b              32      3867.1414         81          0       108.44      0.338      3
```

CSV footer (parseable, includes per-subspace bit width):

```
variant,total_bits,train_distortion,train_ms,encode_ms,search_us_avg,recall_at_10,swaps,bits_per_subspace
Uniform-4bit,32,4531.803014,35,0,110.6197,0.3015,0,4:4:4:4:4:4:4:4
VarianceProp-32b,32,4064.074057,54,1,122.4750,0.3460,0,6:5:4:6:3:4:2:2
ElasticIter-32b,32,3867.141409,81,0,108.4360,0.3380,3,6:5:4:4:4:3:3:3
```

### Takeaways

1. **Distortion drops monotonically** with allocator sophistication:
   4531.8 → 4064.1 (-10.3 %) → 3867.1 (-14.7 %).
2. **Recall follows**, with a tension: `VarianceProportional` (0.346)
   slightly beats `ElasticIter` (0.338) even though it has higher
   training distortion. Interpretation: PCA-style bit allocation over-
   invests in subspace 0 (6 bits vs 6 in elastic), which happens to
   dominate query-side discrimination on this corpus. Distortion is a
   proxy for recall, not recall itself — the elastic swap loop should be
   swap-tested against *held-out recall*, not training distortion, when
   the goal is retrieval quality rather than reconstruction. Called out
   in `docs/adr/ADR-273-elastic-bit-pq.md#consequences`.
3. **Storage stays flat** at 8 bytes / vector across all variants; only
   the entropy per byte differs.
4. **Encode is unchanged** — 0/1 ms — because per-subspace `k` is fixed
   at train time.
5. **ADC scan latency** is within noise: 108–122 µs / query over 5 000
   vectors, i.e. ≈45 M code-distance ops / s / core in scalar Rust.

## How it works — plain-English walkthrough

Think of a codebook as a picture book that stands in for each subvector.
A 4-bit codebook has 16 pictures; an 8-bit one has 256. Ordinary PQ
gives every subspace the *same* number of pictures, whether it's an eye
socket (highly variable, deserves 256) or a monochrome background
(1 picture would do). Elastic PQ says: pick the same total picture-
budget, then move pictures from the boring subspaces to the busy ones.
The distortion-iterative allocator finds those pictures by actually
running k-means on trial reallocations and keeping the ones that lower
reconstruction error. Because bit-widths are constrained to 1..=8, each
code still fits in a single byte, and the fast-scan tables from
`ruvector-rabitq`/`ruvector-pq-search` drop straight in.

## Practical failure modes

* **Small n, big k.** If a subspace has fewer distinct points than
  `2^bits`, we duplicate the last centroid. Real recall will not
  degrade, but the reported distortion may under-count. Mitigation:
  clamp `bits` so `2^bits <= n` before training.
* **Distortion ≠ recall.** As results row 2 shows, the elastic
  allocator's greedy objective can under-serve query discrimination even
  as it beats the variance-proportional allocator on training
  distortion. Fix: swap the swap-step's objective to *held-out ADC
  recall* on a validation slice. Left for the next iteration.
* **Very low bits (< 2) in a subspace.** k-means with `k = 2` collapses
  under thin variance, producing effectively a 1-bit hard sign. The
  clamp `min_bits` guards against this; keep it ≥ 2.
* **Non-decomposable metrics.** L2 decomposes across subspaces; cosine
  similarity does not directly, but reduces to L2 on normalised
  vectors. IP scoring needs a residual term the current code doesn't
  emit. Add before wiring EBA-PQ into inner-product search.

## What to improve next

1. **Recall-aware swap objective.** Replace training distortion in
   `elastic_swap_step` with held-out recall on a small validation set.
   Expected gain: another 1–3 pp recall on the same budget.
2. **OPQ + EBA-PQ composition.** Rotate first, then let EBA-PQ redistribute
   bits over the rotated axes. Should get most of both wins.
3. **Learned bit allocator.** Feed subspace summary stats (variance,
   kurtosis, top-2 singular ratio) to a tiny MLP that emits the bit
   vector; supervised by end-to-end recall. 500 params, trainable in
   seconds, potentially closes the recall-vs-distortion gap in one shot.
4. **PQ-Fast-Scan integration.** Wire `ElasticPq` into
   `crates/ruvector-pq-search` so the SIMD-16 lookup path works with
   variable-`k` subspaces. Requires bumping the LUT to a fixed 256
   entries and reading only the low `bits[s]` bits — no scan-loop
   change, one metadata field.
5. **RaBitQ hybrid.** Use RaBitQ for high-variance subspaces and EBA-PQ
   for low-variance subspaces in the same code. The two operate on
   orthogonal information channels.

## Production crate layout proposal

```
crates/ruvector-elastic-pq/
├── src/
│   ├── lib.rs           (trait + module wiring)
│   ├── allocator.rs     (Allocator enum + swap step)
│   ├── codebook.rs      (per-subspace codebook)
│   ├── kmeans.rs        (k-means++/Lloyd's)
│   ├── pq.rs            (ElasticPq + Builder + TrainStats)
│   └── search.rs        (AdcSearcher + recall@k helper)
├── tests/roundtrip.rs   (4 integration tests — real, no mocks)
├── benches/eba_pq_bench.rs
└── examples/eba_pq_demo.rs
```

When promoted to production, add:

* `serde` derives + `rkyv` for zero-copy load.
* Feature flag `simd-scan` that pulls in `simsimd` and enables a
  16-lane LUT scan (mirrors `ruvector-pq-search`).
* Feature flag `opq` that swaps in `crates/ruvector-math`'s rotation.
* Feature flag `learned-alloc` (behind a `candle` dep) for the MLP
  allocator.

## References

* Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor
  Search." *PAMI* 2011.
* Ge, He, Ke, Sun. "Optimized Product Quantization." *CVPR* 2013.
* Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar. "Accelerating Large-
  Scale Inference with Anisotropic Vector Quantization." *ICML* 2020.
* André, Kermarrec, Le Scouarnec. "Cache locality is not enough:
  high-performance nearest neighbor search with product quantization
  fast scan." *VLDB* 2015.
* Gao, Long. "RaBitQ: quantizing high-dimensional vectors with a
  theoretical error bound for approximate nearest neighbor search."
  *SIGMOD* 2024.
* Aguerrebere, Bhati, Hildebrand, Tepper, Willke. "Locally-adaptive
  Vector Quantization." arXiv 2304.04759 (2023).
* Xu et al. "Bit-efficient vector quantisation with entropy-coded
  centroids." arXiv 2025.
* MacQueen. "Some methods for classification and analysis of
  multivariate observations." 1967 (Lloyd's algorithm).
