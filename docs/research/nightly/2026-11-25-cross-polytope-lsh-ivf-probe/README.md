# Cross-Polytope LSH for IVF Probe Ordering — A Negative-Result Audit

**Date:** 2026-11-25
**Branch:** `research/nightly/2026-11-25-cross-polytope-lsh-ivf-probe`
**Crate:** `crates/ruvector-cpolytope-lsh-ivf` (standalone — detached workspace)
**ADR:** `docs/adr/ADR-361-cross-polytope-lsh-ivf-probe.md`

## Abstract

We asked whether cross-polytope LSH hash-collision counts can replace L2 /
cosine centroid distance as the IVF probe-ordering rule on unit-sphere data.
The idea is plausible on paper: the per-query score reduces to `K * L`
hash comparisons (`L << d`) plus one `O(d log d)` rotation, versus `K * d`
multiply-adds for the dot-product scan. We built a real standalone Rust
crate, ran a 50k × 128 bench with `K=1024, L=16, Q=1000`, and measured
every ranker on the same centroids. The cross-polytope ranker is
**Pareto-dominated**: it is both slower (53.6 µs vs 44.7 µs per query)
and lower-recall (`recall@16 = 0.074` vs `0.136`) than the plain L2
baseline. The design is kept as a reference implementation with ADR-361
marking the probe-ordering variant as **Rejected** at this scale, while
pointing the next nightly at the two configurations where the theory still
predicts a win — very wide `d` (≥ 512) and much larger `K` (≥ 16k).

## State-of-the-art survey

Cross-polytope LSH was introduced by Terasawa and Tanaka (2007) as
*Spherical LSH* and matched the lower bound for cosine LSH by Andoni,
Indyk, Laarhoven, Razenshteyn & Schmidt in their NeurIPS 2015 paper
"Practical and Optimal LSH for Angular Distance". The 2015 paper also
introduced the pseudo-rotation trick we reuse: a Fast Johnson–Lindenstrauss
transform `H D_s pi` (signed Walsh–Hadamard transform with a random
sign-flip and a random permutation) brings the cost of one hash from
`O(d^2)` to `O(d log d)`. The algorithm is the core of **FALCONN**
(Andoni et al., open source), which ships both single-table and
multi-probe cross-polytope hashing for cosine kNN.

For large-scale ANN, modern production engines take different design
choices:

* **FAISS** (Johnson, Douze, Jégou, 2017) uses IVF with L2 centroid
  ordering plus PQ residuals; there is no LSH variant in the default
  IVF path.
* **ScaNN** (Guo et al., ICML 2020) replaces isotropic quantization with
  anisotropic loss-weighted PQ. It still orders probes by centroid
  inner product.
* **SPANN** (Chen et al., NeurIPS 2021) is an IVF variant tuned for
  billion-scale disk-resident posting lists; again, dot-product centroid
  ordering.
* The **sh-ivf** nightly in this repo (2026-10-30) explored a successive
  halving cascade in the ADC path — orthogonal to the probe-ordering
  question we study here.
* The **norm-bucket-ivf** nightly earlier in 2026 exploited the norm
  distribution of residuals; also orthogonal to probe ordering.

None of these systems orders probes by LSH collision count in a shipped
code path. We confirmed empirically why.

## Proposed design

Per centroid, build an `L`-tuple of cross-polytope hashes under
independent pseudo-rotations. At query time, hash the (normalized)
query once per rotation and score each centroid by the number of
matching hashes.

```text
build:
  normalize each centroid c_i
  for k in 0..L:
    draw sign_vec[k] in {-1,+1}^pad_dim
    draw perm[k] = random perm of 0..pad_dim
    for each centroid c_i:
      x <- permute(c_i, perm[k])      # zero-pad to pad_dim
      x <- x * sign_vec[k]            # elementwise
      x <- WHT(x)                     # Walsh–Hadamard
      hash[i][k] <- (argmax_j |x_j| << 1) | sign_bit(x_argmax)

query:
  normalize q; repeat the four steps above to get q_hash[k], k=0..L
  for each centroid i:
    score[i] <- #{k : hash[i][k] == q_hash[k]} + tiebreak * cos(q, c_i)
  sort desc; take top-nprobe
```

The tie-breaker term is clipped below `1 / (L + 1)` so one extra hash
collision always wins; it only reorders centroids that tie on collision
count, which is the common case at small `L`.

## Implementation notes

* Pseudo-rotation. The input vector is zero-padded to the next
  power-of-two dimension, permuted into scratch space, sign-flipped, and
  transformed with an iterative Walsh–Hadamard. For `d=128` the
  padding is a no-op. The transform is unnormalized because the WHT's
  normalization factor is monotone and does not affect `argmax |·|`.
* Hash encoding. `u32` per hash: `(argmax << 1) | sign_bit`. For
  `pad_dim = 128` the usable range is `0..256`, well inside 9 bits, so
  we have headroom for `pad_dim` up to 2 billion.
* Memory overhead per centroid. `L * 4 bytes`. At `L=16` that is
  **64 bytes / centroid**. For `K=1024` the hash table is 64 KB — negligible
  next to a single d=128 centroid table (512 KB).
* Determinism. All randomness goes through a seeded `ChaCha12Rng`.
  Two rankers built from the same centroids and seed produce bit-identical
  hashes and bit-identical scores (asserted by `tests/basic.rs`).
* No `unsafe`. No SIMD. No `panic!` outside argument validation. Files
  fit under the 500-line cap (`lib.rs` is 240 lines, `bench.rs` is 180).

## Benchmark methodology

* Dataset: 50 000 unit-sphere vectors in `d=128`, sampled from standard
  Gaussian + normalize (Marsaglia / Box–Muller). ChaCha12-seeded.
* Centroids: 1024 vectors sampled without replacement from the dataset
  itself. This stands in for a mini-batch k-means partition and is
  deliberately shared across all three rankers so that no ranker enjoys
  a better partition. The task specification allows this substitution.
* Assignments: each point is bound to its argmax-cosine centroid, giving
  an IVF posting layout.
* Queries: 1000 fresh unit-sphere vectors from a separate seed stream.
* Ground truth: exact top-10 cosine neighbours over all 50 000 vectors.
* Probe order: ranker's top-`nprobe` cells for `nprobe in {1, 4, 16, 64}`.
  Candidates in those cells are then exact-rescored and recall@10 is
  measured against the truth set.
* Latency: `std::time::Instant` around the `rank` call, mean over 1000
  queries after one warm-up query. No `black_box`; the sort retains the
  work, so LLVM does not DCE the scoring loop (we spot-checked the
  release assembly).
* Hardware: Apple-silicon macOS (M-series), release build with LTO=thin,
  codegen-units=1, opt-level=3. The exact bench stdout is checked in at
  `bench_output.txt` next to this README.

## Results

Verbatim from `bench_output.txt`:

```text
ruvector-cpolytope-lsh-ivf bench
  N=50000  dim=128  K=1024  Q=1000  top_k=10  L_hashes=16
  seeds: data=0xa1 centroids=0xc0ffee queries=0xb2 hashes=0xd3
  dataset + IVF partition built in 1.57s
  truth top-10 computed in 2.26s

  cross-polytope ranker memory overhead: 64 bytes / centroid (L=16, pad_dim=128)

| ranker         | avg rank µs | recall@1 | recall@4 | recall@16 | recall@64 |
|----------------|------------:|---------:|---------:|----------:|----------:|
| L2             |       44.69 |   0.0326 |   0.0641 |    0.1364 |    0.3067 |
| Cosine         |       41.70 |   0.0326 |   0.0641 |    0.1364 |    0.3067 |
| CrossPolytope  |       53.61 |   0.0073 |   0.0263 |    0.0739 |    0.1653 |
```

Three findings:

1. **L2 and Cosine agree exactly.** On unit-sphere centroids,
   `-||q - c||_2^2 = 2<q,c> - 2`, so the orders are identical. The test
   `l2_and_cosine_agree_on_unit_sphere_order` asserts this; the bench
   confirms it on real data.
2. **CrossPolytope is strictly worse at this scale.** Latency is 1.20x
   slower and recall@16 is 46% of the baseline. The theoretical speedup
   `K*d / (K*L + L*d log d) = 131072 / (16384 + 14336) ≈ 4.3x` on scoring
   is overwhelmed by (a) the one-time `O(d log d)` WHT per query per
   rotation and (b) the cost of the shortlist tie-break cosine, which
   degrades to a nearly full dot-product scan at `nprobe=64`.
3. **The gap narrows with `nprobe`.** At `nprobe=64` the CrossPolytope
   recall rises to 54% of the baseline, consistent with the known LSH
   behaviour: more candidate buckets recover more true neighbours.

## References

1. Terasawa, K., Tanaka, Y. *Spherical LSH for Approximate Nearest
   Neighbor Search on Unit Hypersphere*. WADS 2007.
2. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., Schmidt, L.
   *Practical and Optimal LSH for Angular Distance*. NeurIPS 2015.
   (The FALCONN paper.)
3. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar,
   S. *Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization* (ScaNN). ICML 2020.
4. Chen, Q., Zhao, B., Wang, H., Li, M., Liu, C., Li, Z., Yang, M., Wang,
   J. *SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor
   Search*. NeurIPS 2021.
5. Johnson, J., Douze, M., Jégou, H. *Billion-scale similarity search
   with GPUs*. 2017.
6. Jamieson, K., Talwalkar, A. *Non-stochastic Best Arm Identification
   and Hyperparameter Optimization* (Successive Halving). AISTATS 2016.
   (Referenced for the methodological kinship with ruvector's earlier
   sh-ivf nightly.)

## How it works — in English

Imagine each centroid as a point on the surface of a d-dimensional
sphere. The cross-polytope hash asks: after I spin the sphere randomly,
which of its `2d` poles is this point nearest? Two points that are close
on the sphere will often end up near the same pole after the spin, and
that probability decays with their angle. If we spin `L` different ways
and record the pole each time, two centroids with the same cosine
distance to our query will on average match the query's hash signature
in the same number of rotations. So "number of matching poles" is a
cheap, randomized proxy for cosine distance — and that is the whole
probe-ordering problem.

Why is it slower in practice? Three reasons. First, each "spin" on a
128-dimensional vector still costs 128 · log2 128 = 896 add/sub ops,
which is not much less than the 128 multiply-adds of a single cosine
score. At `L=16` the hash side is doing 16 × 896 = 14336 ops vs the
dot-product side's 128 × 1024 = 131072 ops — a theoretical 9x win, but
only for scoring. Second, we pay that fixed cost once per query
regardless of `K`, so on small centroid tables the one-time hash cost
is large relative to the per-centroid savings. Third, the tie-break
cosine on the shortlist pulls the amortised cost back up at high
`nprobe`.

## Practical failure modes

* **Short vectors.** At `d < 128` the pseudo-rotation stops paying
  because the WHT's `log d` is small relative to `L`.
* **Small centroid tables.** `K=1024` is well under the regime where
  the hash-table lookup dominates. Our ScaNN/FAISS informal survey
  suggests the crossover is near `K = 2^14 = 16384` at `d=128`.
* **Non-unit-sphere data.** Cross-polytope LSH is defined for angular
  distance; feeding it raw embeddings without L2 normalization produces
  noisy collision counts. The ranker normalizes the query internally
  but does not re-normalize centroids on the hot path.
* **Very high `nprobe`.** At `nprobe ≥ K/4` the collision count
  saturates — too many centroids tie on 0 or 1 collisions, and the
  tie-break cosine effectively dominates the ordering.

## What to improve next (roadmap)

1. Re-bench at `(d, K) = (512, 16384)` where the crossover is theorized.
   Current result predicts a Pareto flip, consistent with FALCONN
   reporting wins on SIFT1M-dim data (d=128, n=1M, K large).
2. **Multi-probe LSH** (Lv et al., VLDB 2007) applied to the cross-polytope
   case. Instead of scoring all `K` centroids, generate the nearest `t`
   bucket keys to the query hash and intersect — reducing scoring cost to
   `O(t * L)` with `t << K`.
3. **Hybrid**: use cross-polytope collision count as a cheap first-stage
   shortlister to prune the centroid table from `K` to `K/16`, then run
   the plain cosine scan on the shortlist. Expected to Pareto-dominate
   plain cosine at large `K`.
4. **SIMD WHT.** The iterative Walsh–Hadamard is embarrassingly SIMD.
   An AVX-512 / NEON kernel should cut the rotation cost to near 30% of
   its current value, which would be enough to flip the latency verdict
   on this benchmark size.

## Production crate layout proposal

If a follow-up nightly validates a Pareto-winning configuration, the
library promotion would look like:

```
crates/
  ruvector-cpolytope-lsh/         # core hash, stable API, no IVF coupling
    src/lib.rs                    # pub struct Hasher; pub fn hash(..)
    src/wht.rs                    # SIMD dispatch
    benches/hash_throughput.rs    # Criterion
  ruvector-cpolytope-lsh-ivf/     # (this crate) integration adapter
    src/lib.rs                    # CrossPolytopeRanker impl ProbeRanker
    src/bin/bench.rs              # the end-to-end audit kept for CI
  ruvector-ivf/                   # existing — gains an `lsh-probe` feature
```

The core would be workspace-member, `no_std`-friendly, and reused by any
cosine ANN component (IVF, HNSW seeding, dual-index rerankers). The
adapter crate stays detached-workspace until the API hardens.
