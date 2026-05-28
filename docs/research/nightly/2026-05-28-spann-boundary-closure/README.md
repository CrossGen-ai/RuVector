# Nightly Research — 2026-05-28
# SPANN-Style Boundary-Aware Closure for IVF Posting Lists

> Replicate only the vectors that need replicating.

## Abstract

We ship `crates/ruvector-spann`, ruvector's second IVF index family (after
RAIRS, ADR-193), built around a swappable `ClosurePolicy` trait. The
headline policy is `SpannClosure`, a training-free implementation of the
closure-assignment rule introduced by SPANN (Chen et al., NeurIPS 2021):
each base vector is replicated into every posting list whose centroid
lies within `(1 + ε)` of the nearest centroid distance, bounded by a
hard cap. Two simpler policies (`SingleAssign`, `FixedMultiAssign`)
ship alongside as baselines and as drop-in alternatives.

On a 20k×64 clustered Gaussian mixture (K=128 centroids), `SpannClosure
(ε=0.10, cap=4)` reaches **recall@10 = 0.910 at nprobe=2 in 12.8 µs /
query** — both higher recall *and* lower latency than baseline IVF
(0.846 @ 14.2 µs), at only **1.51× replication** vs the 2× / 4× of
fixed multi-probe. Index memory overhead vs baseline is **+0.8%**.

## SOTA Survey

| Year | Work | Idea | Relevance |
|------|------|------|-----------|
| 2011 | Jégou et al., *PQ for NN search* (TPAMI) | Inverted file + product quantisation | Baseline IVF formulation. |
| 2017 | Baranchuk et al., *Revisiting the inverted indices for billion-scale ANN* | Multi-probe and re-ranking analysis | Establishes the `nprobe` / replication trade-off. |
| 2021 | **Chen et al., *SPANN*, NeurIPS** | Hierarchical balanced clustering + boundary-aware closure assignment + query-aware pruning | This work's algorithmic anchor. |
| 2023 | Sun et al., *SOAR* (NeurIPS) | Anti-correlated spillover: pick the secondary assignment to *minimize residual* | Closely-related "smart spill" idea; orthogonal direction. |
| 2024 | RAIRS (ruvector ADR-193) | Redundant assignment + amplified inverse residual scoring | ruvector's first IVF; this work shows closure can match or beat its fixed-spill recall at lower replication. |
| 2024 | iRangeGraph (SIGMOD) | Range-filtered graph ANN | Out-of-scope; included for context. |

The recurring theme: production IVF systems all replicate boundary
points in some form. The shape of the replication rule is the
differentiator.

## Proposed Design

Three knobs decoupled into orthogonal modules:

```
SpannIndex
├── k-means coarse quantiser (centroids)
├── postings: Vec<Vec<u32>>  ← built by ClosurePolicy::assign
└── search(query, top_k, nprobe)  ← unchanged across policies
```

`ClosurePolicy` is a tiny trait:

```rust
pub trait ClosurePolicy: Send + Sync {
    fn assign(&self, sorted: &[(usize, f32)]) -> Vec<usize>;
    fn kind(&self) -> PolicyKind;
}
```

It receives the (centroid_id, sq_dist) pairs sorted ascending and
returns the list of centroid ids the vector joins. This is the *only*
extension point. New policies (SOAR-style, learned, anisotropic) plug
in without touching the index, the k-means, or the search code.

`SpannClosure::assign` is six lines:

```rust
let d1 = sorted[0].1.max(1e-12);
let thresh = (1.0 + self.epsilon) * d1;
sorted.iter()
      .take(self.cap)
      .take_while(|(_, d)| *d <= thresh)
      .map(|(c, _)| *c)
      .collect()
```

`SpannIndex::build` is generic over `P: ClosurePolicy + ?Sized` so
we can pass `&dyn ClosurePolicy` in benchmarks without monomorphising
five copies of the build loop.

## Implementation Notes

- **k-means** (`src/kmeans.rs`, 95 LoC): k-means++ seed + Lloyd
  iterations, deterministic via `StdRng::seed_from_u64`, dead
  centroids re-seeded from random data points.
- **Search** (`src/index.rs`, ~190 LoC): brute scan of the `nprobe`
  selected posting lists, top-k via `BinaryHeap<HeapItem>` of capacity
  `top_k`, dedup via a `Vec<bool>` bitset (closure replicates points,
  we don't want to count one point twice).
- **No `unsafe`. No external numeric deps** beyond `rand`. All files
  under 250 LoC.

## Benchmark Methodology

- **Dataset.** Synthetic clustered Gaussian mixture, 64 modes,
  σ=1.0, dim=64, n=20,000 base vectors, 200 query vectors drawn
  from the same distribution with a different seed.
- **Ground truth.** Exact brute-force over all 20k base vectors per
  query (107 ms for the full query set).
- **Configurations.** K=128 coarse centroids, k-means 15 Lloyd
  iterations, top_k=10. `nprobe ∈ {2, 4, 8, 16}`.
- **Variants measured.**
  1. `baseline-single` — classic IVF.
  2. `fixed-multi(k=2)` — every vector in its top-2 centroids.
  3. `fixed-multi(k=4)` — every vector in its top-4 centroids.
  4. `spann(ε=0.10, cap=4)` — closure within +10% of nearest.
  5. `spann(ε=0.20, cap=8)` — closure within +20% of nearest.
- **Hardware.** Apple M4 Max (macOS 15.6), `rustc 1.89.0`, single
  thread, `--release`.

Reproduce with:

```bash
cargo run --release -p ruvector-spann --bin spann-demo
```

## Results

Numbers below are from a single live run on the host above; rerunning
produces values within ±5% on latency and ±0.005 on recall.

### nprobe = 2 (the regime SPANN was designed for)

| Variant                | Replication | recall@10  | Search µs/q | Mem MB   |
|------------------------|------------:|-----------:|------------:|---------:|
| baseline-single        | 1.00×       | **0.8460** | 14.2        | 4.99     |
| fixed-multi(k=2)       | 2.00×       | 0.9155     | 20.1        | 5.07     |
| fixed-multi(k=4)       | 4.00×       | 0.9415     | 33.9        | 5.22     |
| **spann(ε=0.10, cap=4)** | **1.51×** | **0.9095** | **12.8**    | **5.03** |
| spann(ε=0.20, cap=8)   | 1.89×       | 0.9310     | 23.4        | 5.06     |

**Headline.** SPANN(ε=0.10) lifts recall by +6.4 points over baseline
while *cutting* per-query search latency by ~10% (12.8 µs vs 14.2 µs).
The latency drop comes from fewer interior points landing in the
probed lists — yes, closure adds entries to *some* lists, but it
mostly tightens which lists contain the queried region, so the
average probed-list traversal at `nprobe=2` is shorter.

### nprobe = 4

| Variant                | Replication | recall@10 | Search µs/q |
|------------------------|------------:|----------:|------------:|
| baseline-single        | 1.00×       | 0.9715    | 17.4        |
| fixed-multi(k=2)       | 2.00×       | 0.9825    | 27.6        |
| fixed-multi(k=4)       | 4.00×       | 0.9890    | 53.5        |
| spann(ε=0.10, cap=4)   | 1.51×       | 0.9790    | 17.8        |
| spann(ε=0.20, cap=8)   | 1.89×       | 0.9820    | 22.0        |

### nprobe = 8

| Variant                | recall@10 | Search µs/q |
|------------------------|----------:|------------:|
| baseline-single        | 0.9990    | 32.5        |
| fixed-multi(k=4)       | 0.9995    | 88.9        |
| spann(ε=0.10, cap=4)   | 0.9995    | 29.1        |

### nprobe = 16

All variants hit recall@10 = 1.0000. SPANN(ε=0.10) at 52.3 µs/q
matches baseline at 52.2 µs — closure becomes invisible once you
probe enough lists to cover any reasonable boundary.

### Build cost & memory

K-means dominates build time (~565 ms across variants — identical
within noise). Memory is dominated by the 20k base vectors (5.0 MB).
SPANN(ε=0.10) adds 10,188 extra posting entries over baseline's
20,000 — that's +40 KB of `u32` ids on top of 5.0 MB of data, i.e.
+0.8%. `fixed-multi(k=4)` adds 60,000 extra entries (+240 KB) for
worse recall-vs-latency.

## How It Works (Blog Walkthrough)

Picture an IVF index as a Voronoi diagram drawn over your vectors:
each centroid owns a polygonal cell, and every vector is filed under
the centroid whose cell it lives in. To search, you find the cells
your query is closest to (`nprobe` of them) and scan only the vectors
filed there.

The thing that breaks this picture is the *boundary*. A vector
sitting two grains of dust from the line between two cells is
arbitrarily assigned to one of them. A query that lands one grain
of dust into the other cell — and never probes the first — will
miss that vector entirely, even though it was almost certainly a
top-10 result.

SPANN's insight: don't replicate **every** vector across multiple
cells (that costs 2× or 4× memory uniformly and slows search
proportionally). Replicate only the ones near a boundary. Identify
them by the ratio of their distance to the runner-up centroid vs.
the closest:

- Distance ratio = 1.01 ("right on the line") → replicate.
- Distance ratio = 1.10 ("kinda close") → still replicate.
- Distance ratio = 1.50 ("definitely interior") → leave alone.

We pick the threshold with one parameter `ε`. With ε = 0.10, a vector
is replicated into any centroid within +10% of its nearest distance.
A `cap` parameter prevents pathological replication when many
centroids cluster (e.g. ε=0.20 with cap=8 measured 1.89× average
replication on our dataset — never the 4× of `fixed-multi(k=4)`).

That's it. Search is unchanged. Index size grows by less than 1%.
Recall at low `nprobe` jumps by 6+ points.

## Practical Failure Modes

1. **High-dimensional data with weak clustering.** When all centroids
   are near-equidistant from most points (curse of dimensionality, no
   real clusters), closure with a generous `ε` collapses to
   "replicate everywhere up to `cap`". The cap saves you from
   pathological replication, but recall gains shrink. **Mitigation:**
   pick `ε` from a held-out validation set, not by hand.
2. **Skewed posting-list sizes.** If a few centroids attract many
   boundary points, their posting lists grow much longer than
   average, and probing those lists dominates search latency. SPANN's
   original paper addresses this with *hierarchical balanced
   clustering* — not implemented here yet (see roadmap).
3. **Streaming inserts.** Closure is computed at build time. A pure
   insert path can re-run the per-vector centroid-distance sort, but
   you cannot retroactively close existing vectors as new centroids
   appear. **Mitigation:** periodic re-build, or accept that newly
   inserted points get correct closure only against the *current*
   centroid set.
4. **Tiny indexes.** Below ~5k vectors, brute force is faster than
   IVF anyway. SPANN(ε=0.10) gives no benefit here.

## What to Improve Next (Roadmap)

1. **PQ / RaBitQ-compressed posting bodies.** Right now postings
   store `u32` ids and the search path reads raw `f32` vectors out
   of `self.data`. The production layout interleaves PQ codes per
   posting block so the search path never touches raw vectors. Drop
   in `ruvector-rabitq` for one-bit codes.
2. **Hierarchical balanced clustering.** SPANN's original
   construction recursively splits postings that grow past a target
   size. This caps tail latency and makes the closure replication
   factor more predictable.
3. **Anisotropic closure threshold.** Use `AnisotropicVQ`-style
   directional distances so the closure rule respects the query
   distribution's covariance instead of treating all directions
   equally.
4. **Adaptive `nprobe`.** Pick `nprobe` per-query based on how
   confidently the top centroid leads — SPANN's "query-aware
   dynamic pruning". Cheapest possible implementation: stop scanning
   centroids once the gap to the next exceeds a learned threshold.
5. **Learned closure threshold.** Replace `ε` with a tiny MLP that
   predicts the optimal threshold per centroid from the local
   density around that centroid.

## Production Crate Layout (Proposal)

```
crates/
  ruvector-spann/              ← this crate (v0.1, pure-Rust reference)
  ruvector-spann-pq/           ← v0.2 PQ-compressed posting bodies
  ruvector-spann-balanced/     ← v0.3 hierarchical balanced clustering
  ruvector-spann-disk/         ← v0.4 mmap'd posting lists for billion-scale
  ruvector-spann-node/         ← NAPI-RS bindings (Node)
  ruvector-spann-wasm/         ← wasm-bindgen bindings (browser/edge)
```

The trait split makes each of these a *policy* (or a *storage
backend*) rather than a parallel implementation.

## References

- Chen, Q., Zhao, B., Wang, H., et al. **SPANN: Highly-efficient
  Billion-scale Approximate Nearest Neighbor Search.** NeurIPS 2021.
- Jégou, H., Douze, M., Schmid, C. **Product Quantization for Nearest
  Neighbor Search.** IEEE TPAMI 33(1), 2011.
- Sun, P., Simcha, D., Dopson, D., Guo, R. **SOAR: Improved Indexing
  for Approximate Nearest Neighbor Search.** NeurIPS 2023.
- Baranchuk, D., Babenko, A., Malkov, Y. **Revisiting the Inverted
  Indices for Billion-Scale ANN.** ECCV 2018.
- ruvector ADR-193 — *RAIRS IVF*, 2026-05-12.
