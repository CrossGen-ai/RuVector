# Ptolemaic Pivot Pruning for Exact k-NN in ruvector

**Nightly research — 2026-07-03**
**Crate:** [`crates/ruvector-ptolemaic`](../../../../crates/ruvector-ptolemaic)
**ADR:** [ADR-272](../../../adr/ADR-272-ptolemaic-pivot-pruning.md)
**Hardware:** Apple M4 Max (arm64), macOS, single-thread, `cargo bench --release`

---

## Abstract

We add **Ptolemaic pivot pruning** — a two-pivot lower-bound technique
derived from Ptolemy's inequality — to `ruvector` as a new crate,
`ruvector-ptolemaic`. The crate ships three trait-swappable backends
(linear scan, triangle-pivot, Ptolemaic-pivot), a farthest-first pivot
selector, and a real cargo-bench harness. On our M4 Max, Ptolemaic
pruning reduces distance-comparison operations (DCOs) by **up to 69.2 %
vs linear scan** and **+15 % over the triangle-inequality bound** at
identical pivot budget, but its per-candidate `O(P²)` inner loop costs
wall-clock time on cheap Euclidean-f32 kernels. The bound wins the
DCO race decisively; the wall-clock win requires either high-dim,
expensive metrics (DTW/EMD), or a rerank-shortlist deployment.

## SOTA survey

Pivot-based bound families for metric-space kNN:

- **Triangle-inequality (AESA / LAESA)** — Yianilos, 1993; Micó et al.
  Classical single-pivot lower bound. `d(q,x) >= |d(q,p) - d(x,p)|`.
- **Ptolemaic Access Method (PAM)** — Hetland, SISAP 2009. Proves
  Ptolemy's inequality is a valid, strictly tighter bound in
  Euclidean-embeddable spaces and demonstrates it on colour-histogram
  metrics; leaves ANN-integration open.
- **Bregman ball / TI-index** — Cayton, ICML 2008. Divergence-space
  analogue, complements PAM for KL-based similarity.
- **RaBitQ family** (SIGMOD 2024, extended SIGMOD 2025) — rotation +
  bit-quantised distance approximations. Trades exactness for speed.
  Present in workspace: `ruvector-rabitq`.
- **CAGRA / GPU-graph indexes** (RAPIDS, 2024) — orthogonal, GPU-only.
- **DiskANN / Vamana v2** (Neurips 2019, 2024 revisions) — graph-index
  workhorse; complementary; present as `ruvector-diskann`.

Competitor changelogs (Milvus 2.4, Qdrant 1.9, Weaviate 1.25, Pinecone,
LanceDB 0.10, FAISS 1.8) all rely exclusively on graph or IVF-PQ
structures; none ship a Ptolemaic bound. This gap makes the technique
useful as an *exact rerank* stage where a small candidate set from an
approximate index must be certified.

## Proposed design

```
┌────────────────────────────────────────────────────────────────┐
│                        KnnIndex trait                          │
│      fn search(&self, q, k) -> (Vec<Neighbor>, SearchStats)    │
└────────────────────────────────────────────────────────────────┘
        │                     │                     │
   LinearScan          TrianglePivot          PtolemaicPivot
   (baseline)          (single-pivot LB)      (pair-pivot LB, ours)
                              │                     │
                        ┌─────┴──────────────┬──────┘
                        │    PivotTable      │
                        │  N·P table         │
                        │  P·P table         │
                        │  farthest-first    │
                        │  or random select  │
                        └────────────────────┘
```

Per-query for `PtolemaicPivot`:

1. Compute `d(q, p_j)` for `j ∈ [0, P)` — `P` DCOs (amortised over N).
2. For each candidate `x_i`:
   1. Triangle bound `lb_1 = max_j |d(q,p_j) - d(x_i,p_j)|`.
   2. Ptolemaic bound over all pairs `(a, b) ∈ P × P, a < b`:
      `lb_2 = max_{a<b} |d(q,p_a)·d(x_i,p_b) − d(q,p_b)·d(x_i,p_a)| / d(p_a,p_b)`.
   3. `lb = max(lb_1, lb_2)`. Prune if `lb >= τ` (current k-th
      distance).
   4. Otherwise compute true `d(q, x_i)` and push into a bounded
      top-k structure.

Two guarantees, both unit-tested:

- **Exactness** (`triangle_and_ptolemaic_match_linear`) — identical
  top-k IDs to `LinearScan` for every query.
- **Bound dominance** (`ptolemaic_dominates_triangle`) — Ptolemaic
  prune count ≥ triangle prune count on every query, per-run.

## Implementation notes

- Zero unsafe, zero SIMD intrinsics — auto-vectorisation on aarch64
  handles the tight f32 sum-of-squares loop.
- The pivot table is row-major and `f32`; memory is
  `4·(P·dim + N·P + P²)` bytes. Sample: `N=20 000, P=10, dim=32`
  → `782.9 KB`.
- `TopK` is a small in-place-sorted vector (k ≤ 100 in practice);
  Ptolemaic pruning cares about `τ` cheaply, not about heap sift cost.
- No `rayon` — kept single-threaded to make DCO / wall-clock numbers
  comparable between backends.

## Benchmark methodology

- Deterministic Gaussian-mixture generator, seeded (`gen_dataset`,
  `gen_queries`).
- Same dataset, same queries, same pivot set feed all three backends.
- Farthest-first pivot selection (SOTA-standard).
- Metrics collected per query: `dcos`, `pruned`, `bounds_checked`,
  reported as per-query averages over 50 queries.
- Wall clock via `std::time::Instant`, `--release`.
- Correctness cross-check on every query in the correctness test
  suite (`cargo test -p ruvector-ptolemaic`, 3/3 pass).

## Results

Raw output (from `cargo bench -p ruvector-ptolemaic`, M4 Max, single-thread):

```
n=  2000 dim= 16 k=10 p= 4 | pivot-tbl=  31.6KB | Lin 2000.0/q 0.43ms | Tri 1359.9/q (32.0% cut) 0.62ms | Pto 1341.2/q (32.9% cut) 1.53ms | Pto-vs-Tri +1.4%
n=  2000 dim= 16 k=10 p= 8 | pivot-tbl=  63.2KB | Lin 2000.0/q 0.42ms | Tri  756.6/q (62.2% cut) 0.77ms | Pto  698.3/q (65.1% cut) 4.27ms | Pto-vs-Tri +7.7%
n=  2000 dim= 16 k=10 p=16 | pivot-tbl= 127.0KB | Lin 2000.0/q 0.43ms | Tri  725.6/q (63.7% cut) 1.10ms | Pto  616.7/q (69.2% cut)14.46ms | Pto-vs-Tri +15.0%
n=  5000 dim= 32 k=10 p= 8 | pivot-tbl= 157.5KB | Lin 5000.0/q 1.51ms | Tri 3104.2/q (37.9% cut) 1.99ms | Pto 2891.9/q (42.2% cut)11.73ms | Pto-vs-Tri +6.8%
n=  5000 dim= 64 k=10 p= 8 | pivot-tbl= 158.5KB | Lin 5000.0/q 3.40ms | Tri 3939.8/q (21.2% cut) 3.68ms | Pto 3810.1/q (23.8% cut)17.03ms | Pto-vs-Tri +3.3%
n=  5000 dim=128 k=10 p=12 | pivot-tbl= 240.9KB | Lin 5000.0/q 8.41ms | Tri 4558.8/q ( 8.8% cut) 8.63ms | Pto 4430.3/q (11.4% cut)34.29ms | Pto-vs-Tri +2.8%
n= 10000 dim= 64 k=20 p=12 | pivot-tbl= 472.3KB | Lin10000.0/q 7.53ms | Tri 6978.7/q (30.2% cut) 8.55ms | Pto 6476.4/q (35.2% cut)50.80ms | Pto-vs-Tri +7.2%
n= 20000 dim= 32 k=10 p=10 | pivot-tbl= 782.9KB | Lin20000.0/q 5.30ms | Tri11797.3/q (41.0% cut) 8.36ms | Pto11564.2/q (42.2% cut)67.06ms | Pto-vs-Tri +2.0%
```

Highlights:

- **Best DCO cut**: `dim=16, P=16` → **69.2 % DCOs eliminated** vs. linear.
- **Best Ptolemaic-over-triangle**: same config → **+15 % additional cuts**.
- **Bound-tightness monotonicity holds in every row** (Ptolemaic ≥ Triangle
  by construction; test enforced).
- **Wall-clock regression**: the `O(P²)` inner loop makes Ptolemaic slower
  than triangle in every tested (N, dim, P). The bound wins the algorithmic
  race but loses the constant-factor race against auto-vectorised f32
  Euclidean on M4.

## How it works (walkthrough)

Imagine four points on a page: query `q`, candidate `x`, and two pivots
`p_a`, `p_b`. Ptolemy says that for any such four points in a plane
(and, by embedding, in `R^d`):

```
|qx| · |p_a p_b|  +  |qp_b| · |xp_a|   ≥   |qp_a| · |xp_b|
```

which reshuffles to a lower bound on `|qx|` — the thing we're trying
to avoid computing. Every one of those right-hand-side quantities is
either (i) already stored in the pivot table (`|xp_a|`, `|xp_b|`,
`|p_a p_b|`) or (ii) computed once per query (`|qp_a|`, `|qp_b|`).
So the "expensive" `d(q, x)` distance never has to be evaluated for
any candidate whose Ptolemaic lower bound already exceeds the
running k-th radius `τ`. That's the entire technique — a few
multiplies replacing a `dim`-dimensional dot product.

## Practical failure modes

- **Degenerate pivot pairs.** If `d(p_a, p_b) ≈ 0`, the divisor
  collapses and the bound is useless (division skipped, treated as
  `-∞`). Farthest-first traversal makes this virtually impossible
  for `P ≤ 32`, but random-pivot mode can hit it — guarded in code.
- **Non-Ptolemaic metrics.** Ptolemy's inequality is *not* valid
  for e.g. hyperbolic distance or Bregman divergences. Using
  `PtolemaicPivot` there would return wrong results. Trait bound
  is Euclidean-only in the crate's public docs.
- **Very low-dim, cheap distance.** As benchmarks show, the constant
  factor of the pair loop drowns any DCO saving. Rule of thumb:
  don't turn this on below `dim ≈ 64` unless the metric itself is
  slow.
- **Pivot budget too high.** `O(P²)` per candidate — pushing P past
  16 wastes work without proportionate bound improvement.

## What to improve next

Prioritised roadmap:

1. **Ptolemaic-rerank over ANN candidate lists.** Hook `PtolemaicPivot`
   as an *exact* rerank layer on top of `ruvector-diskann` or HNSW —
   the shortlist is small (k' ≪ N), so `O(P²)` overhead is amortised
   across a handful of candidates where DCO savings are the dominant
   term.
2. **Best-pair pre-selection per query.** For each query, compute
   `|qp_j|` and pre-rank pivot pairs by expected bound tightness; skip
   pairs whose numerator is guaranteed small. Trims the `P²` loop.
3. **SIMD the pair loop.** Even 4-lane f32 vectorisation of the
   `(qp_a * xp_b − qp_b * xp_a)` computation would close most of the
   wall-clock gap against triangle.
4. **AESA extension.** Materialise the full `N × N` distance table for
   ≤10k datasets; Ptolemaic bounds become essentially free
   (no runtime multiplies).
5. **Non-Euclidean Ptolemaic-embeddable metrics.** Extend the trait
   to declare `is_ptolemaic()`; enable safely for χ² / Hellinger.

## Production crate layout proposal

Move to production-grade in three steps:

```
crates/ruvector-ptolemaic/
├── src/
│   ├── lib.rs               # traits + baseline backends (this PR)
│   ├── pivot.rs             # selectors (this PR)
│   ├── rerank.rs            # NEW: KnnRerank trait + shortlist adapter
│   ├── simd.rs              # NEW: portable-simd f32 pair-bound kernel
│   └── serde.rs             # NEW: PivotTable (de)serialisation
├── benches/ptolemaic_bench.rs
└── tests/correctness.rs
```

Public API additions (behind features):

- `feature = "rerank"` — `PtolemaicReranker<Ann: ApproxKnn>`.
- `feature = "simd"` — swap in SIMD kernel on x86_64/aarch64.
- `feature = "serde"` — `PivotTable::save` / `load` for cold indexes.

## References

- Hetland, M.L. (2009). *Ptolemaic Access Methods*. SISAP 2009.
- Yianilos, P.N. (1993). *Data structures and algorithms for
  nearest neighbor search in general metric spaces*. SODA.
- Chávez, E., Navarro, G., Baeza-Yates, R., Marroquín, J.L. (2001).
  *Searching in metric spaces*. ACM Computing Surveys 33(3).
- Cayton, L. (2008). *Fast nearest neighbor retrieval for Bregman
  divergences*. ICML.
- Gao, J., et al. (2024). *RaBitQ: Quantizing High-Dimensional
  Vectors with a Theoretical Error Bound*. SIGMOD.
- Micó, M.L., Oncina, J., Vidal, E. (1994). *A new version of the
  nearest-neighbour approximating and eliminating search algorithm
  (AESA)*. Pattern Recognition Letters.
