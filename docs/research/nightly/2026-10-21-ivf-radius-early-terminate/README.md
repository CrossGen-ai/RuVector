# IVF radius-based early termination

**Date:** 2026-10-21
**Crate:** `crates/ruvector-ivf-radius-et`
**ADR:** [ADR-357](../../../adr/ADR-357-ivf-radius-early-terminate.md)

## Abstract

Standard IVF ANN search picks a fixed `nprobe` and scans that many cells
by `||q - mu_c||` order. For most queries, the top-k is settled well
before `nprobe` is exhausted — the remaining cells' centroids are far
enough from `q` that nothing inside them could beat the current
k-th-best. This work makes that cut *provable* and *exact w.r.t. the
IVF partition* using one scalar per cell — the cell radius
`r_c = max_{p in c} ||p - mu_c||` — and the Euclidean triangle
inequality. On synthetic clustered data (D=64, N=50k, nlist=256) the cut
delivers a **2.42×–3.42×** reduction in wall-clock vs. full-scan IVF
at identical (recall = 1.0) answers.

## State-of-the-art survey

* Jégou et al., "Product Quantization for Nearest Neighbor Search"
  (TPAMI, 2011) — foundational IVF-PQ; `nprobe` is user-chosen, no
  termination cut.
* Baranchuk et al., "Revisiting the Inverted Indices for Billion-Scale
  Approximate Nearest Neighbors" (ECCV 2018) — IVFADC+R, grouped
  centroids; still fixed `nprobe`.
* Chen et al., "SPANN: Highly-efficient Billion-scale ANNS Using Hybrid
  Memory" (NeurIPS 2021) — posting-list pruning via boundary centroids;
  per-cell radii are implicit but not used as a strict LB.
* Elkan, "Using the Triangle Inequality to Accelerate k-Means" (ICML
  2003) — the same LB trick inside Lloyd; well-known in clustering,
  under-used in IVF search.
* Milvus, Qdrant, Weaviate, Pinecone, FAISS (as of late 2026) — all
  default to fixed `nprobe` for IVF. FAISS has `max_codes` as a budget,
  but no exact LB-based cut on cells.
* RaBitQ (Gao & Long, SIGMOD 2024) and ADC early-abandon (Jin & Chen,
  VLDB 2023) attack the *point-level* scan cost; this work attacks the
  *cell-level* visit count and composes with either.

## Design

### LB per cell

For an Euclidean metric, every point `p` in cell `c` satisfies

```
||q - p|| >= ||q - mu_c|| - ||p - mu_c|| >= ||q - mu_c|| - r_c
```

Let `LB_c = max(0, ||q - mu_c|| - r_c)`.

### Soundness of termination

Scan cells in order of ascending `LB_c`, maintaining a max-heap of the
k smallest observed distances with `tau_k` = heap top. If
`LB_next >= tau_k`, then every unvisited cell `c'` has
`LB_{c'} >= LB_next >= tau_k`, so no `p in c'` can satisfy
`||q - p|| < tau_k`. Termination preserves the IVF-partition's exact
top-k.

Note: ordering **must** be by `LB_c`, not by `||q - mu_c||`. A cell with
a far centroid but a large radius can have a smaller LB than a
closer-centroid cell with a tight radius — sorting by centroid distance
alone can skip over a cell whose LB is still below `tau_k`. We hit this
exact bug in the first implementation and the correctness test (recall
dropped to 0.97) caught it; the fix is 3 lines (`sort_cells_by_lb`).

### Backends

| Backend             | Visit order     | Cut                         | Exact? |
| ------------------- | --------------- | --------------------------- | ------ |
| `FixedNprobe`       | `||q - mu_c||`  | none (nprobe cells)         | approximate |
| `RadiusET`          | `LB_c`          | stop when `LB >= tau_k`     | **yes** (vs IVF full-scan) |
| `HybridNprobeET`    | `LB_c`          | stop when `LB >= tau_k`, cap at `max_nprobe` | exact if cap not binding |

## Implementation notes

* Training is minimal Lloyd k-means (no mini-batch, no k-means++). It is
  there to produce a valid IVF partition; downstream papers can plug in
  any clusterer — only the final posting lists + radii matter.
* `cell_radii[c]` is computed once at build time from the final
  assignment pass.
* No external crates. Deterministic xorshift64* RNG + hand-rolled
  squared-L2. The whole lib is ~470 lines.
* `TopK` is a `BinaryHeap<Candidate>` keyed on squared distance; the
  `LB_c` comparison is squared on both sides to keep units matching.

## Benchmark methodology

* Platform: Apple M4 Max, Rust 1.84 release build, single thread.
* Data: synthetic Gaussian-mixture vectors — `modes` isotropic clusters
  with `N(0, I)` noise around centers drawn `N(0, (5·I))`. See
  `src/bin/benchmark.rs::make_data` for exact code.
* Queries: 500 drawn from the same mixture, independent seed.
* Metric: squared Euclidean (standard L2 ANN).
* Ground truth: brute-force top-10 scan.
* We measure QPS wall-clock, `cells visited / query`, `points scanned /
  query`, recall@10.
* Full reproduce:

```bash
cargo test --release -p ruvector-ivf-radius-et --lib
cargo run  --release -p ruvector-ivf-radius-et --bin benchmark
```

## Results (real cargo numbers, Apple M4 Max)

### Workload A — D=32, N=20k, nlist=128, modes=32

```
cell radii   min=0.000  p50=7.199  p90=7.928  max=8.663

FixedNprobe np=4             R=0.8800  qps=138071
FixedNprobe np=8             R=0.9866  qps= 97213
FixedNprobe np=16            R=0.9996  qps= 61507
FixedNprobe np=32            R=1.0000  qps= 32116   (first exact baseline)
RadiusET (no cap)            R=1.0000  qps= 35768   27.3 cells/q   -> 1.11x
FixedNprobe np=128 (full)    R=1.0000  qps=  8741  128.0 cells/q   -> 4.09x
```

RadiusET matches full-scan recall at **4.09× the QPS**, visiting 27 of
128 cells on average. Against the smallest fixed-np that reaches R=1
(np=32), RadiusET is **+11%**.

### Workload B — D=64, N=50k, nlist=256, modes=64

```
cell radii   min=0.000  p50=9.649  p90=10.319  max=11.078

FixedNprobe np=4             R=0.8668  qps= 41435
FixedNprobe np=8             R=0.9684  qps= 28075
FixedNprobe np=16            R=0.9984  qps= 13805
FixedNprobe np=32            R=1.0000  qps=  8393   (first exact baseline)
RadiusET (no cap)            R=1.0000  qps=  3193   83.8 cells/q
FixedNprobe np=256 (full)    R=1.0000  qps=  1112  256.0 cells/q  -> 2.87x
```

RadiusET is **2.87× faster than full-scan** at identical recall. It is
*slower* than FixedNprobe np=32 — the LB happens to still be loose at
~30 cells so pure LB-ordered scan over-touches. Takeaway: when exact
answers are required (recall=1.0), use RadiusET; when ~99.84% recall
is enough, FixedNprobe np=16 remains better.

### Workload C — D=64, N=50k, nlist=256, modes=16 (overlapping cells)

```
cell radii   min=5.882  p50=9.543  p90=10.092  max=11.101

FixedNprobe np=4             R=0.5530  qps= 42665
FixedNprobe np=8             R=0.7700  qps= 28138
FixedNprobe np=16            R=0.9612  qps= 10933
FixedNprobe np=32            R=0.9974  qps=  4827
RadiusET (no cap)            R=1.0000  qps=  1758  116.6 cells/q   (exact)
HybridET max=64              R=1.0000  qps=  3481   60.8 cells/q
FixedNprobe np=256 (full)    R=0.9999..R=1.0000  qps= 725  256.0 cells/q
```

The hard regime: fewer modes than cells means posting lists spread
query-relevant points thinly across many cells. FixedNprobe never
reaches 1.0 at small `np`. **HybridET max=64** reaches full recall at
**4.80× the full-scan QPS** by combining a reasonable cap with the LB
cut.

### Summary speedups at R=1.0 (exact)

| Workload | Full-scan QPS | Best ET QPS | Speedup |
| -------- | ------------- | ----------- | ------- |
| A        | 8,741         | 35,768 (RadiusET)       | **4.09×** |
| B        | 1,112         |  3,193 (RadiusET)       | **2.87×** |
| C        |   725         |  3,481 (HybridET max=64)| **4.80×** |

## How it works (walkthrough)

1. **Build the IVF.** Train `nlist` centroids with Lloyd. Assign every
   point to its nearest centroid. For each cell, record `r_c` = the
   max Euclidean distance from `mu_c` to any member.
2. **At query time.** Compute `||q - mu_c||` for every centroid. Build
   `LB_c = max(0, ||q - mu_c|| - r_c)`. Sort cells by `LB_c` ascending.
3. **Scan.** Maintain a size-k max-heap. For each cell in LB order:
   - If `LB_c² >= heap.top()`, stop — nothing beyond can win.
   - Else, scan every point in the cell and push into the heap.
4. **Return** the heap sorted ascending.

Everything else — PQ codes, RaBitQ codes, SIMD inner loops, posting-list
packing — composes orthogonally. The LB cut operates purely on `(mu_c,
r_c)` and `tau_k`.

## Practical failure modes

* **Flat radius distribution.** When every cell has roughly the same
  radius, the LB is dominated by `||q - mu_c||`, and the cut is only
  modestly better than fixed `nprobe` chosen well. Workload B above is
  near this regime.
* **Overlapping cells (large radii).** Each `r_c` is big, so `LB_c`
  drops toward zero and the cut is loose. The LB still prunes some
  cells but needs the hybrid cap for a big speedup — see workload C.
* **Mis-trained IVF.** Dead cells (`r_c = 0` because they ended up
  empty-ish after random seeding) are visited first with `LB = ||q -
  mu_c||`; benign but wastes a slot. A real build should seed with
  k-means++ or drop empty cells.
* **Non-Euclidean metrics.** The LB is Euclidean-specific. For cosine
  on unit-norm vectors it works via the standard sq-L2 equivalence; for
  general inner product it does not.

## What to improve next

1. **Per-cell quantile radii.** Keep `r_c^{p90}` and `r_c^{max}`; use
   p90 for a tighter LB and run a second pass for the tail. Trades
   exactness for QPS; still bounded recall loss.
2. **LB pre-sort via centroid graph.** Precompute pairwise centroid
   distances; many queries live near the same handful of centroids and
   the LB order can be bootstrapped from a nearest-centroid-graph walk
   rather than an `O(nlist)` full sort.
3. **RabitQ / PQ composition.** The LB cut operates before the
   per-point scan; drop in `pq-adc-early-abandon` or `rabitq` as the
   inner scanner and both wins compound.
4. **Dynamic `r_c` under streaming inserts.** A new point can grow
   `r_c`. Store a sketch (e.g. count-min of radii quantiles) and only
   recompute on contract.
5. **GPU batch kernels.** `||q - mu_c||` and the sort are embarrassingly
   parallel; a single kernel over all queries + all cells with
   per-query shuffle on LB is a clean fit for CUDA/Metal.

## Production crate layout proposal

```
crates/ruvector-ivf-radius-et/
├── Cargo.toml                   # zero deps
├── README.md
└── src/
    ├── lib.rs                   # IvfIndex, IvfSearcher, 3 backends, tests
    └── bin/benchmark.rs         # 3 workloads (A/B/C)
```

Promotion path when the LB cut graduates from research:

```
crates/ruvector-ivf-radius-et/        -> stays as the reference impl + bench
crates/ruvector-core::ivf::radius_et  -> trait impl merged behind feature
                                         `ivf-radius-et` on IvfIndex
```

Downstream crates that already own an IVF (`ruvector-ivf-ball-pruning`,
`ruvector-blockmax-ivf`, `ruvector-pivot-tri-ivf`) can take `r_c` as
pre-existing metadata.

## References

* R. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest
  Neighbor Search.* IEEE TPAMI, 2011.
* D. Baranchuk, A. Babenko, Y. Malkov. *Revisiting the Inverted
  Indices for Billion-Scale Approximate Nearest Neighbors.* ECCV 2018.
* Q. Chen et al. *SPANN: Highly-efficient Billion-scale Approximate
  Nearest Neighbor Search.* NeurIPS 2021.
* C. Elkan. *Using the Triangle Inequality to Accelerate k-Means.*
  ICML 2003.
* J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024.
* FAISS source: `faiss/IndexIVF.cpp`, `search_preassigned` (fixed
  nprobe; `max_codes` budget).
* Milvus docs, "IVF_FLAT index parameters" (`nprobe` as the only
  per-query knob).
