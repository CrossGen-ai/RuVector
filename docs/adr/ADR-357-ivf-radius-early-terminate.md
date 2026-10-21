# ADR-357 — IVF probe-loop early termination via per-cell radius

* **Status:** Accepted (research/nightly, 2026-10-21)
* **Date:** 2026-10-21
* **Owner:** nightly research
* **Crate:** `crates/ruvector-ivf-radius-et`
* **Research:** [`docs/research/nightly/2026-10-21-ivf-radius-early-terminate`](../research/nightly/2026-10-21-ivf-radius-early-terminate/README.md)

## Context

Every IVF implementation in ruvector today — `ruvector-blockmax-ivf`,
`ruvector-ef-ivf-posting`, `ruvector-pivot-tri-ivf`,
`ruvector-ivf-ball-pruning`, `ruvector-pq-adc-early-abandon` — picks a
scalar `nprobe` and scans that many cells by centroid-distance order.

Two consequences:

1. Picking `nprobe` is a static recall/latency trade-off per workload.
   The point at which fixed `nprobe` reaches exact recall is often
   several multiples of the point at which the k-th-best is already
   stable.
2. For a *provable* exact answer you must set `nprobe = nlist` (full
   scan), which is wasteful: on our workload B (D=64, N=50k,
   nlist=256), 2.87× of the per-query wall-clock is spent in cells
   that cannot possibly contain a top-k result.

Every point `p` assigned to IVF cell `c` with centroid `mu_c` and
radius `r_c = max_{p in c} ||p - mu_c||` satisfies the Euclidean
triangle lower bound `||q - p|| >= ||q - mu_c|| - r_c`. Define
`LB_c = max(0, ||q - mu_c|| - r_c)`. If cells are visited in increasing
`LB_c`, the moment `LB_c >= tau_k` (the current k-th-best distance),
every unvisited cell has `LB_{c'} >= LB_c >= tau_k`, and no unvisited
point can beat the running top-k.

This gives an *exact-vs-IVF-full-scan* early-termination rule using one
scalar per cell. Milvus/Qdrant/Weaviate/Pinecone/FAISS do not ship it;
FAISS `max_codes` is a budget, not a provable cut. Elkan's trick is
well known inside k-means training but is under-used at query time.

## Decision

Ship a research-grade crate `ruvector-ivf-radius-et` with:

1. A minimal `IvfIndex` carrying `(centroids, posting_lists, cell_radii,
   vectors)`, trained with Lloyd k-means.
2. A trait `IvfSearcher { fn search(index, q, k) -> (topk, stats); }`
   with three in-tree backends: `FixedNprobe` (baseline), `RadiusET`
   (exact LB cut), `HybridNprobeET` (LB-sorted scan with a cap).
3. A benchmark binary that runs three synthetic workloads (A: small
   well-separated, B: medium balanced, C: overlapping cells) and
   reports recall@10, QPS, cells/q, points/q against brute-force
   ground truth.
4. A `cells sorted by LB_c` invariant, enforced by `sort_cells_by_lb`
   — not by `||q - mu_c||`. (A first-pass impl sorted by centroid
   distance; the unit test `radius_et_achieves_full_recall_vs_brute`
   caught a 0.97 recall drop and the fix is baked in.)

The crate takes zero external dependencies — a deterministic
xorshift64* RNG, hand-rolled squared-L2, and the standard library's
`BinaryHeap`. Under 500 lines total (lib ~ 470, benchmark ~ 140).

## Consequences

### Positive

* `RadiusET` is **exact w.r.t. IVF full-scan** and never worse by
  more than the LB sort/compute cost.
* Measured end-to-end wall-clock speedups vs. full-scan IVF at
  recall = 1.0 on an M4 Max:

  | Workload | Full-scan QPS | ET QPS           | Speedup |
  | -------- | ------------- | ---------------- | ------- |
  | A (D=32, N=20k)        | 8,741  | 35,768 (RadiusET)       | **4.09×** |
  | B (D=64, N=50k)        | 1,112  |  3,193 (RadiusET)       | **2.87×** |
  | C (D=64, N=50k, 16 modes) | 725 |  3,481 (HybridET max=64)| **4.80×** |

* The cut is one scalar per cell and is orthogonal to any per-point
  acceleration (PQ, RaBitQ, SIMD) — it composes.

### Negative

* On workloads where the practical answer is "99% recall at a tenth of
  the latency," a well-chosen fixed `nprobe` can beat `RadiusET`
  (workload B: FixedNprobe np=16 at R=0.9984 is 4.3× RadiusET's QPS).
  The ET rule is strongest in two cases: when exact recall matters,
  or when cells overlap heavily and fixed `nprobe` never reaches 1.0.
* `cell_radii` is one f32 per cell — negligible overhead (nlist=256 →
  1 KiB) — but it is a static quantity. Streaming inserts can grow
  `r_c` and relax the LB over time; a production impl needs an
  update path.
* The lower-bound is Euclidean-specific. Angular/cosine via sq-L2
  equivalence works; general inner-product does not.

### Alternatives considered

* **`max_codes` budget (FAISS).** Stops after a fixed point count;
  not a provable cut. Rejected on grounds of correctness.
* **Sort by `||q - mu_c||`, cut at `||q - mu_next|| - r_max >= tau_k`.**
  Uses the *global* `r_max`, loosest possible LB. We would under-prune
  compared to per-cell LB.
* **Centroid-only pruning (`||q - mu_c||`-based nprobe).** Equivalent
  to today's `FixedNprobe`; is the baseline we beat.
* **Learned per-query `nprobe` predictor.** Interesting (there are
  recent 2024–2026 papers on this), but trades correctness for a
  learned confidence; orthogonal to this cut and composable with it.

## Rollout

* Nightly branch only: `research/nightly/2026-10-21-ivf-radius-early-terminate`.
* No upstream (ruvnet) PR; the fork keeps nightly research private per
  the 2026-05-09 user instruction.
* Promotion path if adopted: fold `cell_radii: Vec<f32>` into
  `ruvector-core::ivf::IvfPartition`, add a `search_radius_et` method
  behind an `ivf-radius-et` cargo feature, let downstream IVF crates
  opt in. The research crate stays as the reference impl + bench.
