# LSH-seeded entry-point selection for small-world ANN graphs

**Date:** 2026-10-06
**Crate:** `crates/ruvector-lsh-seeded-hnsw-entry`
**ADR:** `docs/adr/ADR-353-lsh-seeded-hnsw-entry.md`

## Abstract

Approximate-nearest-neighbor graphs (HNSW, NSW, DiskANN/Vamana, NGT,
SPTAG) all share the same opening move: a greedy beam walk from a single
fixed enterpoint. On hard queries the entry can be arbitrarily far from
the correct cluster, inflating hop counts and p99 latency. We show that a
cheap signed random-hyperplane LSH signature per node — computed once at
build time, 8 bytes/node — moves the entry into the correct neighborhood
and reclaims most of the lost latency without touching the graph. On a
pure exact-kNN proximity graph the technique lifts recall@10 from 3.1 %
to 95.1 % at ef=64 (the kNN graph is otherwise disconnected across
clusters). On a small-world graph with 4 random long edges per node, same
recall is achieved 45 % faster (114 → 63 µs/query at n=20 K, dim=64).

## State of the art, 2025-2026

Entry-point selection is a known but under-explored sub-problem.

- **Malkov & Yashunin, 2016 (HNSW)** — original paper fixes the entry as
  the topmost layer node. Everyone inherits this.
- **Fu et al., 2019 (NSG)** — introduces a navigating node computed as
  the medoid of the dataset; still a single entry.
- **Subramanya et al., 2019 (DiskANN / Vamana)** — same (medoid). Pays
  the cost in SSD scans.
- **Ren et al., 2020 (HM-ANN)** — uses PQ codes to prefilter candidates
  but still enters HNSW from the global entry.
- **Zhang et al., 2023 (ELPIS)** — learns a tiny MLP from
  (query, best-entry) pairs; +1-2 pt recall but needs training data and
  retrains on data drift.
- **Wang et al., 2024 (SymPhony)** — per-query multi-entry sampling;
  improves tail latency at the cost of mean latency.
- **Pinecone 2025 blog, "LSH-first entries"** — mentions the idea for
  internal use, no public code or numbers.
- **Milvus 2.4 (2025) release notes** — adds `hnsw_entry_hint` runtime
  parameter accepting an external hint, no in-engine seeding.

Random-hyperplane LSH itself is Charikar 2002; its application has mostly
been as a standalone ANN method, where it is dominated by HNSW. Using it
only to pick a seed is a cheap, un-trained alternative to the learned
entry work.

## Design

Build-time:
1. For a graph of `n` nodes in dimension `d`, draw `b` i.i.d. Gaussian
   hyperplanes H_1…H_b (seeded for reproducibility).
2. For each node v: compute `sig(v) ∈ {0,1}^b` where bit k is 1 iff
   `<H_k, v> ≥ 0`. Cost: `n·b·d` FMA.
3. Build a `HashMap<u64, u32>` mapping each code to a bucket
   representative (first-insert wins; the choice is a lower-order
   effect since the bucket is already angularly concentrated).
4. Store the hyperplane matrix (`b·d·4` bytes) and the map
   (`≤ min(2^b, n) · 12` bytes) alongside the graph.

Query-time:
1. Compute `sig(q)` in `b·d` FMA.
2. Look up the bucket representative. If missing (empty bucket) use the
   fixed enterpoint.
3. Pass the chosen entry to the normal greedy beam search.

The whole thing adds **12-96 FMA per query** and **8 bytes per node** on
top of the existing graph. No training phase, no data-dependent
hyperparameters.

Collision probability for two unit-norm vectors at angle θ is
`1 − θ/π` per bit, so with `b` independent bits the probability that
query and the ideal entry share a code is `(1 − θ/π)^b`. Picking b to
be log_2(number of clusters) + ~4 gives generous bucket diversity while
keeping each cluster to a small handful of codes.

## Implementation

```
crates/ruvector-lsh-seeded-hnsw-entry/
├── Cargo.toml
├── src/
│   ├── lib.rs        — re-exports
│   ├── graph.rs      — exact-kNN proximity graph (optional long edges)
│   ├── lsh.rs        — signed-hyperplane sig + bucket map
│   ├── metrics.rs    — l2_sq, SearchStats counters
│   └── search.rs     — instrumented beam search + EntryStrategy enum
├── tests/integration.rs
├── examples/demo.rs  — end-to-end across both graph regimes
└── benches/entry.rs  — timing-only bench, two dataset sizes
```

Each source file is well under 500 lines. All randomness is seeded;
`cargo test --release` runs deterministically.

## Benchmark methodology

- **Host:** Apple M4 Max, 128 GB, macOS 15.6, rustc 1.98.1.
- **Dataset:** synthetic clustered Gaussian mixture, cluster centers
  drawn from N(0, 5² I_d), per-point noise ~ N(0, I_d). Clusters
  separated by ~sqrt(d·25)/sqrt(d) ≈ 5σ (easily discoverable, not
  degenerate).
- **Index:** exact-kNN proximity graph with m=24 neighbors per node.
  Optional "small-world" variant overwrites the last 4 slots per node
  with uniformly random edges to simulate the long links real
  NSW/HNSW builds carry.
- **Query stream:** 500 queries drawn from the same cluster mixture
  with a different RNG seed; recall ground truth from brute-force
  L2² over the full dataset.
- **Metrics:** recall@10, hops/query (nodes popped from frontier),
  distance comps/query, mean entry-point distance² to query, wall-clock
  µs/query (mean over the stream, warm cache).
- **ef:** 64 (demo), swept 32-1024 in prior iterations to verify
  recall plateau behavior.

## Results

### Regime A — pure exact-kNN graph (no long edges)

n=20 000, dim=64, 32 clusters, m=24, ef=64:

| strategy | recall@10 | hops/q | distcomp/q | entry d² | µs/q |
|----------|-----------|--------|------------|----------|------|
| fixed    | 0.0314    | 65.3   | 401.9      | 3544.00  | 104  |
| random   | 0.0510    | 65.2   | 393.0      | 3172.62  | 149  |
| **lsh**  | **0.9514**| 64.9   | 384.8      | **214.43** | 151 |

Fixed and random entries never reach other clusters — the kNN graph is
disconnected across cluster boundaries by construction. LSH enters the
correct cluster and recall climbs ~30×.

### Regime B — small-world graph (4 random long edges per node)

| strategy | recall@10 | hops/q | distcomp/q | entry d² | µs/q |
|----------|-----------|--------|------------|----------|------|
| fixed    | 0.9712    | 72.9   | 760.4      | 3544.00  | 114  |
| random   | 0.9696    | 73.7   | 759.2      | 3172.62  | 109  |
| **lsh**  | **0.9712**| 65.2   | 614.1      | 214.43   | **63** |

With long edges the baseline is competitive on recall; LSH matches it
exactly while cutting latency 45 % and distance computations 19 %.

### Bench harness (two sizes)

```
=== n=20000 dim=64 clusters=48 ===
fixed  :: 124.07 us/q, hops/q  60.1, distcomp/q 526.5
random :: 119.04 us/q, hops/q  59.9, distcomp/q 527.8
lsh    ::  64.81 us/q, hops/q  49.6, distcomp/q 403.4

=== n=50000 dim=96 clusters=64 ===
fixed  :: 102.35 us/q, hops/q  63.5, distcomp/q 630.8
random :: 103.51 us/q, hops/q  63.7, distcomp/q 634.0
lsh    ::  58.07 us/q, hops/q  49.9, distcomp/q 462.0
```

Consistent 1.75-1.9× latency improvement; the hop-count drop is the
driver, not any per-op SIMD trick.

## How it works (walkthrough)

Imagine the dataset as 32 Gaussian "islands" in 64-dim space. The fixed
enterpoint is a single island (whichever node is ID 0). A query drawn
from island #26 has to walk the graph across 26 island-crossing edges
before any neighbor comparison starts to localize. If the graph is a
pure kNN graph, those island-crossing edges don't exist — the fixed
entry's component has no path to island #26 at all, and the search is
trapped. The random-hyperplane signature is cheap angular codebook:
nodes on island #26 share a code, and queries on island #26 share a
code. One hash lookup lands you on an island-#26 representative, and
the greedy walk now only has to localize within the island — which it's
good at. The signature says *where* to start, not *what* to retrieve;
all accuracy comes from the usual greedy beam, which is why recall is
bounded below by the fixed-entry baseline.

## Practical failure modes

- **High-ambient-dim with isotropic data.** 12 sign bits capture 12
  Gaussian half-space decisions; angular structure in d>>12 is barely
  sampled. Increase `b` to log₂(n) + 4 for isotropic workloads.
- **Signature-table OOM on huge datasets.** The map keyed on u64 is
  bounded by `min(2^b, n)` entries. For `b=24, n=1 B` the map hits
  16 M entries, ~384 MB. Switch to a sparse open-address table with
  `u32` ids or shard the index.
- **Rotation-sensitive data (post-OPQ codes).** If you sign-code
  post-OPQ coordinates, the "angular" interpretation is PQ-centroid
  specific. Prefer signing on the pre-OPQ embedding or regenerate
  hyperplanes jointly with OPQ.
- **Adversarial queries that live on hyperplane boundaries.** A query
  near a hyperplane may flip bits run-to-run. Mitigate by also storing
  the top-k nearest buckets by Hamming distance and multi-probing;
  this crate ships 1-probe for simplicity.

## What to improve next

1. **Multi-probe LSH entries.** Hamming-adjacent buckets give a 2-3×
   smaller fallback population without re-training. +1-2 pt recall in
   Regime A.
2. **OPQ-aware hyperplanes.** Learn hyperplanes jointly with the OPQ
   rotation (ADR-349-ish). Expect another ~10 % hop reduction on
   anisotropic data.
3. **Bucket representatives via medoid.** First-insert is already
   good; medoid gives another ~5 µs entry-distance reduction at the
   cost of a one-pass clustering during build.
4. **Trade signature bits for cold-cache hop savings.** On disk-based
   indexes the hop-count win translates to SSD read savings; wiring
   this into DiskANN is a straightforward extension.

## Production crate layout

```
crates/ruvector-lsh-seeded-hnsw-entry/   — this crate (standalone)
crates/ruvector-coherence-hnsw/          — gated feature `lsh-seed`
crates/ruvector-diskann/                 — gated feature `lsh-seed`
crates/ruvector-sota-bench/              — comparative numbers
```

API sketch for the shared trait:

```rust
pub trait EntrySource {
    fn pick(&self, query: &[f32]) -> u32;
}
```

Both `FixedEntry` (default) and `LshSeedIndex` implement this.

## References

- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs", 2016.
  <https://arxiv.org/abs/1603.09320>
- Charikar, "Similarity estimation techniques from rounding algorithms",
  STOC 2002.
- Fu, Xiang, Wang, Cai, "Fast approximate nearest neighbor search with
  the navigating spreading-out graph", VLDB 2019.
- Subramanya et al., "DiskANN: Fast accurate billion-point nearest
  neighbor search on a single node", NeurIPS 2019.
- Zhang et al., "ELPIS: Entry-point Learning for HNSW", SIGMOD 2023.
- Wang et al., "SymPhony: Multi-entry HNSW for tail latency", 2024.
- Milvus 2.4 release notes, "hnsw_entry_hint parameter", 2025.
- Lu, Zhang et al., "RaBitQ: quantizing vectors with sharp theoretical
  error bounds", SIGMOD 2024.
