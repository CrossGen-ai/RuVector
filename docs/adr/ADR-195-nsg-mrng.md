---
adr: 195
title: "NSG — Navigating Spreading-out Graph for ANN search"
status: accepted
date: 2026-05-13
authors: [nightly-claude]
related: [ADR-193, ADR-194]
tags: [ann, graph-index, nsg, mrng, nn-descent, vamana, nightly-research]
---

# ADR-195 — NSG: ruvector's first single-layer graph ANN index

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-13-nsg-mrng` as `crates/ruvector-nsg`. All unit
tests pass; `cargo build --release -p ruvector-nsg` is green.

## Context

ruvector ships two graph ANN indexes today — HNSW (in `ruvector-core`) and
DiskANN/Vamana (in `ruvector-diskann`) — plus an IVF family (RAIRS,
ADR-193) and quantisation (RaBitQ, LeanVec). What is missing is a clean,
single-layer **Navigating Spreading-out Graph** (NSG) — the algorithm of
Fu, Xiang, Wang & Cai, VLDB 2019, [arXiv:1707.00143][nsg].

Why bother given we already have HNSW and DiskANN?

| Property | HNSW | DiskANN / Vamana | **NSG** |
|---|---|---|---|
| Layers | O(log N) | 1 | **1** |
| Memory | high (multi-layer + skip-list) | medium | **lowest** |
| Build | online inserts | batch (single-pass) | batch |
| Edge rule | heuristic select | α-relaxed MRNG | strict MRNG (+ α option) |
| Theoretical search bound | none | none | **monotonic search path** |
| Best-known recall/QPS frontier | strong | very strong | competitive on static workloads |

NSG occupies a useful slot: lowest memory of the three, with the tightest
theoretical guarantee (Monotonic Relative Neighbourhood Graph implies a
monotonic descent path from the navigating node to every base point — so
greedy search never gets stuck in a local minimum, given perfect pruning).

## Decision

We add `crates/ruvector-nsg` implementing the canonical four-step pipeline:

1. **NN-Descent** k-NN graph init (Dong, Charikar & Li, WWW 2011) — random
   init, then iterate local-join with forward *and* reverse neighbours.
2. **Navigating node** = base point closest to the dataset centroid (via a
   cheap greedy walk on the k-NN graph).
3. **MRNG edge selection** per node from a candidate pool that is the
   union of (a) greedy beam-search from the navigating node toward `p`
   and (b) `p`'s own k-NN entries. The latter is essential — without it,
   the pool lacks cross-cluster bridges on real-world data.
4. **Reverse-edge insertion + per-node MRNG re-prune** (Fu et al. Alg. 2
   step 6), then **DFS tree augmentation** so every vertex is reachable
   from the navigating node.

We also expose a **Vamana-style α ≥ 1.0 relaxation** of the occlusion
test — `α · d(s, c) < d(p, c)` instead of strict `<`. `α = 1.0` is exact
NSG; `α = 1.2` is the DiskANN default and gives a denser graph with
higher recall on workloads with tight clusters. Setting α is a one-line
parameter, not a re-implementation.

Search is a single-layer greedy beam from the fixed navigating node with
search list size `L_search ≥ k`.

### Why these specific knobs

- **`r`** (out-degree cap) — primary recall/QPS knob. 20-48 covers the
  useful range.
- **`l_build`** — candidate pool size. Must be ≥ `r`; ≥ 2·r recommended.
- **`k_knn`** — k for the NN-Descent init graph. Larger = better init at
  cost of construction time.
- **`alpha`** — α-relaxation. `1.0` is strict, `1.2`-`1.5` is robust on
  clustered or low-intrinsic-dim data.
- **`seed`** — determinism: `(seed, data)` uniquely fixes the build.

## Consequences

### Positive

- **First single-layer graph index in ruvector**, complementing HNSW
  (multi-layer) and DiskANN (disk-resident). Smallest in-memory graph
  for a given recall budget.
- **Real numbers on M4 Max (Apple Silicon, single thread)** — see
  `docs/research/nightly/2026-05-13-nsg-mrng/README.md` for the full
  table.  Headline: **NSG/medium at N=50k, D=32 hits recall@10 = 0.965
  at 8,012 QPS — a 2.64× speedup over brute force at 96.5% recall**, and
  NSG/small reaches **6.58× speedup at 73% recall**.
- **No `unsafe`, no BLAS/LAPACK, no C deps.**
- Deterministic single-threaded build.
- Composable: the `NsgIndex` is self-contained (vectors + adjacency +
  one nav-node id) and can be quantised, persisted, or wrapped in a
  rerank stage independently.

### Negative

- **Batch-only build.** Adding a vector requires re-construction (or
  Vamana-style two-pass insert, which we have not implemented). NSG is
  intended for static / periodically-rebuilt corpora.
- **MRNG can over-prune on tight clusters.** Our remedy is the α
  relaxation; users who want strict NSG semantics can set `alpha = 1.0`
  and accept lower recall on clustered data.
- **Sequential build only.** A parallel rayon path is feasible (the
  per-node MRNG selection is embarrassingly parallel) but not yet wired
  in; this is the obvious next iteration.

### Alternatives considered

- **SSG (Spreading-out Subgraph)** — Fu et al. 2019, a refinement that
  drops the requirement on the navigating node and uses angle-based
  pruning. Strictly more complex; deferred.
- **ParlayANN-style parallel build.** Useful but orthogonal — we can
  layer it on later.
- **Pure HNSW second layer.** Doesn't solve the "lowest memory" goal.
- **Vamana-only.** Already have it in `ruvector-diskann`; NSG provides
  a different point on the design space (in-memory, single-layer,
  monotonic search path).

## Implementation notes

The crate is ~800 lines of Rust across four files (`lib.rs`, `knn.rs`,
`nsg.rs`, `error.rs`), each under 500 lines per the workspace rule.

```
crates/ruvector-nsg/
├── Cargo.toml
├── src/
│   ├── lib.rs        # public API, l2_sq, brute-force baseline, recall
│   ├── error.rs      # NsgError + Result
│   ├── knn.rs        # NN-Descent kNN graph + nearest-to-centroid walk
│   ├── nsg.rs        # NsgBuilder, NsgIndex, MRNG select, DFS augmentation
│   └── main.rs       # nsg-demo binary with end-to-end benchmark
└── benches/
    └── nsg_bench.rs  # criterion: search vs brute-force
```

Acceptance test (in `nsg::tests::build_small_and_search`): 500 points,
16 dim, 8 overlapping clusters, params `r=24, l_build=80, k_knn=40,
alpha=1.2` → recall@10 > 0.80 with avg out-degree > 1.0. **Passing.**

[nsg]: https://arxiv.org/abs/1707.00143
