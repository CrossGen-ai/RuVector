---
adr: 195
title: "SPANN-Style Boundary-Aware Closure for IVF Posting Lists"
status: accepted
date: 2026-05-28
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ivf, ann, vector-search, spann, posting-list, recall, nightly-research]
---

# ADR-195 — SPANN Boundary Closure: Replicate Only the Vectors That Matter

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-28-spann-boundary-closure` as
`crates/ruvector-spann`. All 7 unit tests pass; `cargo build --release
-p ruvector-spann` is green; `cargo run --release -p ruvector-spann
--bin spann-demo` produces the numbers in this document.

## Context

ruvector's first IVF family landed in ADR-193 (`ruvector-rairs`),
which adds a *fixed* secondary-list assignment to every base vector
(redundant assignment + residual-amplified scoring). This raises recall
at low `nprobe` but pays a uniform 2× index-size cost regardless of
whether a given point actually benefits from spill.

The deeper problem is that **most points do not need to be replicated**.
A vector deep inside its Voronoi cell is reached by exactly one probe
list and never benefits from being in others. The points that hurt IVF
recall are the ones near a cell *boundary* — they are roughly equidistant
to two or more centroids, and a single-assign IVF often files them under
the "wrong" centroid relative to a nearby query.

SPANN (Chen et al., NeurIPS 2021) introduced a now-standard solution:
**closure assignment**. Each base vector is assigned to every centroid
whose distance is within a multiplicative threshold of the nearest
centroid, bounded by a hard cap. Interior points pay no replication;
boundary points get extra coverage *exactly where it raises recall*.

This is qualitatively different from RAIRS's fixed redundant assignment
and from naive multi-probe with `k=2` or `k=4`. It is closer in spirit
to SOAR's anti-correlated spillover (ADR-related work) but cheaper —
no orthogonality term, no second pass.

## Decision

Add `crates/ruvector-spann`, a self-contained IVF implementation built
around a `ClosurePolicy` trait so the assignment strategy is the *only*
moving part. Three policies ship in v0.1:

| Policy             | Rule                                                                            |
|--------------------|---------------------------------------------------------------------------------|
| `SingleAssign`     | Classic IVF — vector → 1 nearest centroid                                       |
| `FixedMultiAssign` | Multi-probe — vector → top-`k` nearest centroids unconditionally                |
| `SpannClosure`     | SPANN — vector → all centroids with `dist ≤ (1+ε)·dist_min`, capped at `cap`    |

Search is unchanged across policies (find `nprobe` nearest centroids,
linearly scan their posting lists, dedupe by id). Only the build-time
posting-list contents change — which makes apples-to-apples comparison
trivial and lets us isolate the *closure* effect from any other
implementation difference.

`SpannIndex::build` is generic over `P: ClosurePolicy + ?Sized` so
trait objects work in benchmarks without monomorphising five copies.

## Consequences

### Positive

- **Pareto improvement at low `nprobe`.** On a 20k×64 clustered Gaussian
  mixture (K=128 centroids), `spann(ε=0.10, cap=4)` reaches
  recall@10 = 0.910 at `nprobe=2` for **12.8 µs / query** —
  *faster* than baseline single-assign (14.2 µs at 0.846 recall) and
  *cheaper* than `fixed-multi(k=2)` (20.1 µs at 0.916 recall). The
  closure spends replication only where it pays off.
- **Replication factor < 2×.** SPANN(ε=0.10, cap=4) replicates at
  ~1.51× vs `fixed-multi(k=2)`'s 2.00× and `fixed-multi(k=4)`'s 4.00×.
  Memory overhead vs baseline is just **+0.8%** of total index bytes
  (the data vectors dominate; posting-list ids are `u32`).
- **Composable.** Because the trait abstracts the only thing that
  varies, future policies (anisotropic closure, learned closure,
  SOAR-style residual closure) drop in without touching search.
- **Auditable.** Single crate, no `unsafe`, no external linear-algebra
  dep, all files ≤ ~250 lines.

### Negative / Trade-offs

- **Build cost is the same shape as IVF + a per-vector sort over `K`
  centroids.** That sort is the dominant build cost at large `K`.
  For now this is acceptable (565 ms to build the 20k×64 benchmark);
  HNSW-on-centroids would amortise it for larger `K`.
- **Replication is bounded by `cap`, not by recall target.** A future
  iteration could pick `cap` adaptively from a held-out recall curve.
- **No PQ/RaBitQ compression of posting-list bodies yet.** This crate
  stores raw vectors; the production layout (see roadmap) interleaves
  compressed codes per posting block.

### Alternatives Considered

1. **Fixed multi-assignment (`k=2`, `k=4`).** Simple, but uniformly
   wasteful — measured overhead is 2-4× replication for recall gains
   that closure achieves at 1.51×. Kept as a baseline policy in the
   same crate.
2. **SOAR (anti-correlated spillover).** Closer to RAIRS's design.
   Requires an orthogonality term during build and a residual at
   search time, both of which add code surface. Targeted for a future
   `SoarClosure` policy under the same trait.
3. **Extending `ruvector-rairs` in place.** Rejected — RAIRS's
   redundant-list contract is a *fixed* 2× spill with a residual-amp
   reranker; bolting an optional closure rule onto it would muddle the
   ADR-193 semantics. A clean new crate keeps both indices honest.

## Implementation Notes

- `kmeans.rs` implements k-means++ seeding + Lloyd updates with
  dead-centroid reseeding. Deterministic via `StdRng::seed_from_u64`.
- `policy.rs` defines the `ClosurePolicy` trait and the three concrete
  policies. Five unit tests pin down the assignment math (single,
  multi, boundary inclusion, interior exclusion, cap respect).
- `index.rs` does the posting-list build + a bitset-deduped top-k
  search with a `BinaryHeap<HeapItem>` of capacity `top_k`.
- `src/main.rs` is the runnable benchmark binary (`spann-demo`)
  emitting the table in the research doc.
- `benches/spann_bench.rs` adds Criterion micro-benchmarks for the
  three search paths.

## Reproducibility

```bash
cargo build --release -p ruvector-spann
cargo test  --release -p ruvector-spann
cargo run   --release -p ruvector-spann --bin spann-demo
cargo bench -p ruvector-spann
```

Numbers in this ADR were collected on the host listed in the research
document. The dataset is synthetic clustered Gaussian (64-mode mixture,
σ=1.0); CLAUDE.md forbids checking in datasets, so the benchmark
generates its own with fixed seeds.
