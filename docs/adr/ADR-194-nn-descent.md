---
adr: 194
title: "NN-Descent — Approximate k-NN Graph Construction in ruvector"
status: accepted
date: 2026-05-27
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193]
tags: [knn-graph, ann, nn-descent, graph-build, nightly-research]
---

# ADR-194 — NN-Descent: Approximate k-NN Graph Construction in ruvector

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-27-nn-descent-graph-build` as
`crates/ruvector-nndescent`. `cargo build --release -p ruvector-nndescent`
succeeds; `cargo test --release -p ruvector-nndescent` passes 7/7 unit
tests; `cargo run --release -p ruvector-nndescent --bin nndescent-demo`
produces the real numbers cited in the research README.

## Context

ruvector's graph-based ANN family (HNSW in `ruvector-core`, DiskANN in
`ruvector-diskann`, NSG/MRNG, RoarGraph, SymphonyQG) and several
quantization indexes (RaBitQ, LeanVec) either consume or seed from an
existing k-NN graph. Today there are exactly two ways to obtain that
graph inside the workspace:

1. **Brute force**: O(N²·D) distance calls. Fine up to N≈10⁴; intractable
   at production corpus sizes.
2. **Side-effect of HNSW insertion**: the search-quality graph, not a
   pure k-NN graph; expensive to recompute for offline batch use.

Every competitor (FAISS, Milvus 2.4, Weaviate, LanceDB, CAGRA's CPU path)
ships NN-Descent (Dong, Charikar, Li 2011) as the canonical batch builder
because it converges in O(N·k·iters) distance calls with iters typically
< 10, and because the local-join structure parallelises trivially.

ruvector lacked it. Adding it unlocks faster builds for at least six
existing crates and matches the SOTA toolkit competitors take for granted.

## Decision

Add `crates/ruvector-nndescent`, a single-purpose crate with a narrow
public API:

```rust
pub trait Metric { fn dist(&self, a: &[f32], b: &[f32]) -> f32; }
pub trait KnnGraphBuilder { fn build(&mut self, data: &[Vec<f32>], k: usize) -> BuildReport; }
pub struct BruteForce<M: Metric>;   // baseline + ground-truth helper
pub struct NnDescent<M: Metric>;    // the algorithm
```

Algorithm follows the 2011 paper with the three production refinements
that landed in PyNNDescent / CAGRA: `rho` sampling, reverse-neighbour
lists, and per-iteration `is_new` flags on a bounded max-heap.

The crate is **sequential** in this first cut. Parallelism via `rayon`
is intentionally deferred to keep the PoC small and reviewable, and to
let us land an honest single-thread baseline before any "X× faster"
claim that bundles in core scaling.

## Consequences

**Positive.**
- ruvector now has a SOTA batch builder for k-NN graphs (measured: 1.7×
  wall-clock and 3× distance-call reduction vs. brute at N=5,000, 64-D;
  scaling is sub-quadratic, so the gap widens with N).
- Six downstream crates (`ruvector-diskann`, `ruvector-roargraph`,
  `ruvector-leanvec`, `ruvector-nsg`-equivalents, `ruvector-lvq`,
  `ruvector-cluster`) can be wired to use it as a seed step.
- Closes a competitive gap vs. Milvus 2.4 / Weaviate / FAISS.

**Negative.**
- Below ~N=1,500 brute is faster *and* exact; callers must branch on N.
- Vanilla NN-Descent (no reverse lists) collapses to 0.55 recall at
  N=5,000 — reverse lists are not optional in production.
- `rho < 1.0` did not save calls on isotropic Gaussian data here; the
  knob is documented as dataset-dependent, not a free speed-up.
- One more workspace member to keep building green.

**Risk.** Recall on real-world corpora (SIFT-1M, GIST-1M, DEEP-10M) is
not yet measured. The synthetic Gaussian numbers are a smoke test, not
a production claim.

## Alternatives considered

- **HNSW-graph-as-k-NN-graph.** What we do today. Couples graph quality
  to a *search* objective rather than a *construction* objective; also
  ~3× slower to build than NN-Descent on equivalent k.
- **DEG (Hezel et al. 2024).** Dynamic exploration graph supports
  online inserts elegantly, but build cost matches NN-Descent only with
  significantly more code and an explicit deletion model. Out of scope
  for a one-night PoC; tracked as a follow-up.
- **GPU-only via CAGRA.** Would force ruvector to take a CUDA dep. CAGRA's
  CPU build path is literally NN-Descent; we now have that path natively.
- **Numba/Python wrapper around PyNNDescent.** Violates the Rust-only
  constraint and adds a Python runtime to the deploy surface.

## Verification

```sh
cargo build --release -p ruvector-nndescent          # green
cargo test  --release -p ruvector-nndescent          # 7/7 passing
cargo run   --release -p ruvector-nndescent --bin nndescent-demo
# Real numbers reproduced in docs/research/nightly/2026-05-27-nn-descent-graph-build/README.md
```

## References

- Dong, Charikar, Li — *Efficient k-NN Graph Construction for Generic
  Similarity Measures*, WWW 2011.
- Ootomo et al. — *CAGRA: Highly Parallel Graph Construction and ANN
  Search for GPUs*, arXiv:2308.15136 (2024).
- PyNNDescent — https://github.com/lmcinnes/pynndescent
- Research note: `docs/research/nightly/2026-05-27-nn-descent-graph-build/README.md`
