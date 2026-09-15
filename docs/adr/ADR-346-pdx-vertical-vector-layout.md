# ADR-346: PDX Dimension-Partitioned Vector Layout for Prunable Scans

## Status

Experimental (accepted for prototype, not for production). The crate
`ruvector-pdx-scan` is added as a standalone, detached-workspace nightly
experiment. Integration into `ruvector-cluster` / `ruvector-pq-search` is
deferred until (a) `core::simd` masked stores stabilize on stable Rust and
(b) IVF numbers close the gap with the horizontal baseline on clustered
(non-Gaussian) data.

## Context

Ruvector's brute-force and IVF-list distance kernels use row-major
horizontal storage: each vector's `d` `f32` dimensions are contiguous, and
the L2 kernel iterates over `d` inside a per-vector reduction. This is
what every Rust ANN crate on crates.io does today.

Kuffo, Krommydas, and Boncz (SIGMOD 2024, "PDX: A Data Layout for Vector
Similarity Search") propose the "vertical" alternative: group vectors into
blocks of `B` and, within a block, store dimension `j` for all `B` vectors
contiguously — a `d × B` column-major tile. Distance kernels then keep `B`
per-lane accumulators; SIMD units process `B` candidates in lockstep with
no cross-lane reductions in the hot loop. Optionally, after every
`dim_step` dimensions, partial per-lane distances are compared to the
current top-k threshold; lanes exceeding it are pruned.

The question this ADR answers: **does the PDX layout reproduce the paper's
2–4× brute-force speedups in portable Rust without hand-rolled AVX-512,
and is the pruning benefit real on Ruvector-shaped IVF workloads?**

## Hypothesis

```text
Given n=50 000 IID Gaussian vectors at d=128, k=10, 100 queries, the PDX
column-block layout scanned by portable Rust (opt-level=3, lto=thin,
codegen-units=1, no explicit intrinsics) will:

  H1: exceed the horizontal baseline by at least 2× on plain (unpruned)
      brute-force with recall 1.000.
  H2: exceed the horizontal baseline by at least 1.5× on branchless
      block-level pruned brute-force with recall 1.000.
  H3: NOT exceed the horizontal baseline on per-lane pruned brute-force
      (predicting that `if active[lane]` defeats autovectorization).
  H4: on IVF (64 lists, nprobe in {4,8,16}) match horizontal recall at
      any given nprobe and beat horizontal wall-clock on at least one
      nprobe setting.
```

## Decision

Land the `crates/ruvector-pdx-scan` crate as a detached-workspace nightly
experiment with:

* `HorizontalStore` and `PdxStore` behind a shared `VectorStore` trait.
* Four brute-force scans (`brute_horizontal`, `brute_pdx`,
  `brute_pdx_pruned`, `brute_pdx_block_pruned`).
* A k-means IVF driver stacked on either backend
  (`IvfHorizontal`, `IvfPdxPruned`).
* A real-numbers benchmark binary (`pdx-bench`) and a real unit test
  suite (7 tests, all passing) that includes an "IVF-PDX must equal
  IVF-horizontal at nprobe=all" recall equivalence test.

Do NOT wire `PdxStore` into `ruvector-cluster` or `ruvector-pq-search`
in this iteration.

## Measured outcome

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| H1 (plain PDX ≥ 2×) | **Confirmed** | 2.94× (0.553 ms vs 1.624 ms/query) |
| H2 (block-pruned ≥ 1.5×) | **Confirmed** | 1.86× (0.874 ms/query) |
| H3 (per-lane pruned < 1×) | **Confirmed** (as predicted, a *loss*) | 0.34× (4.722 ms/query) |
| H4 (IVF beats horizontal at some nprobe) | **Rejected** | 0.37–0.40× across nprobe∈{4,8,16}, recall matched |

The layout hypothesis holds strongly; the pruning hypothesis holds only in
the branchless block form and does not survive the IVF nprobe study on
Gaussian data.

## Consequences

**Positive.**
* Ruvector gains a working PDX prototype, sole such in the Rust ecosystem
  as of 2026-09.
* 2.94× brute-force speedup is available today for any workload that can
  tolerate a per-list rebuild cost — flat, small, read-heavy indexes.
* Establishes a baseline against which `core::simd` masked-store variants
  can be measured when they stabilize.
* Provides a companion layout for future PQ+PDX fusion (compresses the
  scan and compounds SIMD gains).

**Negative.**
* Per-lane pruning is a **net loss** in portable Rust. This is not a bug;
  it is a real limit of autovectorization plus lane-predicate branches.
  Anyone reading only the paper's headline numbers would be surprised.
* IVF pruning does not pay for itself on Gaussian data at the nprobe
  values Ruvector cares about. Real gains likely require clustered data
  where the top-k threshold tightens fast.
* PDX build cost (`O(n * d)` transpose) is real and forbids drop-in use
  in streaming/write-heavy indexes without amortization.

**Neutral / operational.**
* Detached workspace = no impact on `cargo build --workspace`. Removal /
  promotion is a single-directory operation.

## Alternatives

1. **Row-major + std::simd f32x8/f32x16.** Cheaper, no build cost, but
   caps out at ~1.3–1.5× on horizontal reductions and cannot express the
   PDX pruning idea.
2. **Row-major + ADSampling.** Achieves pruning inside horizontal
   storage via random projection; recall is probabilistic, and requires
   query-time projection cost. Complementary to, not competing with, PDX.
3. **Hand-rolled AVX-512 intrinsics.** Would match the paper's advertised
   pruned speedups. Rejected here for portability (Ruvector targets Apple
   Silicon, x86_64, and small ARM cores). Revisit under `cfg(target_feature)`.
4. **Do nothing.** Rejected: the layout question is worth a real answer,
   and the negative pruning result is itself valuable.
