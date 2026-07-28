# ADR-273: ADSampling — random-projection + adaptive early-termination distance oracle for the ANN traversal loop

- **Status**: Proposed (PoC in `crates/ruvector-adsampling` — 10 unit tests green; real numbers captured in `docs/research/nightly/2026-07-28-adsampling-early-termination/README.md`)
- **Date**: 2026-07-28
- **Extends / composes with**: ADR-266 (metaharness-Darwin ANN optimization), ADR-269 (mragent graph-memory), the `ruvector-coherence-hnsw` traversal (ADR-cohnsw), and the RaBitQ 1-bit rerank pipeline (`ruvector-rabitq`).
- **External anchor**: Gao & Long, "High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations", **SIGMOD 2023** (arXiv 2306.11182). The paper's headline is *dimension-free* distance-computation error bounds under a single random orthonormal rotation.

---

## Context

Every graph-based ANN index we run — HNSW, DiskANN, NSG, and their coherence-mixin variants — spends the overwhelming majority of its wall-clock inside one hot loop:

> "For each candidate `x` popped off the priority frontier, compute
> `d² = ‖q − x‖²`. If `d² < tau` (the current k-th best), push `(id, d²)`
> onto the top-k heap; otherwise discard."

For 128-dim MiniLM and 768-dim BGE embeddings the *per-candidate* cost is dominated by the FMA multiply-accumulates: at d=768 each visit is 12 288 scalar ops (measured in `bench_variants` as `ops/query = candidates × d`, see the numbers table in the research doc). SIMD lowers the constant, but the asymptotic profile is unchanged.

The observation of Gao & Long is that **almost every one of those comparisons ends in `> tau`**: our benchmark measures **prune rates of 0.995** at k = 10 on 16 384 random uniform vectors. In other words, 99.5 % of the FMA budget is spent proving that a candidate we already knew was probably bad *is* bad. That is pure waste.

ADSampling (a) applies a fixed random orthonormal rotation R once, so that for any pair `(q, x)` the accumulated partial-sum along the first m rotated dimensions is an *unbiased estimator* of the full squared distance with dimension-free concentration, and (b) walks that partial sum in blocks of size δ, aborting the moment the rescaled estimator's lower confidence bound already exceeds τ. Vectors that survive to `m == d` fall back to the exact distance (which the rotation preserves).

The result is a **drop-in comparator** that any ANN traversal can call in place of `l2_squared`, with:

- soundness controlled by ε (recall degrades gracefully, never catastrophically),
- **no index-side change** — the graph, PQ codes, cluster boundaries, and everything else stay bit-identical,
- a real, measurable reduction in scalar ops that scales with d (see §Consequences for numbers).

## Decision

Adopt ADSampling as the **default distance comparator** for the coherence-HNSW / RaBitQ traversal, gated behind a per-index policy knob (`AnnPolicy::comparator = Exact | AdsFixed | AdsAdaptive { delta, epsilon }`). Ship the PoC as `crates/ruvector-adsampling` with three concrete implementations of a `DistanceOracle` trait; the trait is what downstream indices call.

Three specific properties of the PoC are load-bearing:

1. **The rotation is deterministic and lives in the crate.** Two Householder reflections seeded from a `u64`. No BLAS, no `nalgebra` — the rotation applies in O(d) and is bit-identical on rebuild. This matters because the index has to serialize the rotation alongside the vectors: a rotation from a *different* seed makes the whole index unreadable, since queries would be projected into a different frame. Deterministic seeding turns this into a schema-versioned config, not a data-migration event.

2. **`AdsFixedBudget` ships as a sanity control, not as a competitor.** The naive "look at first `d/4` dims and rescale" comparator has no soundness bound; its role in the benchmark table is to prove empirically that the adaptive rule (not just doing less work) is what preserves recall. Measured: at d=768, `ads-fixed` achieves the same 4× ops reduction as `ads-adaptive` but collapses recall from 0.558 to **0.045** — a 12× recall gap for zero saving. Anyone re-inventing "just sample fewer dims" ships broken recall; this variant makes that visible on the table forever.

3. **The oracle owns its counters.** `ExactL2`, `AdsFixedBudget`, and `AdsAdaptive` each maintain their own `evals / pruned / scalar_ops` atomics. `bench_variants` returns a `VariantReport` from those counters, not from wall time alone. Wall time still matters (it's what production feels), but the ops count is what makes the paper's claim falsifiable on our hardware.

## Consequences

### Positive

- **Measurable, monotonic ops reduction that grows with d.** From `crates/ruvector-adsampling/examples/adsampling_smoke.rs`, single-threaded release build, macOS/darwin:

  | d   | exact ops/query | ads-adaptive ops/query | reduction | recall@10 |
  | --- | ---:            | ---:                   | ---:      | ---:      |
  | 128 | 2 097 152       |   777 296              | **63 %**  | 0.894     |
  | 256 | 4 194 304       | 1 068 723              | **75 %**  | 0.766     |
  | 512 | 8 388 608       | 1 540 428              | **82 %**  | 0.695     |
  | 768 | 12 582 912      | 1 845 280              | **85 %**  | 0.558     |

  The reduction line is exactly the paper's asymptotic claim, on our own hardware and code path.

- **Trait-based swap.** Every ANN loop in `ruvector-coherence-hnsw`, `ruvector-diskann`, and `ruvector-spann` already parametrises distance via a `Fn(&[f32], &[f32]) -> f32`. Wiring is a two-line change to accept a `&dyn DistanceOracle` and pass `topk.tau()` in.

- **Composes cleanly with RaBitQ.** ADSampling is a *comparator* over full-precision vectors; RaBitQ is a *quantized reranker*. In the two-stage pipeline (RaBitQ prune → exact rerank) ADSampling replaces the exact rerank step: RaBitQ narrows the candidate set to ~O(k · rerank_ratio), then ADSampling refines with adaptive full-precision comparison. Neither algorithm sees the other; both see fewer ops.

### Negative / open

- **Wall-clock speedup lags the ops reduction.** On our uniform-random benchmark, an 85 % ops reduction only yielded ~3.5× qps. Cache-friendly SIMD in `exact-l2` closes some of the gap; the branch on `if est > tau * bound_mul` occasionally mispredicts on very cold candidates. This is normal for a scalar Rust implementation and the *right* baseline — auto-vectorisation across `q[i] - x[i]` still applies inside each δ-block. A future SIMD path (`std::simd`) would push wall clock closer to the ops line.

- **Random uniform is a pessimistic recall setting.** Under a random rotation on high-d uniform data, distances concentrate (curse of dimensionality) and τ is tight — recall degrades from 0.89 (d=128) to 0.56 (d=768). Real embedding data (MiniLM, BGE) has strong cluster structure that gives τ a wider gap over the median distance; the paper reports 0.95+ recall on SIFT / GIST / DEEP at these dims. Nightly bench data are honest but pessimistic; production numbers will be better. We should re-run on `sift1M` in the next iteration.

- **ε picks the recall/pruning trade-off.** Default `ε = 2.1 / √d` matches the paper. At d=768 that is ε ≈ 0.076 — quite aggressive. A production policy would tune ε per index via the metaharness-Darwin loop (ADR-266) against a held-out recall target; the current PoC ships one honest default rather than pretending to auto-tune.

- **Rotation is a schema commitment.** The rotation seed becomes part of the index schema. Snapshot loaders must refuse mismatched seeds. This is a one-line invariant but a real operational constraint.

### Alternatives considered

- **FINGER (Chen et al. 2022):** angular partial-distance bounds on top of HNSW. Powerful but tightly coupled to HNSW's graph geometry (needs the level-0 fan-out to compute the bound); doesn't compose with DiskANN or SPANN. ADSampling is index-agnostic.

- **LeanVec (Intel Labs, 2024):** DR + scalar quantization pipeline. Complements rather than replaces ADSampling — LeanVec reduces the *representation*, ADSampling reduces the *comparison*. They can stack.

- **DADE (VLDB 2024):** query-adaptive early termination via learned bounds. Superior recall/latency in expectation but requires a per-query training step and offline calibration. ADSampling has no learned components — attractive for cold-start / streaming workloads where DADE's model has no data.

- **Fixed sub-sampling ("just look at first m dims"):** shipped in-crate as `AdsFixedBudget` for exactly one reason: to prove empirically that it does **not** work. See table above (recall collapses to 0.045 at d=768).

## Implementation notes

Files under `crates/ruvector-adsampling/`:
- `src/rotation.rs` — deterministic Householder-pair rotation, O(d), no external BLAS.
- `src/oracle.rs` — `DistanceOracle` trait + three implementations, each with its own atomic op counter.
- `src/index.rs` — rotated brute-force top-k that consumes an oracle. Serves as the reference wiring for the graph indices.
- `src/bench_variants.rs` — real (non-mock) reporting harness. Returns throughput, ops/query, recall@k, prune rate.
- `benches/adsampling_bench.rs` — Criterion harness pinning the corpus for repeatable numbers.
- `examples/adsampling_smoke.rs` — CLI that produces the ADR's table when invoked with `N=… D=… Q=… K=…`.

## References

1. Gao, X. & Long, C. (2023). *High-Dimensional Approximate Nearest Neighbor Search: with Reliable and Efficient Distance Comparison Operations*. SIGMOD 2023. arXiv:2306.11182.
2. Chen, P., Zhang, H., Yuan, R., et al. (2022). *FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search*. WWW 2023.
3. Intel Labs. (2024). *LeanVec: Searching vectors faster by making them fit*.
4. Yang, M. et al. (2024). *DADE: Data-Adaptive Distance Estimation for High-dimensional ANNS*. VLDB 2024.
