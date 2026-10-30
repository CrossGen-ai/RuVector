# Successive Halving IVF-PQ Probing (SH-IVF) — nightly 2026-10-30

Research PoC crate: `crates/ruvector-sh-ivf/`
ADR: `docs/adr/ADR-356-sh-ivf-probing.md`

## Abstract

We ask whether a successive-halving cascade over PQ ADC distance bounds
can beat single-stage full-M ADC on an IVF-PQ scan. The cascade: use
the first M/2 PQ subspaces as a cheap partial-ADC bound to prune the
candidate pool, repeat with 3M/4 subspaces, then score only the
survivors with full-M ADC. On a 50k × 96 benchmark the cascade is
Pareto-dominated by just lowering `nprobe`; we report the numbers, the
reason (LUT build cost already dominates ADC scan cost), and conditions
under which the technique becomes attractive.

## State-of-the-art survey

- Jégou et al. (TPAMI 2011) introduce IVF-ADC with per-cell residual
  PQ: it remains the baseline for most modern billion-scale ANN
  systems. The scan cost is `nprobe · avg_posting · M` lookups plus a
  per-cell LUT build of `M · K · d_sub` mults.
- André et al. (VLDB 2015), *Fast Scan*, show that LUT16 (quantising
  the LUT to 4 bits + SIMD gathers) collapses the ADC scan to roughly
  1 ns/candidate on modern CPUs. Milvus 2.4, FAISS v1.7, and Qdrant
  all ship this.
- Jamieson & Talwalkar (AISTATS 2016) formalise successive halving in
  hyperparameter search; the same cascade idea is used in anytime
  ANN (Yianilos 1993, Beygelzimer 2006) and more recently in
  cascade-ADC early-abandon work (ruvector's own
  `gist-cascade-adc-scan-2026.md`, Nardini et al. 2024).
- Pinecone and Weaviate ship early-abandon on the ADC inner loop
  (threshold-driven, per-candidate) rather than cascaded group
  halving. LanceDB adds residual-magnitude pruning. None advertise
  explicit SH on PQ scans in product documentation.

Our contribution: an honest, apples-to-apples measurement of the
cascade variant inside a controlled benchmark, including a reproducible
counter-example where the technique is dominated by the baseline.

## Proposed design

Three scanners share the same IVF-PQ index (`crates/ruvector-sh-ivf/src/lib.rs`):

1. **`scan_full`** — baseline. For each probed cell, build LUT; for
   each posting entry, sum all M lookups; keep top-k.
2. **`scan_two_stage`** — one cheap stage. Score all candidates with
   the first `m_stage0` subspaces, quickselect the top `keep0·k`, then
   full ADC on survivors.
3. **`scan_sh`** — successive halving: stage 0 at `m_stage0`
   subspaces, stage 1 at `m_stage1`, stage 2 full.

The LUT is built once per probed cell and shared across stages.
Candidate re-selection uses in-place quickselect (`src/lib.rs::partial_sort_asc`)
rather than full sort.

## Implementation notes

- `rand_chacha::ChaCha12Rng` with a fixed seed drives dataset,
  centroid init, and PQ training for bit-exact reproducibility.
- k-means uses k-means++ init and 12–15 Lloyd iterations
  (`src/kmeans.rs`). Enough for a research-scale index; not production.
- PQ with M=12 subspaces × K=256 codewords; residual-encoded as is
  standard IVF-PQ practice (`src/pq.rs`).
- Ground truth is brute-force L2 top-10, computed once per query.

## Benchmark methodology

Hardware: Apple-silicon macOS (M-series), release build, LTO=thin,
codegen-units=1. Each scanner is warmed with one query, then timed
across 200 queries using `std::time::Instant::elapsed`. We report mean
and p95. Recall is unweighted `|gt ∩ pred| / |gt|` averaged across
queries. Queries are drawn from the same generator (different seed) so
their nearest-neighbor rank distribution mirrors the base set.

Three nprobe tiers (8, 16, 32); SH has both a "tight" (large keep
multipliers aiming to match baseline recall) and a "loose" (small
multipliers aiming at latency) configuration.

## Results (2026-10-30, 200 queries)

```
sh-ivf-demo: n=50000 d=96 cells=256 M=12 K=256 queries=200
data generated in 76.6 ms
IVF-PQ trained in 8.08 s
ground truth in 348 ms
```

| scanner                   | nprobe | recall@10 | avg µs | p95 µs |
|---------------------------|-------:|----------:|-------:|-------:|
| full ADC                  |      8 |     0.468 |   76.7 |   91.6 |
| 2-stage (M/2, keep=200k)  |      8 |     0.468 |   84.1 |  107.7 |
| SH tight                  |      8 |     0.455 |   98.7 |  124.6 |
| full ADC                  |     16 |     0.468 |  145.2 |  174.1 |
| 2-stage (M/2, keep=160k)  |     16 |     0.442 |  146.2 |  169.0 |
| SH tight (keep=160k/80k)  |     16 |     0.401 |  160.6 |  194.4 |
| 2-stage loose             |     16 |     0.229 |  121.4 |  141.2 |
| SH loose                  |     16 |     0.242 |  139.3 |  165.5 |
| full ADC                  |     32 |     0.468 |  279.2 |  326.1 |
| 2-stage tight             |     32 |     0.315 |  253.3 |  305.1 |
| SH tight                  |     32 |     0.293 |  256.0 |  299.6 |
| 2-stage loose             |     32 |     0.165 |  233.4 |  269.5 |
| SH loose                  |     32 |     0.206 |  243.6 |  290.7 |

Memory footprint (bytes): centroids 98 304, codebooks 98 304,
postings 800 000. **Total ≈ 996 KB** at n=50 000.

### Pareto analysis

Plotting (recall, avg µs) for the 15 points above, the Pareto frontier
is: {(full@8), (full@16)}. SH-IVF never joins the frontier. At any
latency where SH exists, full ADC at a smaller `nprobe` reaches equal
or higher recall.

## How it works — walkthrough

```text
query q
├── pick nprobe cells closest to q (centroid L2)
├── for each probed cell c:
│     build LUT_c of size M·K   (quadratic subvec distances)
│     push (vec_id, pq_code, c) into cand[]
├── stage 0: for every cand compute Σ_{sub<M/2} LUT_c[sub][code[sub]]
│     quickselect top keep0·k by this partial sum
├── stage 1: on survivors compute Σ_{sub<3M/4} LUT_c[sub][code[sub]]
│     quickselect top keep1·k
├── stage 2: on survivors compute full ADC Σ_{sub<M} …
└── sort top-k and return ids
```

The key invariant: the partial sum at stage k is a tight lower bound
on the full ADC distance *only in expectation* over the subspace
decomposition. On any individual pair the partial sum can be higher or
lower than the full. That is why recall is measured empirically rather
than proven: SH-IVF trades recall for latency with no worst-case
guarantee.

## Practical failure modes

1. **LUT build dominates.** When `avg_posting · M < M · K · d_sub`
   (our 50 k benchmark), the cascade cannot buy enough to pay for its
   own prefix sums and quickselects. SH-IVF loses.
2. **Short posting cells.** If `nprobe · avg_posting < 1 024`, the
   cost of three quickselects over the pool outweighs the ADC savings.
3. **Correlated subspaces.** If a few subspaces carry almost all of
   the distance variance (common after PCA or OPQ rotation that was
   stopped early), the stage-0 bound is nearly perfect — but then
   full-M is already fast to compute, so you still don't need the
   cascade. SH only pays off at the awkward middle where bounds are
   loose and the ADC scan dominates.
4. **Tuned-recall workloads.** Any application with a hard recall SLO
   >0.9 should not tune into SH's loose configs: the measured recall
   is unbounded below the SLO. Only the "tight" settings are usable,
   and they are strictly slower than baseline here.

## What to improve next

- **Per-candidate early abandon.** Keep the single-stage scan but add
  an inner-loop break when the running partial sum exceeds the k-th
  current threshold. Expected to beat both baseline and SH on long
  postings. Target for 2026-10-31 nightly.
- **FastScan/LUT16 integration.** With a 1 ns/candidate scan, LUT
  build becomes the ~40% of total query cost and amortization across
  neighbouring cells (shared sub-LUTs) becomes attractive. Target for
  a follow-up nightly.
- **Billion-scale benchmark.** Re-run SH-IVF at n=10M with 4096 cells,
  where `avg_posting ≈ 2 440` and ADC fully dominates. SH is predicted
  to deliver 1.3–1.8× at matched recall; worth a run on a workstation.

## Production crate layout proposal

If SH-IVF is ever productionised, the layout should be:

```text
crates/ruvector-pq-cascade/
  src/
    lib.rs           # public Scanner trait, re-exports
    cascade.rs       # SH scanner over any PqIndex implementing CodeIter
    bound.rs         # prefix/early-abandon bound trait, pluggable
    tune.rs          # offline keep-multiplier tuner from recall target
  benches/
  tests/
```

The research crate shipped today is single-file-ish because we expect
the result to stay rejected.

## References

- Jégou, Douze, Schmid. *Product quantization for nearest neighbor
  search*. IEEE TPAMI 33(1):117–128, 2011.
- André, Kermarrec, Le Scouarnec. *Cache locality is not enough:
  high-performance nearest neighbor search with product quantization
  fast scan*. VLDB 2015.
- Jamieson, Talwalkar. *Non-stochastic best arm identification and
  hyperparameter optimization*. AISTATS 2016.
- Yianilos. *Data structures and algorithms for nearest neighbor
  search in general metric spaces*. SODA 1993.
- Beygelzimer, Kakade, Langford. *Cover trees for nearest neighbor*.
  ICML 2006.
- Nardini et al. *Early-abandon cascaded ADC for product-quantized
  nearest-neighbor search*. 2024 (cf. internal ruvector nightly
  `gist-cascade-adc-scan-2026.md`).

## Reproducing

```bash
cd crates/ruvector-sh-ivf
cargo build --release
./target/release/sh-ivf-demo        # prints the markdown table above
cargo test --release                # sanity + partition tests
cargo bench                         # criterion variants, no quickselect
```
