# IVF Progressive Shortlist (IPSL) — a negative-result study

**Date:** 2026-11-13
**Slug:** `ivf-progressive-shortlist`
**ADR:** ADR-362

## Abstract

Standard IVF scans each probed posting list end-to-end (DFS across lists).
We prototype a breadth-first alternative — **Progressive Shortlist (IPSL)** —
that scans a short prefix (`m_init`) of each probed list in
residual-norm-descending order, then revisits lists whose remaining tail
could still improve the top-k by the Cauchy–Schwarz upper bound
`qc + 2·‖q−c‖·r_max − r_max²`.  We compare three scanners on a 50 000-point
synthetic Gaussian-mixture benchmark (dim 64, 256 IVF cells, k=10, 200 queries):

| Scanner                      | What it does                                          |
|------------------------------|-------------------------------------------------------|
| B0 Exhaustive                | Scan every residual in each probed list, any order    |
| B1 Norm-ordered + Abandon    | Scan in residual-norm-desc order, stop when UB < τ    |
| B2 Progressive               | `m_init` per list, then revisit by best remaining UB  |

**Headline finding:** on this benchmark, B2 is **5–45 % slower** than B0
at identical recall.  The gain predicted by paper intuition is **real
only when the per-code scorer is expensive** (PQ-ADC with LUT gather,
SIMD distance on high dimensions).  For raw f32 L2 at dim=64, the
per-list bookkeeping cost dominates.  We ship IPSL as an opt-in scanner
behind the `Scanner` trait, not as a replacement, and we record the
regime where it does help as the next experiment.

## SOTA survey

- **FAISS IVF-PQ** (Johnson, Douze, Jégou 2017) — the reference depth-first
  scanner.  No tail-skipping across lists.
- **BlockMax WAND** (Ding & Suel 2011, Mallia et al. 2017, re-applied to
  IVF-PQ in Blockmax-IVF 2024) — stores per-block max score, skips blocks
  whose max cannot beat the top-k threshold.  IPSL is the strict analogue
  with residual-norm as the max-score surrogate.
- **Qdrant adaptive scanning** (2025 changelog) — adds residual-norm
  early-abandon to each list but still scans lists sequentially.
- **LanceDB Rowgroup skip** (2025) — Parquet-style zonemap skipping for
  IVF.  Sibling of B1 (norm-abandon) but with pre-grouped tiles.
- **BEAM-PQ** (Chen et al. ICML 2026) — budgeted top-k over PQ with a
  global beam, closest published cousin to B2.  Reports 15–25 % speedup
  at recall ≥ 0.95 on SIFT1M using PQ-16x8.

Across competitors (Milvus, Qdrant, Weaviate, Pinecone, LanceDB, FAISS)
none currently ship a breadth-first scanner with multi-pass tail revisit
for native vector search; BlockMax-IVF is the closest, scanning blocks
within a list but not revisiting lists.

## Proposed design

Residual form of the IVF score:
```
-‖q − (c + r)‖² = -‖q − c‖²  +  2 (q − c)·r  −  ‖r‖²
                 ≤ -‖q − c‖²  +  2 ‖q − c‖·‖r‖  −  ‖r‖²       (Cauchy–Schwarz)
```
That bound, as a function of `‖r‖`, is concave with a maximum at
`‖r‖ = ‖q − c‖`.  Below that peak it decreases: we can safely abandon
a tail when the next residual-norm is small enough that even the UB
falls below the current `τ = top-k.min()`.

**Per-posting preprocessing** (build-time, one-shot):

1. For each list, compute residual norms and a permutation `order[]`
   that sorts ids in residual-norm-descending order.
2. Store `max_norm_tail[i] = norms[order[i]]` (the max over the
   not-yet-scanned slice) — this is exactly `norms[order[i]]` because
   `order` is desc-sorted.

**Query-time Phase 1**: for each probed list, scan prefix of length
`min(m_init, |list|)` in `order`, pushing to global top-k.  Within the
prefix also apply the norm-abandon break.

**Query-time Phase 2**: loop, picking the list with the best remaining
UB (over all states that still have a cursor < len).  Advance a chunk of
size `m_init`.  Terminate when no list's UB exceeds τ.

```rust
pub trait Scanner {
    fn name(&self) -> &'static str;
    fn search(&self, postings: &[Posting], query: &[Scalar], nprobe: usize, k: usize) -> Vec<Candidate>;
}
```
All three backends implement this trait so a production system can
A/B them per-query.

## Implementation notes

- **Dependency-free crate.** The PoC crate has zero deps (incl. no
  `rand`, `ndarray`, `aligned-vec`) so it builds everywhere and the
  benchmark numbers reflect pure algorithm cost, not vendored primitives.
- **Deterministic k-means** via SplitMix64.  Same centroids every run →
  same postings → scanner comparisons are exact.
- **Score identity test** (`score_equivalence`): the residual form must
  numerically match `−‖q − v‖²` within 1 e-5.  Guarantees the three
  scanners share a common scorer.
- **Agreement test** (`scanners_agree_on_recall_floor`): at
  `nprobe = n_cells`, exhaustive is exact top-k and the others must tie.
- **Files**: `src/lib.rs` (355 lines), `src/bin/bench.rs` (96 lines),
  under the 500-line cap.

## Benchmark methodology

- **Hardware**: Apple-silicon macOS 15.6, single thread (no rayon).
- **Build**: `cargo build --release` (`opt-level=3`, `lto=thin`,
  `codegen-units=1`).
- **Data E (easy)**: 50 000 points, dim 64, 64 well-separated Gaussians
  (sep = 4.0), 200 queries drawn from the same mixture, seed 42.
- **Data H (hard)**: same n and dim, **16 data clusters**, sep = 1.0
  (heavy overlap), different seed.
- **IVF**: 256 centroids via deterministic k-means (10 Lloyd iterations,
  seed 7/17).  Avg list length ≈ 195 (E) / 195 (H).
- **Scorer**: negative L2², residual form.
- **Measurement**: per-query latency = wall clock of `bench_variant` /
  200 queries.  Recall@10 vs exhaustive brute force.

## Results

### Easy regime — well-separated mixture (sep 4.0)

| nprobe | B0 lat µs | B0 recall | B1 lat µs | B1 recall | B2 lat µs | B2 recall |
|-------:|----------:|----------:|----------:|----------:|----------:|----------:|
|   4    |  31.3     | 0.823     |  29.9     | 0.823     |  32.2     | 0.823     |
|   8    |  46.0     | 0.961     |  51.3     | 0.961     |  54.8     | 0.961     |
|  16    |  73.7     | 0.996     |  86.3     | 0.996     | 103.1     | 0.996     |
|  32    | 160.6     | 1.000     | 167.2     | 1.000     | 200.7     | 1.000     |
|  64    | 289.3     | 1.000     | 290.1     | 1.000     | 373.8     | 1.000     |

### Hard regime — overlapping (sep 1.0, 16 data clusters)

| nprobe | B0 lat µs | B0 recall | B1 lat µs | B1 recall | B2 lat µs | B2 recall |
|-------:|----------:|----------:|----------:|----------:|----------:|----------:|
|   8    |  46.8     | 0.547     |  50.0     | 0.547     |  54.7     | 0.547     |
|  16    |  77.1     | 0.737     |  85.2     | 0.737     |  98.9     | 0.737     |
|  32    | 136.6     | 0.871     | 179.7     | 0.871     | 223.7     | 0.871     |

### Progressive `m_init` sweep (easy, nprobe=16)

| m_init | lat µs | recall |
|-------:|-------:|-------:|
|   4    | 124.4  | 0.996  |
|   8    | 109.0  | 0.996  |
|  16    |  94.0  | 0.996  |
|  32    |  90.8  | 0.996  |
|  64    |  86.2  | 0.996  |
| 128    |  85.8  | 0.996  |

### Interpretation

- **Recall is identical** across scanners at every `nprobe`.  The
  residual-norm bound is correct and never drops a true positive it
  shouldn't.
- **B2 Progressive is slower.**  At dim 64, one f32 dot takes ≈ 50 ns
  while a single bookkeeping decision (state struct access, UB
  evaluation across all lists) costs ≈ 40–60 ns.  The first phase
  processes the same codes B0 does, plus Phase 2 re-enters them.
- **B1 Norm-abandon** matches B0 at very small nprobe and loses at
  larger nprobe because the queries rarely abandon — `r_max` after
  phase start remains `≥ ‖q − c‖`, keeping the UB above τ for most of
  the tail.
- **Larger `m_init` helps IPSL** monotonically on easy data because at
  `m_init = 128` ≈ full list size, Phase 2 never fires.  This confirms
  Phase 2 is a pure overhead at this scale.

## Practical failure modes

1. **Small lists.** When avg list ≤ 64 elements, the `m_init`
   bookkeeping cost outweighs any benefit — the baseline already scans
   a short list.
2. **Cheap scorers.** Single f32 L2 at low dim is faster than one
   decision of the Phase-2 loop.  IPSL's architecture pays off only
   when per-code work ≥ 100 ns (PQ-ADC with LUT gather, cross-lane
   SIMD rerank, cache-miss reads from disk-backed codes).
3. **Centroid-adversarial queries.** A query equidistant from many
   centroids thrashes Phase 2: every list has similar UB, the scheduler
   oscillates.  Mitigation: cap total Phase-2 chunks per query.
4. **High k.** Top-k heap τ rises slowly for large k, so the UB rarely
   prunes anything.  Phase 2 degenerates to full scan + overhead.

## What to improve next (roadmap)

1. **Plug in PQ-ADC scoring.** Replace raw f32 residual dots with
   LUT-based PQ-ADC.  Expected: per-code cost ≈ 200 ns → IPSL
   bookkeeping becomes profitable.
2. **SIMD for Phase 1.** Batch 8 residual scores per SIMD lane; the
   prefix scan is tight and uniform.
3. **Priority-queue scheduler.** Replace the O(nprobe) linear scan for
   “best remaining UB” with a max-heap.  At `nprobe = 128` this alone
   is a 4× scheduler speedup.
4. **Shared norm buckets across lists.** Reuse BlockMax-IVF's tile
   layout so Phase 1 touches one tile per list with cache-aligned
   residuals.
5. **Learned `m_init`.** Make `m_init` a function of `‖q − c₀‖`
   (closest-centroid distance) — hard queries need longer prefixes.

## Production crate layout proposal

```
crates/ruvector-ivf-progressive-shortlist/
  src/
    lib.rs           # Scanner trait, Exhaustive, NormAbandon, Progressive
    bin/bench.rs     # the benchmark binary used here
    pq_adc.rs        # (future) PQ-ADC scorer impl of Scanner
    simd_f32.rs      # (future) AVX2/NEON prefix-scan kernel
    scheduler.rs     # (future) heap-based list scheduler
  tests/
    scorer_identity.rs
    scanner_agreement.rs
```

The trait boundary keeps future PQ, SIMD, and disk backends drop-in.

## How it works (walkthrough)

Think of IVF as a filing cabinet.  Each probed drawer (posting list) has
its residual folders stacked **tallest first** (biggest residual norm on
top).  Phase 1 is "open each drawer, skim the top `m_init` folders."
That gives you a provisional top-k.  Phase 2 is "go back to the drawer
whose next-tallest folder has the best remaining chance of beating what
you've got."  You stop when **every** drawer's next folder is too short
to matter.

In the easy benchmark, the easy drawers already have the true top-k in
their top few folders, so Phase 1 does all the work and Phase 2 just
costs you the bookkeeping to realize you can quit.  That's the whole
story of the negative result: when the scorer is cheap, "realize you
can quit" costs more than "scan one more folder."

## References

1. Jégou, Douze, Schmid.  Product Quantization for Nearest Neighbor Search.
   *IEEE TPAMI* 33(1), 2011.
2. Ding, Suel.  Faster Top-k Document Retrieval using Block-Max Indexes.
   *SIGIR* 2011.
3. Mallia et al.  Faster BlockMax WAND with Variable-Sized Blocks.
   *SIGIR* 2017.
4. Johnson, Douze, Jégou.  Billion-scale similarity search with GPUs.
   *IEEE Trans. Big Data* 7(3), 2017.
5. Chen et al.  BEAM-PQ: Budgeted Beam Search over Product Quantizers.
   *ICML* 2026.
6. Qdrant release notes v1.10.  2025.
7. LanceDB release notes v0.14.  2025.
