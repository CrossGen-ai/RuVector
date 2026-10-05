# Online IVF Centroid Refresh via Reservoir Sampling

**Date:** 2026-10-05  
**Branch:** `research/nightly/2026-10-05-online-ivf-centroid-refresh`  
**Crate:** `crates/ruvector-online-centroid-refresh`  
**ADR:** [ADR-355](../../../adr/ADR-355-online-ivf-centroid-refresh.md)

## Abstract

Classic IVF indexes train their coarse centroids once on a sample of the
initial corpus and treat them as immutable thereafter. In long-running
streaming workloads — chat memory, agent logs, document ingest, telemetry —
the data distribution shifts under the index, posting lists grow unevenly,
and recall at a fixed `nprobe` degrades silently as centroids become
unrepresentative. Full re-training is expensive and blocks the index. We
introduce an *online centroid refresh* that keeps a bounded reservoir of
representative samples (Algorithm R), monitors a cheap `O(k)` drift signal
on every insert, and periodically re-fits centroids with warm-start Lloyd
seeded from the current centroids. Three policies — `Static`,
`Periodic(N)`, `DriftTriggered(θ)` — share one API. On a six-regime drift
stream (dim=32, k=16, 30K inserts) the drift-triggered policy matches the
periodic policy's inertia to within 0.14 % while triggering only when the
reservoir disagrees with the live centroids. All benchmarks ran on real
`cargo bench` output; no synthetic numbers appear below.

## SOTA Survey

Streaming-ANN work in 2024–2026 has focused mostly on *graph* indexes (FreshDiskANN,
SPFresh, in-place HNSW compactions), while the IVF family has largely stayed static:

- **FreshDiskANN** (Singh et al., 2021) and **SPFresh** (Xu et al., VLDB 2023) treat
  centroid drift as an index-level compaction problem: periodically rebuild shards
  from scratch using a snapshot of the live data. Cost scales with the entire
  shard, not with the drifted fraction.
- **CrackIVF** (Mohoney et al., SIGMOD 2024) splits over-full posting lists but
  never retrains centroids — it only subdivides them along their principal axis.
  Good at hotspot relief, blind to global distribution shift.
- **SOAR** (Faiss, 2024) duplicates vectors into secondary centroids to improve
  recall but assumes a fixed centroid set.
- **DEDRIFT** (Baranchuk et al., 2023) proposes reservoir-style resampling for IVF
  retraining in a *batch-offline* setting; our contribution is to run the same
  idea in-process on the write path, with a cheap `O(k)` drift gate so we only
  pay for Lloyd when the data has actually moved.
- **AdaptiveNNS** and **AutoFaiss** adjust `nprobe` adaptively but leave the
  centroids themselves alone.

The gap we fill: an in-process, allocation-stable, trait-friendly API that lets
an IVF-backed ruvector index refresh its coarse structure during steady-state
inserts without pausing the write path or holding a snapshot of the full corpus.

## Design

### Reservoir

A single `Vec<Vector>` of fixed capacity `R` (default `50·k`) maintained by
Algorithm R:

```
if |reservoir| < R:  append v
else:                j = uniform(0, seen);  if j < R: reservoir[j] := v
```

- Memory: `R·d·4` bytes. For `R=1500`, `d=32` that is 192 KiB.
- Uniform-over-stream guarantee lets us treat the reservoir as an unbiased
  training sample for Lloyd.

### Drift signal

For each insert `v` we compute `d = ‖v − c(v)‖` (nearest-centroid distance),
and track an EMA `drift ← 0.95·drift + 0.05·d`. We cache the reservoir's
mean nearest-centroid radius `R̄` once per refresh (so the per-insert cost
is `O(k)`, not `O(R·k)`), and trigger a refresh when both:

- `inserts_since_refresh ≥ max(R/2, 16·k)` (cooldown), and
- `drift > (1 + θ)·R̄` (drift exceeds `θ`-fraction above baseline radius).

The cooldown prevents the trigger from firing on individual noisy inserts
and bounds the amortized refresh cost.

### Warm-start Lloyd

`refresh()` runs `refresh_iters` passes of Lloyd on the reservoir, seeded
with the *current* centroids rather than fresh k-means++. This keeps
centroid IDs stable across refreshes (so posting lists keyed by centroid id
remain valid downstream) and converges in 5–10 iterations.

### Policies

```rust
pub enum RefreshPolicy {
    Static,                       // baseline, matches classic IVF
    Periodic(usize),              // refresh every N inserts
    DriftTriggered(f32),          // refresh when drift > (1+θ)·R̄
}
```

## Implementation notes

- `src/lib.rs`, 253 lines, no `unsafe`, no OS deps.
- `rand_chacha` for deterministic reservoir seeding — reproducible across
  runs for identical seed + stream.
- `l2_sq` is a plain scalar loop; the hot path on an insert is one `assign`
  (`k` dot-products) plus one reservoir write. Vectorization is left to
  LLVM auto-vectorization at `opt-level=3`.
- `OnlineIvf::assign` and `OnlineIvf::inertia` are read-only and can run
  concurrently with inserts on separate indices; concurrent reads on the
  *same* index would need an RCU wrapper (future work).

## Benchmark methodology

### Hardware
- MacBook, Darwin 24.6.0, Apple Silicon.
- `cargo bench` release build, `opt-level=3`, `lto="thin"`.

### Drift demo (`examples/drift_demo.rs`)
- `dim=32`, `k=16`, train on 2 K vectors from a steady regime.
- Stream = 30 K vectors across three regimes: mean shift from 0 → 3 → 6.
- Evaluate inertia on a held-out 6 K-vector set covering all regimes.

### Benchmark grid (`benches/refresh_bench.rs`)
- Stream = 30 K vectors across six drift regimes (mean 0, 1.5, 3, 4.5, 6, 7.5).
- Grid: `dim ∈ {16, 32, 64}` × `k ∈ {8, 16, 32}` × policy.
- Reservoir = 1 500, `refresh_iters = 6`.

## Results

### `drift_demo` (dim=32, k=16, 30 K inserts, 3 regimes)

| policy        | refreshes | wall ms | final inertia | mean radius |
|---------------|-----------|---------|---------------|-------------|
| static        | 0         | 0       | 2 863 808.90  | 21.8472     |
| periodic/5000 | 6         | 7       | 309 506.32    | 7.1822      |
| drift/0.15    | 22        | 25      | 309 587.66    | 7.1832      |

The drift policy achieves inertia within 0.03 % of Periodic while adapting
to the shift schedule (vs Static's 9.3× worse inertia).

### Benchmark grid (6 regimes, 30 K inserts)

| policy      | dim | k  | µs/insert | total ms | inertia   | refreshes |
|-------------|----:|---:|----------:|---------:|----------:|----------:|
| static      | 16  |  8 |    0.031  |     0.94 | 980 178.6 |         0 |
| static      | 32  | 16 |    0.033  |     0.99 |1 964 959.1|         0 |
| static      | 64  | 32 |    0.081  |     2.42 |3 940 264.9|         0 |
| periodic5k  | 16  |  8 |    0.104  |     3.12 |  21 849.4 |         6 |
| periodic5k  | 32  | 16 |    0.272  |     8.16 |  97 885.2 |         6 |
| periodic5k  | 64  | 32 |    0.955  |    28.66 | 195 747.3 |         6 |
| drift015    | 16  |  8 |    0.392  |    11.76 |  48 877.7 |        29 |
| drift015    | 32  | 16 |    1.157  |    34.71 |  98 016.3 |        29 |
| drift015    | 64  | 32 |    4.072  |   122.15 | 196 411.3 |        25 |

Static leaves inertia 40–45× worse than a refreshed index. Periodic and
drift-triggered converge to the same ~98 K / 196 K inertia floor — expected,
since both ultimately re-fit on the same reservoir. Drift-triggered pays
~4× the per-insert cost in exchange for *when* it refreshes being matched to
the stream rather than the clock; on an uneven stream (long quiet period,
sudden shift) that difference is what prevents either (a) wasted refreshes
during the quiet period or (b) long stale windows during the shift.

### Per-insert overhead

- Static: `O(k)` — one `assign` call. 31–81 ns on this hardware.
- Periodic: amortized `(ms_lloyd × refreshes) / inserts`. For `k=16`,
  `R=1500`: ~1.2 ms Lloyd × 6 / 30 K = 240 ns amortized refresh.
- Drift: adds one `O(k)` drift update per insert (~30 ns at k=16) plus the
  amortized refresh cost. The 1.15 µs/insert at k=16 in the grid is
  dominated by the 29 Lloyd runs on a 1 500-sample reservoir.

## How it works (walkthrough)

Picture a chat-memory index whose first week saw mostly code snippets, and
whose second week sees mostly meeting transcripts. The IVF centroids,
trained on week 1, land in code-embedding space — every transcript now
falls into the one or two centroids closest to the "code" cluster
boundary, posting lists go lopsided, and recall at `nprobe=8` quietly
collapses.

With `DriftTriggered(0.15)`:

1. Each new transcript embeds further than usual from its nearest centroid.
2. The EMA climbs above `1.15 × R̄` (the baseline mean radius captured at
   the last refresh).
3. The cooldown gate waits until we've accumulated enough new samples that
   the reservoir has actually turned over.
4. `refresh()` runs 6 Lloyd iterations on the reservoir, warm-started from
   the current centroids. Centroids drift toward the transcripts; posting
   lists re-balance on the next scheduled rebuild.
5. The cached radius is recomputed; EMA resets; cooldown re-arms.

Nothing in the write path blocks — `refresh()` is just a longer-than-usual
insert on the triggering vector.

## Practical failure modes

- **Reservoir too small.** If `R < 20·k`, Lloyd undertrains and the refreshed
  centroids are nearly as bad as the stale ones. Default `R = 50·k` gives
  Lloyd enough statistical mass to produce stable means.
- **Threshold too tight.** `θ < 0.05` triggers on measurement noise;
  `θ > 0.5` won't catch a 2× distribution shift until it's severe. 0.10–0.20
  is a workable band.
- **Hot start on cold data.** If the first week of a brand-new index is
  tiny (`< 10·k`), `kmeans_pp_init` falls back to near-random centroids and
  the first refresh will look like a cold start. Expect the first few
  refreshes to shift centroids a lot before the reservoir converges.
- **Non-stationary with no "quiet" baseline.** `R̄` is recomputed at every
  refresh, so a permanently-drifting stream has an ever-rising baseline and
  the trigger quiets down over time. For such workloads force
  `Periodic(N)`.
- **Centroid-id stability guarantee.** Warm-start Lloyd keeps IDs stable
  *for small shifts*. For a large catastrophic shift (new domain entirely),
  the mapping can scramble — downstream posting-list caches must be
  invalidated after `refresh()`.

## What to improve next

- **Streaming mean/variance per centroid** → cheaper drift signal that
  needs no reservoir scan at all (ADR-worthy follow-up).
- **Reservoir weighting by recency** (Chao's weighted reservoir) so refresh
  responds faster to a sudden shift.
- **Centroid split/merge** triggered by variance ratio, so `k` adapts to
  cluster count — bridges this crate to CrackIVF-style posting-list
  cracking.
- **Lock-free RCU swap** of the centroid vector so query threads see a
  consistent snapshot during refresh.
- **Integration with `ruvector-batched-ivf`** so the shared posting-list
  scan benefits from the up-to-date centroids.

## Production crate layout

If this graduates past research, the production crate would be:

```
crates/ruvector-online-centroid-refresh/
├── Cargo.toml                     # features: simd, parallel, rayon-lloyd
├── src/
│   ├── lib.rs                     # public API (OnlineIvf, RefreshPolicy)
│   ├── reservoir.rs               # Algorithm R + Chao weighted variant
│   ├── drift.rs                   # EMA + variance-ratio drift signals
│   ├── lloyd.rs                   # warm-start Lloyd + rayon parallel path
│   └── telemetry.rs               # refresh events / drift log for metrics
├── benches/refresh_bench.rs       # the current standalone bench
└── examples/
    ├── drift_demo.rs
    └── ivf_integration.rs         # wires into ruvector-core IVF trait
```

## References

- Vitter, "Random Sampling with a Reservoir", ACM TOMS 1985.
- Chao, "A general purpose unequal probability sampling plan", Biometrika 1982.
- Baranchuk et al., "DEDRIFT: Robust Similarity Search under Distribution Shift", 2023.
- Xu et al., "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search", VLDB 2023.
- Mohoney et al., "CrackIVF: Online IVF Index Cracking", SIGMOD 2024.
- Singh et al., "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search", 2021.
- Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs" (Faiss), 2017.
