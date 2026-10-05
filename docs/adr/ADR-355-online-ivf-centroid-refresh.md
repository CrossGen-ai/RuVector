# ADR-355: Online IVF Centroid Refresh via Reservoir Sampling

**Status:** Proposed  
**Date:** 2026-10-05  
**Crate:** `ruvector-online-centroid-refresh`  
**Research:** [docs/research/nightly/2026-10-05-online-ivf-centroid-refresh](../research/nightly/2026-10-05-online-ivf-centroid-refresh/README.md)

## Context

RuVector's IVF-family crates (`ruvector-batched-ivf`, `ruvector-blockmax-ivf`,
`ruvector-ef-ivf-posting`, `ruvector-soar-ivf`, `ruvector-bet4-ivf-bench`)
all share the classical assumption that the coarse centroid set is trained
once at index-build time and never changes. For long-lived agent-memory
and streaming-ingest workloads this assumption is wrong: distribution
shift silently degrades recall at a fixed `nprobe` because the centroids
no longer carve the vector space the way the live data does.

Existing options:
- **Full retrain.** O(corpus·k·iters). Blocks writes, needs a corpus snapshot.
- **CrackIVF-style splits.** Fixes posting-list hotspots but not global drift.
- **Fixed-cadence rebuild.** Simple but wastes CPU during quiet windows and
  reacts late during a sudden shift.
- **No refresh at all.** Status quo — observable recall decay over days/weeks.

The research PoC shows that a bounded reservoir plus a cheap `O(k)` drift
signal gives us a third option: in-process, write-path-friendly, with
compute matched to actual data movement.

## Decision

Introduce `ruvector-online-centroid-refresh` as a research-tier crate
exposing a trait-friendly `OnlineIvf` with three refresh policies
(`Static`, `Periodic(N)`, `DriftTriggered(θ)`), a reservoir maintained by
Algorithm R, and warm-start Lloyd re-fitting. The crate is standalone and
does not change any existing IVF crate; integration is deferred until the
RCU-swap and metrics follow-ups land (see ADR consequences).

Policy defaults:
- `reservoir_capacity = 50 · n_centroids`
- `refresh_iters = 5`
- `DriftTriggered` cooldown `= max(R/2, 16·k)` inserts, θ default `0.15`.

Public API lives at `OnlineIvf::{new, insert, assign, refresh,
drift_score, inertia}`.

## Consequences

### Positive
- Streaming IVF indexes can hold their recall-vs-nprobe curve under
  distribution shift without a full retrain.
- Drift-triggered mode spends compute only when the data actually moves;
  on steady streams it degenerates to `Static` after the cooldown gate.
- API is additive — no change to any existing crate until an integrator
  wires it into `ruvector-batched-ivf` et al.
- Deterministic reservoir (`ChaCha8Rng` + fixed seed) makes benchmarks
  reproducible.

### Negative
- Warm-start Lloyd on a 1 500-sample reservoir costs ~1–30 ms depending on
  `k` and `d`. During that call, inserts on the *same* index pause.
- Reservoir memory is a floor of `50·k·d·4` bytes. For `k=1024, d=768`
  that is 150 MiB — needs a tunable cap for large-k indexes.
- Centroid-id stability is only guaranteed for small shifts. Downstream
  posting-list caches must subscribe to a refresh event (not yet wired).

### Neutral / follow-ups
- RCU-swap of the centroid vector so query threads see a consistent
  snapshot during refresh — tracked in the "What to improve next"
  section of the research doc. Needs its own ADR once implemented.
- Streaming per-centroid mean/variance (ADR-worthy) would make the drift
  signal reservoir-free.
- Chao weighted reservoir would speed up response to sudden shift.
- Integration with `ruvector-batched-ivf` is explicitly out of scope for
  this ADR.

## Alternatives considered

1. **Full periodic retrain (status quo on some IVF crates).** Rejected:
   requires a corpus snapshot and blocks writes for the whole retrain.
2. **CrackIVF-style posting-list cracking (ADR-crackivf).** Not an
   alternative — complementary. Cracking fixes hotspots within the current
   centroid set; refresh moves the centroids.
3. **Drift signal via per-centroid online variance.** Rejected for v1 to
   keep the crate small; retained as a follow-up because it would remove
   the reservoir from the drift-check hot path.
4. **Soft-clustering (GMM / Online k-means with learning rate).**
   Rejected: changes the index semantics (vectors would belong to multiple
   centroids weightedly) and would ripple into every IVF scan kernel.
5. **External re-trainer process.** Rejected: adds operational surface and
   loses the "no corpus snapshot" property.
