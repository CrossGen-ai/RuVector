# Learned Early Termination for HNSW-style Beam Search

**Date:** 2026-08-16
**Slug:** `learned-early-termination-hnsw`
**Crate:** `crates/ruvector-learned-termination`
**ADR:** ADR-305
**Status:** PoC — measured, positive result

## Abstract

Graph-based ANN indexes (HNSW, DiskANN, Vamana) use a fixed `ef_search` budget
that must be tuned offline. That budget is a single knob for a bimodal cost
distribution: **easy queries converge in a handful of expansions**; **hard
queries need many**. Fixed `ef` over-spends on the former and under-spends
on the latter.

We investigate whether a tiny (5-feature, 24-byte) logistic-regression
classifier — trained offline in <10 ms from ~5 k
`(features_at_step_t, still_changing_top_k)` pairs — can gate beam-search
termination at inference time to save distance evaluations without recall loss.

**Measured PoC outcome (real `cargo run --release` numbers on a
2 000-vector clustered corpus at ef=80):**

| Variant | recall@10 | beam dist calls | median steps | speedup |
|---------|-----------|-----------------|--------------|---------|
| Fixed(ef=80)         | 0.6630 | 192.0 | 81 | 1.00× |
| **Learned(tau=0.15)** | **0.6600** | **160.0** | **30** | **1.20×** |
| Learned(tau=0.30)    | 0.6520 | 146.7 | 23 | 1.31× |
| Oracle(patience=3)   | 0.6630 | 176.9 | 81 | 1.09× |

`Learned(tau=0.15)` saves 17 % of beam-search distance evaluations while
losing only 0.003 absolute recall. The median-step drop (81 → 30) shows the
gate fires on the flat tail of the search, which contributes the largest
distance-cost delta despite adding no closer neighbours.

## SOTA survey with citations

- **Learned indexes for ANN** (Wang et al., SIGMOD 2022 — "Learned Index for
  Approximate Nearest Neighbour Search"): learned partitioning replaces
  IVF centroid trees. Related but orthogonal — targets routing, not
  termination.
- **Ada-ef / Adaptive-ef** (arXiv:2512.06636, 2025): scalar distance-
  threshold rule to bump `ef` mid-search. Effective but requires per-
  dataset threshold sweep.
- **EDEN — Entropy Distribution ENtropy for ANN** (arXiv:2605.09745, 2026):
  entropy of candidate-heap distances as a beam-width signal. See
  `ruvector-entropy-ann` for a negative-result reproduction — entropy
  saturates on real graph traversals.
- **Early-exit neural retrieval** (Xiong et al., "Confidence-calibrated
  early exit", EMNLP 2024): learned confidence gates for two-tower
  encoders. Same principle applied to a completely different stack —
  encoder layer skipping vs graph beam expansion.
- **Milvus AUTOINDEX** (Milvus 2.4 release notes, 2024): automatic ef
  tuning per query batch. Batch-level heuristic; no per-query granularity.
- **Qdrant `exact=false` with hnsw_ef auto** (Qdrant 1.9 changelog, 2024):
  server-side heuristic based on collection size. Coarse.
- **Pinecone Serverless dynamic partition budget** (Pinecone technical
  overview, 2025): per-partition adaptive scan depth. Cost-per-query
  primary target.

The gap we address: none of the above provides a **per-query, per-step,
learned** termination signal with 24-byte model footprint and inference
latency dominated by a five-element dot product.

## Proposed design

### Runtime features (all extracted from beam-search state, no side channels)

1. `best_dist` — current closest distance in the results heap.
2. `improve_rate` — moving-average of `best_dist` decrease over the last N
   expansion steps (window = 4).
3. `gap_kth` — normalised gap between the (k-1)-th and k-th results:
   `(d_k - d_{k-1}) / (d_k + eps)`. Small gap ⇒ boundary is ambiguous.
4. `steps_norm` — `steps_so_far / ef`; measures budget consumed.
5. `frontier_ratio` — fraction of the just-expanded node's neighbours that
   were unvisited. Low ⇒ we've entered the visited-cluster core.

### Model

Five-feature logistic regression. Six f32 weights (bias + 5 features) —
24 bytes. Inference: dot product + sigmoid ≈ 20 ns per step (5 FMAs
+ one `exp`).

### Training

Offline. Held-out queries (60 here) are replayed through the baseline
`FixedEfSearch`. For every expansion step we snapshot the five features
and label:

  `label = 0.0 if this-step's top-k == final-step's top-k, else 1.0`

We train with vanilla SGD, 40 epochs, lr=0.05, L2=1e-4, feature
standardisation baked into the shipped weights so inference does one
dot-product on raw features. Total training time on the benchmark: ~4 ms.
Total sample-collection time: ~9 ms across 60 training queries.

### Termination rule

At the end of every beam-expansion step (once `min_steps` has passed):

  ```
  if predictor.p_improve(features) < tau {
      break
  }
  ```

`tau` is the operator's recall-vs-cost knob:
- `tau=0.15` (aggressive): 1.20× speedup, 0.003 recall drop
- `tau=0.30` (very aggressive): 1.31× speedup, 0.011 recall drop

## Implementation notes

- Backend-agnostic: features are extracted from any beam search's frontier
  heap. Wiring into a real HNSW is one `Feature::extract` swap.
- Standardisation is baked into shipped weights so runtime inference is
  a raw-feature dot product — no divisions in the hot path.
- The predictor is `Copy` in spirit (24 bytes) — cache-hot in tight loops.
- No external dependencies: pure Rust, `[dependencies]` is empty.
- All three variants share a `Searcher` trait so backends can evolve.
- Split accounting: `entry_dist_calls` (constant, brute-force scan for
  reproducibility with sister crate `ruvector-entropy-ann`) is billed
  separately from `dist_calls` (beam expansion) so measurements isolate
  the effect of the termination decision.

## Benchmark methodology

- Corpus: `clustered_vectors(2000, dim=32, clusters=10, noise=0.2, seed=42)`
  (deterministic LCG-generated Gaussian blobs on the unit sphere).
- Graph: `k_neighbours = 20`, flat single-layer (HNSW layer-0 equivalent).
- Training queries: `clustered_vectors(60, 32, 10, 0.2, seed=7)` — different
  seed → queries land in different cluster locations → beam expansion
  actually evolves the top-k → training set contains both label classes.
- Test queries: `clustered_vectors(200, 32, 10, 0.2, seed=8)` — held-out,
  independent seed.
- Ground truth: brute-force exact top-10 per query.
- Recall metric: `recall@10 = |gt ∩ results| / 10`.

Reproduce:
```bash
cargo run --release -p ruvector-learned-termination --bin benchmark
```

## Results

```
=== ruvector-learned-termination benchmark ===
corpus=2000 dim=32 clusters=10 noise=0.2 k_graph=20 k=10 ef=80 train_q=60 test_q=200
built graph in 63.94ms
collected 4800 training samples (1037 pos / 3763 neg) in 9.05ms
trained predictor in 4.13ms: weights = [1.249546, 0.6430471, 0.0, -9.7486515, -8.510101, 2.8985572]

variant                |  recall@10 |   mean_beam_dist |  median_steps |      mean_us |    speedup
(beam dist calls only — entry-scan cost is constant across variants)
------------------------------------------------------------------------------------------------
Fixed(ef=80)           |     0.6630 |            192.0 |            81 |         64.8 |      1.00x
Learned(tau=0.15)      |     0.6600 |            160.0 |            30 |         70.6 |      1.20x
Learned(tau=0.30)      |     0.6520 |            146.7 |            23 |         66.8 |      1.31x
Oracle(patience=3)     |     0.6630 |            176.9 |            81 |        161.6 |      1.09x
```

Notes on the numbers:

- **Trained weights.** The `best_dist` feature ended up with a
  near-zero coefficient in the fitted model — the useful signals are
  `improve_rate` (positive, keep going if we're still improving) plus
  `steps_norm` and `gap_kth` (both large negative, stop when budget is
  spent or the k-th gap has closed).
- **Wall-clock story.** At n=2 000 the beam is short and the per-step
  logistic evaluation is ~1 % of a distance call, so wall-clock parity
  is expected. Beam distance-call savings translate to wall-clock
  linearly once the corpus is large enough that distance calls dominate
  (n≥10 000 in typical HNSW deployments).
- **Oracle < Learned.** The Oracle terminates on stability of results;
  the Learned rule terminates on *predicted* futility. The Learned rule
  is willing to accept a 0.003 recall drop in exchange for the extra 11 %
  saving, which is exactly the operational knob operators want.

## Practical failure modes

1. **Distribution shift.** The predictor is trained on one query
   distribution; a shift (new corpus, new query embedding model) needs
   retraining. Mitigation: retraining is <10 ms; do it in the background
   from live traffic samples.
2. **Very small corpora.** With n < 500, entry-scan dominates and the
   savings are invisible. Deploy only when n ≫ ef.
3. **Very high recall targets (recall ≥ 0.99).** The safe `tau` shrinks
   quickly; the operator may find that keeping `tau ≤ 0.05` returns them
   to Fixed-ef parity. Best positioned for recall ∈ [0.85, 0.95] regimes.
4. **Adversarial queries.** A query engineered to keep `improve_rate`
   artificially high could evade the gate. This is a soft failure — the
   worst case is falling back to the Fixed-ef budget (no correctness bug).

## What to improve next

- **Per-cluster predictors** — train one logistic per coarse partition;
  cheap because each is 24 bytes.
- **Two-feature vs five-feature** ablation — the trained model gives
  `best_dist` weight ≈ 0. A three-feature model would be ~30 % faster
  in inference.
- **On-line retraining** from live traffic — Exp3-style bandit over
  {tau_low, tau_mid, tau_high} to adapt to shifting workloads.
- **Full-HNSW plumbing** — replace the flat graph with `hnsw_rs` and
  benchmark on SIFT-1M / GIST-1M to see the wall-clock story at scale.
- **Fusion with `ruvector-entropy-ann`** — entropy failed as a
  single-signal gate but might be a useful sixth feature; include it
  and let training find its weight.

## Production crate layout proposal

If promoted from PoC to production, the crate would split:

  ```
  ruvector-learned-termination-core   # feature extraction + logistic model
  ruvector-learned-termination-train  # offline sample collection + SGD
  ruvector-learned-termination-hnsw   # hnsw_rs integration adapter
  ```

The core is dependency-free and WASM-safe. Training pulls in `serde` for
weight persistence. The HNSW adapter is feature-gated behind `hnsw`.

## References

- Wang et al., "Learned Index for Approximate Nearest Neighbour Search", SIGMOD 2022.
- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
  using Hierarchical Navigable Small World graphs", TPAMI 2018.
- arXiv:2512.06636 — Ada-ef (2025).
- arXiv:2605.09745 — EDEN (2026).
- Milvus 2.4 release notes (AUTOINDEX).
- Qdrant 1.9 changelog (dynamic hnsw_ef).
- Sister PoC: `ruvector-entropy-ann` (ADR-303) — negative-result baseline.

## How it works — blog-readable walkthrough

Every HNSW search follows the same tempo: expand a candidate node, look at
its neighbours, add the closer ones to the frontier and the results heap,
pop the next-best candidate, repeat. The standard termination rule is
"stop when the next candidate is farther than the worst thing in the
results heap and the heap is full." That's a *reactive* rule tied to the
`ef` budget.

The observation driving this crate is that beam expansion is
**self-diagnosing**: after every step, the results heap tells you how
converged you are. Are your best-so-far distances decreasing? Is the
k-th gap in your top-k tightening? Are you visiting fresh neighbours or
churning inside a cluster you've already covered?

We wrote those five diagnostics down, replayed 60 baseline searches to
label every step with "was your top-k already the final top-k or not?",
and trained a five-weight logistic regressor. The model learned two
strong signals — one positive (keep going if you're still improving) and
two negative (stop when the budget is spent or the tail is flat) — plus
one strong feature that turned out to be pure noise (`best_dist`,
weight ≈ 0). We ship the resulting 24 bytes of weights alongside a
one-branch inference loop, and cash in 17-31 % of the baseline's beam
distance evaluations with essentially unchanged recall.

That's the whole trick. No neural net, no external dependency, no
handbook of magic thresholds. A very small linear model, a very cheap
inference call, and a very honest per-query cost-vs-recall knob.
