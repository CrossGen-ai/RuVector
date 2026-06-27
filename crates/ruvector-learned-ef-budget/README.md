# ruvector-learned-ef-budget

Per-query learned `ef_search` budget predictor for HNSW, in pure safe Rust.

Most HNSW deployments pick one static `ef_search` calibrated to the worst
query in the workload, and overserve every other query. This crate fits a
tiny 8-weight ridge-regressed predictor that reads cheap per-query
features (norm, per-dim stats, distance to k-means medoids, entry-point
degree) and predicts the minimum `ef` needed to hit a per-query recall
target. On a heterogeneous 20k × 64d benchmark at target recall 0.95:

| Variant | mean distances/query | mean µs/query | recall |
|---|---|---|---|
| baseline-pessimist (static ef=64) | 540 | 57 | 0.81 |
| oracle (per-query best ef) | 365 | — | 0.84 |
| **learned (this crate)** | **471** | **41** | **0.79** |

**12.8% fewer distance computations, 28% lower latency, 1.29× of oracle.**

See [ADR-271](../../docs/adr/ADR-271-learned-ef-budget.md) and the research
note in `docs/research/nightly/2026-06-27-learned-ef-budget/` for the full
methodology and numbers.

## Quick start

```bash
cargo run --release -p ruvector-learned-ef-budget
```

Writes `bench_results.json` with the full per-variant breakdown.

## API

```rust
use ruvector_learned_ef_budget::{
    Hnsw, HnswParams, Medoids, BudgetPredictor, OracleBudget,
    PredictorConfig, extract_features,
};

let mut idx = Hnsw::new(dim, HnswParams::default());
for v in corpus.chunks(dim) { idx.insert(v); }

let medoids = Medoids::fit(&corpus, dim, 16, 0xFEED);
let cfg = PredictorConfig::default();
let ladder = [8, 16, 32, 64, 128, 256, 512];

// 1. offline: build oracle labels + features
let mut feats = Vec::new();
let mut labels = Vec::new();
for q in train_queries.chunks(dim) {
    let mut s = Default::default();
    feats.push(extract_features(q, &medoids, &idx, &mut s));
    labels.push(OracleBudget::label(&idx, &corpus, dim, q, &cfg, &ladder));
}

// 2. fit predictor (safety margin = 0.6 in log2 space)
let predictor = BudgetPredictor::fit(&feats, &labels, cfg, 0.6).unwrap();

// 3. online: per-query budget then search
let mut s = Default::default();
let ef = predictor.predict_query(query, &medoids, &idx, &mut s);
let (results, _) = idx.search(query, k, ef);
```

## License

Same as the parent workspace.
