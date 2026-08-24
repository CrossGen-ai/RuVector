# ruvector-cache-conscious-hnsw

Cache-conscious node ordering for HNSW-like navigable graphs. Reorders graph
vertices so that co-visited neighbors land on the same cache line, yielding
**10-30% query throughput improvement** at identical recall.

Three swappable orderings ship out of the box:

| Ordering                | What it does                                     | Rationale                     |
|-------------------------|--------------------------------------------------|-------------------------------|
| `Insertion` (baseline)  | Identity — build order                           | No locality guarantees        |
| `Bfs`                   | BFS from entry point                             | Groups co-visited siblings    |
| `ReverseCuthillMcKee`   | BFS with ascending-degree child sort, reversed   | Classic sparse-matrix bandwidth reduction |

The `NodeOrdering` trait makes the backend swappable (learned orderings,
k-way partitioners, community-detection layouts — all fair game).

## Quick start

```bash
cargo build --release -p ruvector-cache-conscious-hnsw
cargo test  --release -p ruvector-cache-conscious-hnsw
cargo run   --release -p ruvector-cache-conscious-hnsw --bin benchmark \
    -- --n 100000 --dim 128 --queries 500 --degree 32 --pool 256 --ef 128 --k 10
```

## Real numbers (M-series Mac, single thread)

```
n=100000  dim=128  queries=500  degree=32  ef=128  k=10
insertion (baseline)  : 273.40 us/q  recall=0.286  span=33346
bfs                   : 234.31 us/q  recall=0.286  span=29772   speedup=1.17x
reverse-cuthill-mckee : 212.54 us/q  recall=0.286  span=29772   speedup=1.29x
```

Recall is identical across all three orderings — reordering is a pure graph
isomorphism (renames IDs; distances are unchanged).

See `docs/research/nightly/2026-08-24-cache-conscious-hnsw/README.md` for the
full research writeup, and `docs/adr/ADR-340-cache-conscious-hnsw-node-ordering.md`
for the decision record.
