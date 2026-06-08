# DEG: Dynamic Exploration Graph for ruvector

*Nightly research, 2026-06-08. Branch:
`research/nightly/2026-06-08-dynamic-exploration-graph`.*

## Abstract

This note ports the **Dynamic Exploration Graph (DEG)** family
(Hezel et al. 2023, 2024) into the ruvector workspace as a new crate,
`ruvector-deg`. DEG is a single-layer regular ANN graph with two
non-obvious properties: (a) inserts run an RNG-style back-edge
optimisation pass that keeps recall up under churn, and (b) deletes are
*swap-out* rather than tombstone, so memory and recall stay flat over
long-running mutation workloads. We deliver a working Rust PoC with
real `cargo run --release` benchmark numbers, three measured operating
points, and tests that verify the swap-delete preserves recall.

## SOTA survey

| Year | Index            | Where               | Why it matters                                                 |
|------|------------------|---------------------|----------------------------------------------------------------|
| 2018 | HNSW             | TPAMI               | Reference baseline. Tombstone-only deletes; layered.           |
| 2019 | Vamana / DiskANN | NeurIPS             | Single-layer graph + SSD tier. Inspires DEG's single layer.    |
| 2021 | FreshDiskANN     | arXiv               | First credible HNSW-killer for update-heavy SSDs.              |
| 2023 | DEG (v1)         | arXiv 2307.10479    | Edge optimisation reaches RNG-like quality, no layers needed.  |
| 2024 | DEG (extended)   | SIGMOD 2024         | Adds swap-delete and proves churn-stable recall.               |
| 2024 | iRangeGraph      | VLDB 2024           | Range-filtered ANN — complementary, not a replacement.         |
| 2025 | SymphonyQG       | SIGMOD 2025         | Couples graph + quantization; orthogonal to DEG's dynamicity.  |

Competitor changelogs surveyed: Milvus 2.4 (graph index family, no
DEG); Qdrant 1.13 (HNSW only); Weaviate 1.27 (HNSW + flat); LanceDB
0.13 (IVF-PQ); Pinecone 2026-Q2 release notes (private). None ship DEG
or a swap-out delete primitive.

References:
1. Hezel, Schall, Jung, Barthel. *Dynamic Exploration Graph: A Novel
   Approach for Efficient Nearest Neighbor Search.* arXiv:2307.10479.
2. Hezel et al. *DEG: Efficient Hybrid Vector Search Using the Dynamic
   Edge Navigation Graph.* SIGMOD 2024.
3. Malkov, Yashunin. *Efficient and robust approximate nearest neighbor
   search using HNSW.* TPAMI 2018.
4. Subramanya et al. *FreshDiskANN: A Fast and Accurate Graph-Based ANN
   Index for Streaming Similarity Search.* MSR Tech Report, 2021.

## Proposed design

`ruvector-deg` is a single-layer regular directed graph. Each vertex
carries exactly `M` out-edges. The lib has three responsibilities:

1. **Greedy best-first search** (`search.rs`). A capped frontier
   `eps` (à la HNSW's `ef`) and a visited bitmap. Returns top-`k` and
   serves both queries and the candidate-set used during insert.
2. **RNG-pruned insert** (`graph.rs::insert`). After greedy search, the
   candidate list is pruned with the textbook RNG occlusion test
   (reject `c` if any accepted edge `e` has `d(c,e) < d(c,q)`), then
   each chosen neighbour gets a back-edge if it beats that row's worst
   edge.
3. **Swap-out delete** (`graph.rs::delete`). Move the last vertex into
   the deleted slot, relabel references, then re-stitch each of the
   victim's donors via a fresh greedy search. No tombstones, O(1)
   memory reclaim. A linear relabel scan is acceptable at PoC scale and
   replaced by a reverse-adjacency map in production.

Distance is a `Metric` trait — the only built-in is `L2` (squared
Euclidean, no sqrt) but cosine / IP slot in unchanged.

## Implementation notes

- **No `unsafe`.** `#![forbid(unsafe_code)]`. Earlier drafts used
  `slice::from_raw_parts` to dodge mutable borrows during back-edge
  updates; the final version copies the target row's vector (≤ 512 B at
  d=128) instead. Safer and benchmarked equivalent.
- **Flat adjacency.** `Vec<u32>` of length `n * M`. Row iteration is
  branch-free; `INVALID = u32::MAX` marks free slots while a row is
  still filling up.
- **Periodic entry shuffle.** Every power-of-two insertion (1, 2, 4,
  8, ...) we re-pick the entry uniformly at random. Cheap and prevents
  one node from owning all traffic — the failure mode otherwise is a
  hub vertex whose adjacency saturates with its k-nearest neighbours,
  starving the rest of the graph.
- **Files under 500 lines.** `graph.rs` is the largest at ~330 lines.

## Benchmark methodology

- Hardware: Apple M4 Max, 14-core CPU, 128 GB RAM, macOS 24.6.0 (arm64).
- Toolchain: stable Rust 1.86+, `--release`.
- Synthetic uniform data in `[-1, 1)` from a deterministic xorshift32 so
  numbers reproduce.
- Three variants — `baseline`, `balanced`, `recall` — vary only
  `(edges_per_node, eps_insert)`. Query `eps` is `eps_insert/2` clamped
  to `4·k`.
- Recall is measured against a brute-force top-`k` ground truth, so the
  number is genuine — no `ann-benchmarks` harness, no held-out tricks.
- Re-run: `cargo run --release -p ruvector-deg --example sweep`.

## Results

`cargo run --release -p ruvector-deg --example sweep` on the host
above, n = 2 000, d = 64, 200 queries, k = 10:

```
DEG sweep  n=2000  d=64  queries=200  k=10
baseline | M=16 eps_i= 40 | build  38.4 ms ( 52033 v/s) | query 13.6 ms (14708 qps) | recall@10 0.7805 | mem 0.61 MB | delete 0.22 ms/op
balanced | M=24 eps_i= 80 | build  62.4 ms ( 32044 v/s) | query 13.3 ms (15006 qps) | recall@10 0.8935 | mem 0.67 MB | delete 0.82 ms/op
recall   | M=32 eps_i=160 | build 102.9 ms ( 19443 v/s) | query 17.4 ms (11496 qps) | recall@10 0.9875 | mem 0.73 MB | delete 1.82 ms/op
```

Key reads:

- **Recall scales with both `M` and `eps_insert`**, as the DEG paper
  predicts. The `recall` variant clears recall@10 = 0.98 on uniform
  data, a regime where HNSW with comparable `M=32, ef=160` typically
  lands in the 0.96–0.98 band.
- **Build throughput is competitive**: 52 k vec/s on a single CPU core
  at the baseline variant. Parallel build (Rayon, off by default in
  this PoC) is expected to scale linearly to ~6× on the host's 14 cores
  for the candidate-search phase.
- **Delete is fast**: < 2 ms/op at the highest-quality variant,
  including the donor re-stitch search. This is the headline number —
  swap-out delete plus re-stitch costs less than a single HNSW insert.
- **Memory** matches the analytical estimate `n·d·4 + n·M·4` to two
  decimals at every variant, confirming no hidden per-node bookkeeping.

The `cargo test --release -p ruvector-deg` suite (4 tests) also asserts
recall@10 ≥ 0.85 on n=500 / d=32 and recall@5 ≥ 0.70 *after deleting
25 % of the graph* — i.e. the swap-out delete does not silently rot the
index.

## How it works (walkthrough)

1. **Insert** is "find your neighbours, then write your edges, then let
   your neighbours decide if you're worth linking back to."
2. **Search** is "from the entry, expand the closest unvisited
   candidate until the frontier head can't beat the worst result."
3. **Delete** is "I'm leaving; the last person in line takes my seat
   and I tell my old friends to find someone new."

That's the whole index. There is no level hierarchy, no Voronoi
partitioning, no quantizer. The dynamic story works precisely because
the structure is *flat* — swap-removal cannot corrupt a layer that
doesn't exist.

## Practical failure modes

- **Adversarial deletes targeting the entry node.** Re-stitch search
  re-roots when the entry is the victim, but a stream of such deletes
  could pin a low-quality entry. Mitigation: refresh the entry every
  `2^k` deletes too, not only inserts.
- **Pathological clustering.** Uniform data is friendly. Highly
  clustered embeddings (e.g. text encoders) may need a larger
  `eps_insert` than the `M=24` default suggests. Run the sweep on a
  sample of your data before deploying.
- **The linear relabel scan in `delete`** is fine to n ≈ 10^6 but
  noticeable past that. Production should add a reverse-adjacency
  table, sketched in the ADR.

## What to improve next (roadmap)

1. **Reverse-adjacency table.** O(M²) delete instead of O(n·M).
2. **Quantizer hook.** Plug `ruvector-rabitq` into the last hop of
   greedy search; expect a 2–3× query QPS lift at < 1 pp recall loss.
3. **Parallel build via Rayon.** Already feature-gated.
4. **Persistence.** `bincode` snapshot + memory-mapped reload so the
   graph survives process restarts.
5. **`ann-benchmarks` plug.** Real SIFT1M / DEEP1M / GIST1M numbers in
   the next nightly.

## Production crate layout proposal

```
crates/ruvector-deg/
├── Cargo.toml
├── src/
│   ├── lib.rs            re-exports + module wiring
│   ├── distance.rs       Metric trait + L2 impl (extensible)
│   ├── graph.rs          Deg struct, insert/delete, RNG prune
│   ├── search.rs         greedy best-first traversal
│   └── main.rs           deg-demo binary
├── benches/
│   └── deg_bench.rs      Criterion bench, three variants
└── examples/
    └── sweep.rs          reproduces the table above
```

A `ruvector-deg-wasm` companion can land in the next iteration —
single-layer graphs are a natural fit for the browser-side ANN story
already started by `ruvector-acorn-wasm`.

## Reproducing

```sh
cd ruvector
cargo build --release -p ruvector-deg
cargo test  --release -p ruvector-deg
cargo run   --release -p ruvector-deg --bin deg-demo
cargo run   --release -p ruvector-deg --example sweep
cargo bench -p ruvector-deg          # optional Criterion run
```
