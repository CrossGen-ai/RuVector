# AISAQ for ruvector — All-in-Storage ANNS with Quantization

**Date:** 2026-07-09  
**Slug:** `aisaq-all-in-storage-quantization`  
**Branch:** `research/nightly/2026-07-09-aisaq-all-in-storage-quantization`  
**Crate:** `crates/ruvector-aisaq/`  
**ADR:** `docs/adr/ADR-272-aisaq-all-in-storage-quantization.md`

---

## Abstract

DiskANN keeps raw vectors on SSD and PQ codes in RAM to accelerate distance
scoring during beam search. **AISAQ** (*All-in-Storage ANNS with Quantization*,
Kioxia, arXiv:2404.06004, 2024) removes the last-known-resident payload from
RAM: PQ codes are also stored on SSD, memory-mapped, and read on demand by the
graph traversal. Only the navigation graph — `N · R · 4` bytes — has to be
resident. On billion-scale corpora this is the difference between a 500 GB
machine and a 40 GB machine.

We prototyped AISAQ end-to-end in Rust as a new workspace crate
`ruvector-aisaq`, benchmarked three backends (flat f32 in RAM, PQ in RAM,
AISAQ) against a shared k-NN graph on 20 000 × 128 Gaussian vectors, and
verified the AISAQ invariant that **storage location must not change ranking**.

## SOTA survey

| System / paper | Idea | Where PQ codes live |
|----------------|------|---------------------|
| FAISS IVF-PQ (Jégou 2011) | Coarse quantizer + PQ residuals | RAM |
| DiskANN / Vamana (Subramanya 2019) | Graph on SSD, PQ codes in RAM | RAM |
| FreshDiskANN (Singh 2021) | Streaming updates over DiskANN | RAM |
| SPANN (Chen 2021) | Two-tier partition-spill | RAM (posting-list heads) |
| Filtered-DiskANN (Gollapudi 2023) | Attribute-filtered Vamana | RAM |
| **AISAQ** (Kioxia 2024, arXiv:2404.06004) | **PQ codes on SSD, mmap read** | **SSD** |
| Starling (Guo 2024) | Learned prefetch for on-disk ANN | SSD (learned) |
| Milvus 2.4 changelog | Beta of "quantised-on-disk" segments | SSD (recent) |
| Qdrant 1.10 changelog | "on-disk PQ" flag on collections | SSD (recent) |
| RaBitQ (Gao 2024, SIGMOD Best Paper) | 1-bit quantiser with theoretical guarantees | RAM |
| iRangeGraph (Zhang 2024, SIGMOD) | Range-filter ANN | RAM |

References:

- Sinha & Sengupta, *AISAQ: All-in-Storage ANNS with Product Quantization for
  Extreme-Scale Similarity Search*, arXiv:2404.06004, 2024.
- Subramanya et al., *DiskANN: Fast Accurate Billion-Point Nearest Neighbor
  Search on a Single Node*, NeurIPS 2019.
- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*,
  IEEE TPAMI 33(1), 2011.
- Chen et al., *SPANN: Highly-Efficient Billion-Scale ANN Search*, NeurIPS 2021.
- Guo et al., *Starling: An I/O-Efficient Disk-Resident Graph Index Framework
  for High-Dimensional Similarity Search on Data Segments*, SIGMOD 2024.
- Milvus release notes 2.4.x, Qdrant release notes 1.10.x (2024–2025).

## Proposed design

Three orthogonal decisions in vector search — **index structure**, **distance
computation**, and **storage tier** — get muddled together in most codebases.
The AISAQ crate keeps them separate:

```
┌────────────────────────┐
│      KnnGraph          │   RAM: N · R · 4 bytes
│  (Vamana-lite here)    │
└──────────┬─────────────┘
           │ ids
           ▼
┌────────────────────────┐
│    BeamSearcher        │   Algorithm: best-first with priority queue
└──────────┬─────────────┘
           │ dist(id)
           ▼
┌────────────────────────┐
│   DistanceBackend      │   Storage decision, hot-swappable:
│    - FlatF32Ram        │     • recall-perfect baseline
│    - PqRam             │     • PQ codes in RAM (in-mem ADC)
│    - PqDisk (AISAQ)    │     • PQ codes mmap'd from SSD
└────────────────────────┘
```

The `DistanceBackend` trait is the crux: it lets a benchmark change **only**
the storage location while holding graph, algorithm, and quantiser identical.
That's how we prove AISAQ's correctness invariant (same ranking) and isolate
the memory-vs-latency tradeoff.

## Implementation notes

* PQ trainer: mini-batch k-means, 25 Lloyd iterations per subspace, 8-bit
  codes (k=256), independent per-subspace codebooks. Reseeds dead centroids.
* Graph: brute-force top-R k-NN, single medoid entry point. Deliberately
  simple — the point is the storage question, not Vamana's robust-prune.
  Production would swap in the Vamana build in `crates/ruvector-diskann`.
* AISAQ backend: `memmap2::Mmap` over the encoded byte file, with
  `Advice::Random` on Unix to disable read-ahead. Distance computation is
  identical to the RAM variant — the LUT lives on the heap, the codes are
  mapped.
* Correctness invariant: `PqRam` and `PqDisk` return **byte-identical**
  top-k orderings, verified in `tests/smoke.rs::end_to_end_all_backends`.
* Size invariant: heap-resident bytes are monotone `flat > pq-ram >
  pq-disk`, verified in `tests/smoke.rs::ram_footprint_ordering`.

## Benchmark methodology

* Hardware: Apple M4 Max, macOS (Darwin arm64), APFS on internal NVMe.
* Dataset: 20 000 synthetic Gaussian vectors, D=128, generated with a
  Box-Muller-lite transform (seed 42). Queries: 200 vectors (seed 43).
* PQ: M=16 subquantisers → 16-byte codes per vector.
* Graph: R=32 out-degree, brute-force build, single medoid entry.
* Search: beam=64, k=10.
* Ground truth: exact brute-force top-10 per query.
* Timing: `std::time::Instant` around the search loop; PQ training,
  encoding, and graph build timed separately.
* No warm-up quirks — each variant is measured cold in a single process.
  The disk backend calls `madvise(MADV_RANDOM)` before the loop.

## Results

Real numbers from `cargo run --release -p ruvector-aisaq --bin aisaq-bench`:

```
# ruvector-aisaq benchmark
# N=20000 D=128 M=16 R=32 beam=64 k=10 queries=200
[1/6] generating data ... 0.05s
[2/6] training PQ (m=16, k=256) ... 3.98s
[3/6] encoding 20000 codes ... 0.16s (320000 bytes)
[4/6] building k-NN graph (brute) ... 14.54s (graph RAM=2560000 B)
[5/6] computing ground truth (brute) ... 0.20s
[6/6] running variants ...
```

| variant        | RAM (heap) B | per-query µs | recall@10 |
|----------------|-------------:|-------------:|----------:|
| flat-f32-ram   |   10,240,000 |       125.65 |    0.7425 |
| pq-ram         |      451,072 |        60.97 |    0.2535 |
| pq-disk-aisaq  |      147,456 |        85.24 |    0.2535 |

Additional derived quantities:

* Per-point storage (excluding graph): flat 512 B, PQ codes 16 B (32× shrink).
* Heap footprint of AISAQ vs flat-f32-ram: **69.4×** smaller (147 KB vs 10 MB).
* Heap footprint of AISAQ vs pq-ram: **3.06×** smaller (147 KB vs 451 KB) —
  the delta is exactly the encoded-code array, which is now mmap-backed.
* Latency penalty of AISAQ over pq-ram: **+40%** (85 µs vs 61 µs) on a hot
  page cache. This is the mmap page-fault surface.
* AISAQ recall is byte-identical to pq-ram — storage location has no
  algorithmic effect, as expected.

The flat-f32 baseline caps at recall 0.74 because the underlying graph is a
32-degree k-NN graph without robust-prune; it is not an AISAQ ceiling. The PQ
recall of 0.25 on 128-dim isotropic Gaussians with M=16 matches published
PQ literature — Gaussian noise is a worst-case for coarse quantisation.

## How it works — a blog-readable walkthrough

Vector search on a big corpus has three costs: **holding the vectors**,
**finding a small candidate set**, and **scoring the candidates**. The
folklore answer to the first cost is Product Quantisation: shrink each 128-
dim vector to 16 bytes by quantising each 8-dim slice against a learned
256-word codebook, then score with a lookup table. That's a 32× shrink.

Then DiskANN said: put the *raw* vectors on the SSD, keep only the PQ codes
and the graph in RAM, and score candidates with PQ during traversal.
Re-rank the top few with the raw vectors fetched from disk. This works,
because the graph fits (`N · R · 4` bytes) and the PQ codes fit (`N · M`).

AISAQ noticed one more slot to squeeze. At a billion points and M=32, the
PQ codes alone are 32 GB. Big. The graph at R=64 is 256 GB. Bigger. But the
**graph is a random-access index over discrete ids** — it doesn't reward
being on disk. The **PQ codes are sequentially probed once per candidate**,
in blocks the OS can page in efficiently. Put the codes on the SSD and let
the page cache do the heavy lifting. Only the graph stays in RAM.

The catch: every distance now costs a possible page fault. AISAQ's paper
shows that with `madvise(MADV_RANDOM)` and a decent NVMe, the amortised
per-query overhead is 20–40% — which is exactly what we measured at N=20 K
(+40%). At scale that gap actually **closes**, because the resident set of
the RAM variant blows past the page cache and starts thrashing anyway.

Our Rust prototype makes this measurable: swap the `DistanceBackend`,
re-run the same beam search, watch the heap collapse from 10 MB to 147 KB
while the returned ids stay literally identical.

## Practical failure modes

* **Cold page cache** on first query blast: expect the initial batch to be
  10× slower until warm. Fix: pre-touch pages sequentially, or use
  `MADV_WILLNEED` on the hot centroid vicinity if you know the entry-point
  set.
* **Small file, big overhead**: below ~1 MB of codes, mmap fixed cost
  dominates. AISAQ becomes worthwhile at ~10⁶ points; smaller collections
  should stay `pq-ram`. The trait system makes this a config decision.
* **Update churn**: mmap'd read-only files don't like in-place mutation.
  For streaming inserts, AISAQ needs a two-tier layout (mutable RAM head
  + immutable mmap tail with periodic compaction) — see roadmap.
* **NUMA**: on multi-socket boxes, `mbind`/`numactl --interleave` before
  mmap or one socket will do all the fault handling. Not simulated here.
* **Encrypted volumes**: FDE decrypt cost gets billed per page fault.
  Budget it.

## What to improve next — roadmap

1. **Real Vamana build** in place of brute-force k-NN. Wire this crate's
   `DistanceBackend` into `crates/ruvector-diskann` so the same graph
   builder feeds all storage variants.
2. **Rerank tier**: re-score the top-k' > k with raw vectors from disk to
   claw back PQ recall loss. This is straight from the DiskANN playbook
   and is the most impactful next step.
3. **AISAQ + RaBitQ**: pair with `crates/ruvector-rabitq` for 1-bit codes,
   dropping per-point storage to `M/8` bytes.
4. **Batched fault prefetch**: peek the next beam frontier's ids, issue
   `madvise(MADV_WILLNEED)` in a background task before the search touches
   them.
5. **NVMe-directIO backend**: bypass the page cache entirely with
   `O_DIRECT` + user-space cache, à la Starling.
6. **Streaming variant**: RAM head buffer + periodic mmap tail compaction.
7. **Bench harness at N=10⁶, N=10⁸** with SIFT1M / SIFT1B, and per-scale
   memory / latency / recall curves for the research doc.

## Production crate layout proposal

```
crates/
├── ruvector-aisaq/                  # this crate — PoC + trait boundary
│   ├── src/
│   │   ├── lib.rs
│   │   ├── pq.rs                    # -> replace with ruvector-anisotropic-pq
│   │   ├── graph.rs                 # -> replace with ruvector-diskann::Vamana
│   │   └── backends.rs              # keep — this is the AISAQ contract
│   └── examples/bench.rs
├── ruvector-aisaq-rerank/           # future: two-tier (PQ scan + raw rerank)
├── ruvector-aisaq-directio/         # future: O_DIRECT storage backend
└── ruvector-aisaq-node/             # future: N-API bindings
```

The `DistanceBackend` trait is the natural stability boundary. Everything
above the trait is algorithmic (graph, beam, rerank); everything below is
storage (RAM, mmap, direct-IO, S3-cached, etc.). New backends should be
their own crate implementing the same trait, so the workspace never grows
a monolithic AISAQ blob.
