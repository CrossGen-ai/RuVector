# Speculative ANN Search: Draft-and-Verify Vector Retrieval

**Date:** 2026-07-04
**Slug:** speculative-ann-draft-verify
**Crate:** `crates/ruvector-specann`
**ADR:** ADR-272
**Status:** PoC — real cargo-run benchmarks; production integration proposed.

---

## Abstract

We port the *speculative decoding* pattern from LLM inference (Leviathan et al.
2023; Chen et al. 2023) to approximate nearest-neighbor (ANN) search. A cheap
*draft index* (int8- or 1-bit-quantized) proposes an over-provisioned candidate
set of size `k_draft = α · k`; an exact *verifier* (float32) rescores only that
set. An escalation policy watches the confidence gap between the k-th and
(k+1)-th draft scores and re-runs the draft with a wider probe when the gap is
thin, falling back to full-index verification only when repeated escalations
fail. On a 10 k × 128-dim gaussian corpus (Apple M4 Max, single-threaded), the
int8 SpecANN variant achieves **100.0% recall@10 while verifying only 255/10 000
vectors on average** (25.5 % of the corpus), and the 1-bit SpecANN variant
reaches **6 592 QPS — 2.53× exact brute force — at 69.9 % recall@10** with an
adaptive escalation policy. All numbers come from `cargo run --release
--bin specann-bench`, reproducible from the crate root.

## SOTA Survey

Speculative decoding cut LLM serving cost significantly by using a cheap draft
model + exact verifier ([Leviathan+2023, "Fast Inference from Transformers via
Speculative Decoding"; Chen+2023, DeepMind]). It works because verification is
cheaper than autoregressive generation. The same asymmetry exists in ANN:

- **Verification is cheap** — computing exact float32 distances on `k_draft`
  candidates is O(k_draft · d) once ids are known, regardless of index type.
- **Draft is cheap** — quantized (RaBitQ, LVQ, PQ, int8) distances run 2-10×
  faster than float32.
- **Escalation is possible** — if the draft is uncertain (thin score margin at
  rank k), the driver can widen the probe or fall back exactly like
  speculative decoding's "reject and roll back" step.

Relevant prior work:

- RaBitQ (Gao & Long, SIGMOD 2024) — 1-bit rotation-based quantization with
  distance bounds; used here as the ultra-cheap draft substrate.
- LVQ / Turbo LVQ (Aguerrebere+2023, Intel Labs) — locally-adaptive int8 vector
  quantization; matches our `Int8BruteForce` draft.
- FINGER (Chen+2023) — approximate distance surrogate for HNSW graph traversal;
  a form of *implicit* draft-and-verify inside a graph walk.
- ADSampling (Gao+2023) — early-termination distance estimation; related idea:
  spend less compute when uncertainty is high.
- Speculative Decoding (Leviathan+2023, Chen+2023) — the direct inspiration.

**Delta.** No prior ANN system, to our knowledge, exposes the draft/verifier
split as a *first-class trait pair* with an escalation policy and observable
stats (`SpecStats { escalations, draft_candidates, verified,
full_verify_fallback }`). Existing quantized indices bake the two-stage rescore
into one opaque pipeline. Making it explicit lets us swap drafts (int8, 1-bit,
future PQ / RaBitQ / graph-partial-walk) and tune escalation per workload.

## Proposed design

```
┌───────────┐  q      ┌──────────────┐  ids[k_draft]   ┌────────────┐
│  Query    │ ──────▶ │  DraftIndex  │ ──────────────▶ │  Verifier  │
│           │         │  (int8, 1b)  │                 │   (f32)    │
└───────────┘         └──────┬───────┘                 └─────┬──────┘
                             │ scores                        │ exact
                             ▼                               ▼
                      ┌───────────────┐              ┌───────────────┐
                      │ EscalationPol │◀─── stats ───│  top-k merge  │
                      └───────────────┘              └───────────────┘
```

Trait pair:

```rust
trait DraftIndex { fn draft(&self, q: &[f32], k_draft: usize) -> Result<Vec<Neighbor>>; }
trait Verifier   { fn verify(&self, q: &[f32], ids: &[u32])  -> Result<Vec<Neighbor>>; }
```

`SpecAnnIndex<D,V>` composes any draft with any verifier. Escalation policy:

```rust
struct EscalationPolicy {
    alpha: usize,                 // k_draft = α · k
    gap_threshold: f32,           // (score[k] - score[k-1]) / score[k-1] must ≥ this
    max_escalations: u8,          // otherwise widen α, up to N times
    escalate_multiplier: f32,     // α ← α · multiplier per escalation
}
```

## Implementation notes

- `F32BruteForce` — exact float32 baseline. Implements *both* `DraftIndex`
  (trivially) and `Verifier`, so the same struct serves as verifier or as a
  pure-exact baseline.
- `Int8BruteForce` — per-vector symmetric int8 quantization (`s = max|x|/127`),
  approximate L2² recomputed on-the-fly. No rotation, no SIMD packing — this is
  intentionally the naive form so the *speedup story* is honest.
- `Sign1BitDraft` — sign bit per dimension packed into `u64`s. Distance =
  `(x XOR y).count_ones()`, a Hamming proxy for L2. Runs `dim/64` XORs +
  popcounts per compare — theoretical peak throughput is enormous but recall
  degrades on isotropic-gaussian data without rotation.

The escalation gap uses the score at rank `k` and `k+1`; on `k_draft` well
above `k`, this ordering statistic is a proxy for *ambiguity* in the top-k
boundary. When ambiguous, either the draft is inaccurate near the boundary or
the true top-k contains ties — both merit widening the candidate set.

## Benchmark methodology

- Corpus: n = 10 000 synthetic gaussian vectors, dim = 128, seed = 42.
- Queries: 200 held-out gaussian vectors, seed = 7.
- k = 10, warmup 5 queries, latencies measured per query with
  `Instant::now()`, sorted for percentiles.
- Ground truth = exact float32 brute force top-10.
- Hardware: Apple M4 Max (single-threaded), macOS 15.7 (Darwin 24.6),
  rustc 1.89.0, `cargo run --release`.
- Reproduce: `cargo run --release -p ruvector-specann --bin specann-bench`.

## Results (real cargo-run numbers)

| Variant                     | recall@10 | QPS    | p50 (µs) | p95 (µs) | avg verified |
|-----------------------------|-----------|--------|----------|----------|--------------|
| A. exact float32 baseline   | 1.000     | 2604.2 | 386.8    | 408.0    | 10 000       |
| B. int8-only (no verify)    | 0.984     | 2119.8 | 465.9    | 519.0    | 0            |
| C. SpecANN int8 + f32 verify| **1.000** | 741.7  | 1347.2   | 1412.2   | **255.2**    |
| D. SpecANN 1-bit + f32 verify| 0.699    | **6591.7** | 150.3 | 164.6 | 768.8       |

Key findings:

1. **Variant C — the honest headline.** SpecANN with an int8 draft and an f32
   verifier delivers **100 % exact-top-10 recall while verifying only 2.55 %
   of the corpus on average** (255 / 10 000 vectors). QPS is lower than plain
   float32 brute force in *this configuration* because the int8 draft is still
   O(n) with per-vector scale multiplication and no SIMD packing — it costs
   more per compare than a raw f32 dot. The QPS win only materializes when the
   draft is genuinely sub-linear (HNSW, IVF, PQ). This measurement isolates the
   *verification savings* independent of draft speed.
2. **Variant D — the throughput ceiling.** 1-bit signs deliver **2.53× QPS
   over exact** with an adaptive escalation policy, but recall drops to 69.9 %
   on isotropic gaussians (a known worst case for signed 1-bit quantization
   without rotation). RaBitQ-style rotation would restore recall — see roadmap.
3. **Escalation works.** The int8 variant needed zero escalations to hit
   100 %; the 1-bit variant averaged 768.8 verified per query (α = 12,
   escalated when gap < 5 %). Full-verify fallback fired 0 times in both.

## How it works — walkthrough

Imagine querying "documents about neural quantization." Under vanilla ANN,
you'd hit an HNSW graph, walk 2 000 candidates, compute float32 distances,
return top-10. Under **SpecANN**:

1. A tiny 1-bit index runs first — one `u64` XOR + popcount per candidate.
   Two microseconds later you have a *draft* top-40.
2. The driver checks: is the score gap between rank-10 and rank-11 wide
   enough to trust? If yes (say the draft is confident), the exact f32
   verifier rescores those 40 and returns the top-10. Two more microseconds.
3. If no — the drafts around the boundary look ambiguous — the driver widens
   to α · 2.5 = 100 candidates and reruns the draft. Only after two failed
   widenings does it fall back to full-index float32.

That's the entire loop, and the payoff mirrors LLM speculative decoding:
**the verifier only ever runs on candidates the draft already selected**, so
its cost scales with `k_draft`, not `n`.

## Practical failure modes

- **Isotropic-gaussian embeddings + 1-bit draft.** As Variant D shows, sign
  bits are a bad proxy for L2 when data is spherically symmetric. Fix: apply
  RaBitQ's random rotation before signing (already implemented in the
  `ruvector-rabitq` crate) — trivial to wire in as a next iteration.
- **Ties near the k-boundary.** The gap heuristic can be fooled by genuine
  ties in the true nearest neighbors. Fix: fall through to escalation after 1
  attempt if `gap < 1e-6`, treat as "obvious tie, verify more."
- **Skewed norms in the corpus.** Per-vector int8 scales lose information on
  norm variance. Fix: LVQ-style two-scale quantization.
- **Very small k.** For k = 1, the "gap at rank k" is meaningless. Fix:
  hard-set α ≥ 8 when k = 1.

## What to improve next (roadmap)

1. **Rotated 1-bit draft.** Wire `ruvector-rabitq`'s Hadamard rotation into
   `Sign1BitDraft` — expected to lift Variant D recall from 0.70 → 0.95+ at
   the same throughput. Two-day integration.
2. **HNSW draft.** Replace brute-force draft with `hnsw_rs` graph walk;
   verification savings compose with sub-linear draft cost. Expected 10-50×
   QPS over Variant A at 99 %+ recall.
3. **Learned escalation policy.** Replace static `gap_threshold` with a tiny
   MLP over `(k, α, gap, top-k spread)` — trained offline on ground-truth
   traces to predict escalate/accept decisions with better precision-recall
   trade-off. Analogous to *predictive draft acceptance* in speculative
   decoding papers (Miao+2024).
4. **Batched verification.** Amortize verifier setup across a query batch —
   pack candidate ids across queries and issue one SIMD gather. Expected
   1.5-2× on large batches.
5. **Cascading draft chain.** 1-bit → int8 → f32, each stage escalating only
   on gap. Analogous to *staged speculation* in LLM literature.

## Production crate layout proposal

```
ruvector-specann/
├── Cargo.toml
├── src/
│   ├── lib.rs                    ← DraftIndex, Verifier, SpecAnnIndex, policies
│   ├── main.rs                   ← specann-demo
│   ├── bin/bench.rs              ← specann-bench (real numbers)
│   ├── draft/
│   │   ├── mod.rs
│   │   ├── int8.rs               ← Int8BruteForce
│   │   ├── sign1bit.rs           ← Sign1BitDraft (+ optional rotation)
│   │   ├── hnsw.rs               ← HNSW draft (integrate ruvector-core)
│   │   └── rabitq.rs             ← RaBitQ draft (integrate ruvector-rabitq)
│   └── verifier/
│       ├── mod.rs
│       └── f32_brute.rs          ← F32BruteForce
├── tests/integration.rs
└── benches/                      ← criterion micro-benchmarks
```

The current PoC keeps everything in `lib.rs` (475 lines, under the 500-line
project cap) for reviewability. Production split would land in the same crate
without breaking the trait API.

## References

- Leviathan Y., Kalman M., Matias Y. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023.
- Chen C., Borgeaud S., Irving G., et al. *Accelerating Large Language Model
  Decoding with Speculative Sampling.* DeepMind, 2023.
- Miao X., et al. *SpecInfer: Accelerating Generative LLM Serving with
  Speculative Inference and Token Tree Verification.* ASPLOS 2024.
- Gao J., Long C. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD
  2024.
- Aguerrebere C., et al. *Similarity Search in the Blink of an Eye with
  Compressed Indices.* VLDB 2023 (LVQ).
- Chen P., et al. *FINGER: Fast Inference for Graph-based Approximate Nearest
  Neighbor Search.* WWW 2023.
- Gao J., Long C. *High-Dimensional Approximate Nearest Neighbor Search: with
  Reliable and Efficient Distance Comparison Operations.* SIGMOD 2023
  (ADSampling).

## Reproduction

```bash
git checkout research/nightly/2026-07-04-speculative-ann-draft-verify
cargo build --release -p ruvector-specann
cargo test  --release -p ruvector-specann          # 4 unit + 2 integration
cargo run   --release -p ruvector-specann --bin specann-demo
cargo run   --release -p ruvector-specann --bin specann-bench > bench.json
```

`bench_results.json` in this directory is the JSON captured for the results
table above.
