//! NN-Descent bulk kNN graph construction (Dong, Charikar & Li, WWW 2011).
//!
//! Two flavors, gated by `NnDescentConfig::use_reverse`:
//!
//! 1. **Basic** — each iteration explores pairs *within* each point's
//!    current neighbor list. Empirically gets ~70-90% recall@K in
//!    O(N · K²) work per iter.
//!
//! 2. **Local-join (with reverse lists)** — the canonical NN-Descent.
//!    Each point's expansion set is `B[u] ∪ R[u]` where R is the
//!    reverse kNN graph. Pairs are formed between (new × new∪old).
//!    Recall converges 5-15 percentage points higher at the same K
//!    in fewer iterations.
//!
//! Early termination triggers when the number of insertions in an
//! iteration drops below `delta * N * K`.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::distance::DistanceCounter;
use crate::knn_graph::{KnnGraph, KnnGraphBuilder, KnnNeighbor};

#[derive(Clone, Debug)]
pub struct NnDescentConfig {
    /// Sample rate ρ for new/old neighbor sampling. Typical: 0.5.
    pub rho: f32,
    /// Early-termination ratio δ. Typical: 1e-3.
    pub delta: f32,
    /// Hard iteration cap.
    pub max_iters: usize,
    /// If true, expand pairs over B[u] ∪ R[u]; else only B[u].
    pub use_reverse: bool,
    /// Seed for RNG used in initialization and sampling.
    pub seed: u64,
}

impl Default for NnDescentConfig {
    fn default() -> Self {
        Self {
            rho: 0.5,
            delta: 1e-3,
            max_iters: 30,
            use_reverse: true,
            seed: 0xC0FFEE,
        }
    }
}

pub struct NnDescentBuilder {
    pub cfg: NnDescentConfig,
}

impl NnDescentBuilder {
    pub fn new(cfg: NnDescentConfig) -> Self {
        Self { cfg }
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    id: u32,
    dist: f32,
    is_new: bool,
}

impl Slot {
    const SENTINEL: Slot = Slot {
        id: u32::MAX,
        dist: f32::INFINITY,
        is_new: false,
    };
}

impl KnnGraphBuilder for NnDescentBuilder {
    fn name(&self) -> &'static str {
        if self.cfg.use_reverse { "NnDescent-LocalJoin" } else { "NnDescent-Basic" }
    }

    fn build(&self, vectors: &[Vec<f32>], k: usize, counter: &DistanceCounter) -> KnnGraph {
        let n = vectors.len();
        let mut rng = StdRng::seed_from_u64(self.cfg.seed);

        // ---- Init: random neighbors, all marked new ----
        let mut b: Vec<Vec<Slot>> = vec![Vec::with_capacity(k); n];
        for u in 0..n {
            let mut chosen: HashSet<u32> = HashSet::new();
            chosen.insert(u as u32);
            while b[u].len() < k {
                let v = rng.gen_range(0..n) as u32;
                if !chosen.insert(v) {
                    continue;
                }
                let d = counter.measure(&vectors[u], &vectors[v as usize]);
                b[u].push(Slot { id: v, dist: d, is_new: true });
            }
            // sort ascending by dist
            b[u].sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        }

        let delta_threshold = ((self.cfg.delta as f64) * (n as f64) * (k as f64)).max(1.0) as u64;

        for iter in 0..self.cfg.max_iters {
            // Build new[u] / old[u] partitions
            let mut new_set: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut old_set: Vec<Vec<u32>> = vec![Vec::new(); n];
            for u in 0..n {
                for s in &b[u] {
                    if s.is_new && rng.gen::<f32>() < self.cfg.rho {
                        new_set[u].push(s.id);
                    } else if !s.is_new {
                        old_set[u].push(s.id);
                    }
                }
            }
            // Flip newly-sampled new flags so we don't re-process them next iter.
            // (Anything that was new and *not* picked this iter stays new.)
            for u in 0..n {
                let picked: HashSet<u32> = new_set[u].iter().copied().collect();
                for s in b[u].iter_mut() {
                    if s.is_new && picked.contains(&s.id) {
                        s.is_new = false;
                    }
                }
            }

            // Build reverse lists if requested
            if self.cfg.use_reverse {
                let mut r_new: Vec<Vec<u32>> = vec![Vec::new(); n];
                let mut r_old: Vec<Vec<u32>> = vec![Vec::new(); n];
                for u in 0..n {
                    for &v in &new_set[u] {
                        r_new[v as usize].push(u as u32);
                    }
                    for &v in &old_set[u] {
                        r_old[v as usize].push(u as u32);
                    }
                }
                // Subsample reverse lists at rate ρ, cap at ρ·K
                let cap = ((self.cfg.rho * k as f32) as usize).max(1);
                for u in 0..n {
                    sample_into(&r_new[u], cap, &mut rng, &mut new_set[u]);
                    sample_into(&r_old[u], cap, &mut rng, &mut old_set[u]);
                    dedup(&mut new_set[u]);
                    dedup(&mut old_set[u]);
                }
            }

            // Local join: for each u, all (p ∈ new[u]) × (q ∈ new[u] ∪ old[u]), p < q
            let mut update_count: u64 = 0;
            for u in 0..n {
                let new_u = &new_set[u];
                let old_u = &old_set[u];
                for i in 0..new_u.len() {
                    let p = new_u[i];
                    // pairs within new[u]: only i<j to avoid double-work
                    for j in (i + 1)..new_u.len() {
                        let q = new_u[j];
                        if p == q { continue; }
                        let d = counter.measure(&vectors[p as usize], &vectors[q as usize]);
                        update_count += try_insert(&mut b[p as usize], q, d, k) as u64;
                        update_count += try_insert(&mut b[q as usize], p, d, k) as u64;
                    }
                    // pairs new[u] × old[u]
                    for &q in old_u {
                        if p == q { continue; }
                        let d = counter.measure(&vectors[p as usize], &vectors[q as usize]);
                        update_count += try_insert(&mut b[p as usize], q, d, k) as u64;
                        update_count += try_insert(&mut b[q as usize], p, d, k) as u64;
                    }
                }
            }

            if update_count < delta_threshold {
                break;
            }
            let _ = iter;
        }

        // Materialize KnnGraph
        let mut g = KnnGraph::new(n, k);
        for u in 0..n {
            // Filter self-loops (shouldn't exist but safety)
            let mut neigh: Vec<KnnNeighbor> = b[u].iter()
                .filter(|s| s.id as usize != u && s.id != u32::MAX)
                .map(|s| KnnNeighbor { id: s.id, dist: s.dist })
                .collect();
            neigh.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            neigh.truncate(k);
            g.neighbors[u] = neigh;
        }
        g
    }
}

/// Insert (id,dist) into sorted slot buffer if it improves top-k.
/// Returns true iff inserted.
fn try_insert(buf: &mut Vec<Slot>, id: u32, dist: f32, k: usize) -> bool {
    // Reject self / duplicates
    for s in buf.iter() {
        if s.id == id { return false; }
    }
    if buf.len() < k {
        buf.push(Slot { id, dist, is_new: true });
        // bubble into place
        let mut i = buf.len() - 1;
        while i > 0 && buf[i - 1].dist > buf[i].dist {
            buf.swap(i - 1, i);
            i -= 1;
        }
        return true;
    }
    if dist >= buf[k - 1].dist {
        return false;
    }
    buf[k - 1] = Slot { id, dist, is_new: true };
    let mut i = k - 1;
    while i > 0 && buf[i - 1].dist > buf[i].dist {
        buf.swap(i - 1, i);
        i -= 1;
    }
    true
}

fn sample_into(src: &[u32], cap: usize, rng: &mut StdRng, dst: &mut Vec<u32>) {
    if src.len() <= cap {
        dst.extend_from_slice(src);
        return;
    }
    // reservoir sample
    for (i, &v) in src.iter().enumerate() {
        if dst.len() < cap {
            dst.push(v);
        } else {
            let j = rng.gen_range(0..=i);
            if j < cap {
                dst[j] = v;
            }
        }
    }
}

fn dedup(v: &mut Vec<u32>) {
    let mut seen = HashSet::new();
    v.retain(|&x| seen.insert(x));
}

// Silence unused-import warning if Slot::SENTINEL not used externally.
#[allow(dead_code)]
fn _touch() { let _ = Slot::SENTINEL; let _ = KnnNeighbor::SENTINEL; }
