//! Core DEG graph: regular d-out-degree, single-layer, with insert / delete
//! / search and an edge-optimisation pass approximating the RNG (Relative
//! Neighbourhood Graph).
//!
//! Implementation notes:
//!   * Vectors are stored contiguously in a `Vec<f32>` so distance kernels
//!     stay cache-friendly.
//!   * Adjacency is a flat `Vec<u32>` of length `n * d`; row `i` lives at
//!     `[i*d .. i*d + d]`. This keeps neighbour iteration branch-free.
//!   * Sentinel `INVALID = u32::MAX` marks free slots in the adjacency row
//!     (used during construction before the row is full).

use crate::distance::Metric;
use crate::search::{search, SearchResult};
use rand::Rng;

pub type NodeId = u32;
const INVALID: NodeId = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct DegConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Out-degree per node. Typical 16–32 (DEG paper uses d=30 by default).
    pub edges_per_node: usize,
    /// Search width during insertion / optimisation. Typical 60–200.
    pub eps_insert: usize,
}

impl Default for DegConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            edges_per_node: 24,
            eps_insert: 80,
        }
    }
}

pub struct Deg {
    cfg: DegConfig,
    vectors: Vec<f32>,
    adj: Vec<NodeId>,
    n: usize,
    entry: NodeId,
}

impl Deg {
    pub fn new(cfg: DegConfig) -> Self {
        Self {
            cfg,
            vectors: Vec::new(),
            adj: Vec::new(),
            n: 0,
            entry: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn entry(&self) -> NodeId {
        self.entry
    }
    pub fn config(&self) -> &DegConfig {
        &self.cfg
    }

    #[inline]
    pub fn vector(&self, id: NodeId) -> &[f32] {
        let i = id as usize;
        let d = self.cfg.dim;
        &self.vectors[i * d..i * d + d]
    }

    #[inline]
    pub fn neighbours(&self, id: NodeId) -> &[NodeId] {
        let i = id as usize;
        let k = self.cfg.edges_per_node;
        let row = &self.adj[i * k..i * k + k];
        // Truncate at first INVALID for sparse early rows.
        let mut end = k;
        for (idx, v) in row.iter().enumerate() {
            if *v == INVALID {
                end = idx;
                break;
            }
        }
        &row[..end]
    }

    #[inline]
    fn neighbours_mut(&mut self, id: NodeId) -> &mut [NodeId] {
        let i = id as usize;
        let k = self.cfg.edges_per_node;
        &mut self.adj[i * k..i * k + k]
    }

    fn push_vec(&mut self, v: &[f32]) -> NodeId {
        assert_eq!(v.len(), self.cfg.dim, "vector dim mismatch");
        let id = self.n as NodeId;
        self.vectors.extend_from_slice(v);
        self.adj
            .extend(std::iter::repeat(INVALID).take(self.cfg.edges_per_node));
        self.n += 1;
        id
    }

    /// Bulk-build from a flat vectors buffer. Falls back to per-element
    /// `insert` so it stays small + correct; if you need throughput, see
    /// the `examples/build_recall.rs` benchmark for a parallel pattern.
    pub fn build<M: Metric>(&mut self, vectors: &[f32]) {
        let d = self.cfg.dim;
        let n = vectors.len() / d;
        for i in 0..n {
            self.insert::<M>(&vectors[i * d..(i + 1) * d]);
        }
    }

    /// Insert one vector. Returns its assigned NodeId.
    pub fn insert<M: Metric>(&mut self, v: &[f32]) -> NodeId {
        if self.n == 0 {
            let id = self.push_vec(v);
            self.entry = id;
            return id;
        }
        let cfg = self.cfg;
        // Step 1: greedy search for candidate neighbours from the entry.
        let cands = search::<M>(self, v, cfg.edges_per_node, cfg.eps_insert, self.entry);
        let new_id = self.push_vec(v);

        // Step 2: pick edges using RNG-style pruning.
        let chosen = self.rng_prune::<M>(v, &cands, cfg.edges_per_node);
        {
            let row = self.neighbours_mut(new_id);
            for (slot, sr) in row.iter_mut().zip(chosen.iter()) {
                *slot = sr.id;
            }
        }

        // Step 3: back-link each chosen neighbour, evicting a worst edge if
        // the row is already full ("edge-optimisation" step).
        for sr in &chosen {
            self.try_add_back_edge::<M>(sr.id, new_id);
        }

        // Step 4: keep entry as the most-connected node (in practice the
        // first inserted). For balance, we re-pick entry every 1024 inserts
        // to a random node — cheap and avoids one node hogging traffic.
        if self.n.is_power_of_two() {
            let mut rng = rand::thread_rng();
            self.entry = rng.gen_range(0..self.n as u32);
        }
        new_id
    }

    /// RNG-style pruning: greedily accept nearest candidate; reject any
    /// later candidate that is closer to an accepted edge than to the
    /// query (the textbook RNG occlusion test).
    fn rng_prune<M: Metric>(
        &self,
        query: &[f32],
        cands: &[SearchResult],
        budget: usize,
    ) -> Vec<SearchResult> {
        let mut chosen: Vec<SearchResult> = Vec::with_capacity(budget);
        for c in cands {
            if chosen.len() == budget {
                break;
            }
            let mut occluded = false;
            for ex in &chosen {
                let d_ex = M::dist(self.vector(c.id), self.vector(ex.id));
                if d_ex < c.dist {
                    occluded = true;
                    break;
                }
            }
            if !occluded {
                chosen.push(*c);
            }
            let _ = query; // suppress unused warning on metrics that ignore q
        }
        chosen
    }

    /// Add `new` to `target`'s adjacency. If the row is full, evict the
    /// worst (longest) edge if `new` is closer than that worst.
    fn try_add_back_edge<M: Metric>(&mut self, target: NodeId, new: NodeId) {
        let k = self.cfg.edges_per_node;
        // Copy `target`'s vector so we can borrow `self` mutably while still
        // computing distances against it. Cheap: one vector, ~512 B at d=128.
        let target_vec: Vec<f32> = self.vector(target).to_vec();
        let new_dist = M::dist(&target_vec, self.vector(new));

        // Find empty slot first.
        let mut empty: Option<usize> = None;
        let mut worst_idx: usize = 0;
        let mut worst_dist: f32 = f32::MIN;
        let row_start = target as usize * k;
        for s in 0..k {
            let nb = self.adj[row_start + s];
            if nb == INVALID {
                empty = Some(s);
                break;
            }
            if nb == new {
                return; // edge already exists
            }
            let d = M::dist(&target_vec, self.vector(nb));
            if d > worst_dist {
                worst_dist = d;
                worst_idx = s;
            }
        }
        if let Some(s) = empty {
            self.adj[row_start + s] = new;
            return;
        }
        if new_dist < worst_dist {
            self.adj[row_start + worst_idx] = new;
        }
    }

    /// Delete one node by swap-removal. The donor (last node) is moved
    /// into the deleted node's slot, and its neighbours are re-stitched.
    /// O(d^2) — no tombstones, no scan over the graph.
    pub fn delete<M: Metric>(&mut self, victim: NodeId) {
        assert!((victim as usize) < self.n, "victim out of range");
        let last_id = (self.n - 1) as NodeId;
        let k = self.cfg.edges_per_node;
        let d = self.cfg.dim;

        // 1. Pull victim's neighbours' donor list (we'll re-stitch them).
        let donors: Vec<NodeId> = self.neighbours(victim).to_vec();

        if victim != last_id {
            // 2. Move last vector into victim slot.
            let (head, tail) = self.vectors.split_at_mut(last_id as usize * d);
            let src = &tail[..d];
            let dst = &mut head[victim as usize * d..victim as usize * d + d];
            dst.copy_from_slice(src);
            // 3. Move last adjacency row into victim slot.
            for s in 0..k {
                self.adj[victim as usize * k + s] = self.adj[last_id as usize * k + s];
            }
            // 4. Re-label every reference to `last_id` → `victim` in the graph.
            //    Bounded scan: O(n * d) once per delete. Acceptable for the
            //    PoC scale; production would maintain a reverse-index.
            for idx in 0..(self.n - 1) * k {
                if self.adj[idx] == last_id {
                    self.adj[idx] = victim;
                }
            }
        }

        // 5. Trim storage.
        self.vectors.truncate((self.n - 1) * d);
        self.adj.truncate((self.n - 1) * k);
        self.n -= 1;

        // 6. Re-stitch: each donor lost an edge to `victim` (now removed
        //    or relabelled-to-victim). Give each of them a fresh candidate
        //    from `donors \ self`.
        for &donor in &donors {
            // Skip if donor was the victim or got moved out
            if donor == INVALID {
                continue;
            }
            let donor_id = if donor == last_id { victim } else { donor };
            if (donor_id as usize) >= self.n {
                continue;
            }
            // Re-search to find a replacement neighbour for the donor.
            let v: Vec<f32> = self.vector(donor_id).to_vec();
            let entry = if self.entry == victim {
                0
            } else if self.entry == last_id {
                victim
            } else {
                self.entry
            };
            let new_cands = search::<M>(self, &v, k, self.cfg.eps_insert, entry);
            for c in new_cands {
                if c.id == donor_id {
                    continue;
                }
                self.try_add_back_edge::<M>(donor_id, c.id);
            }
        }

        // 7. Fix entry if needed.
        if self.entry == victim {
            self.entry = 0;
        } else if self.entry == last_id {
            self.entry = victim;
        }
        if (self.entry as usize) >= self.n {
            self.entry = 0;
        }
    }

    pub fn query<M: Metric>(&self, q: &[f32], k: usize, eps: usize) -> Vec<SearchResult> {
        if self.n == 0 {
            return Vec::new();
        }
        search::<M>(self, q, k, eps.max(k), self.entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
    }

    fn brute_force(vecs: &[f32], q: &[f32], d: usize, k: usize) -> Vec<NodeId> {
        let n = vecs.len() / d;
        let mut all: Vec<(f32, NodeId)> = (0..n)
            .map(|i| (L2::dist(&vecs[i * d..(i + 1) * d], q), i as NodeId))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[test]
    fn build_and_query_recall() {
        let d = 32;
        let n = 500;
        let vecs = random_vectors(n, d, 42);
        let cfg = DegConfig { dim: d, edges_per_node: 16, eps_insert: 60 };
        let mut deg = Deg::new(cfg);
        deg.build::<L2>(&vecs);
        assert_eq!(deg.len(), n);

        // Query the first 50 stored vectors — recall@10 should be near-perfect.
        let mut hits = 0;
        for i in 0..50 {
            let q = &vecs[i * d..(i + 1) * d];
            let truth = brute_force(&vecs, q, d, 10);
            let got: Vec<NodeId> = deg
                .query::<L2>(q, 10, 60)
                .into_iter()
                .map(|r| r.id)
                .collect();
            for t in &truth {
                if got.contains(t) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f32 / (50.0 * 10.0);
        assert!(recall > 0.85, "recall {recall} below 0.85");
    }

    #[test]
    fn delete_preserves_recall() {
        let d = 16;
        let n = 200;
        let vecs = random_vectors(n, d, 7);
        let cfg = DegConfig { dim: d, edges_per_node: 12, eps_insert: 40 };
        let mut deg = Deg::new(cfg);
        deg.build::<L2>(&vecs);

        // Delete 50 nodes.
        for _ in 0..50 {
            let v = (deg.len() / 2) as NodeId;
            deg.delete::<L2>(v);
        }
        assert_eq!(deg.len(), 150);

        // The remaining graph must still find self-queries for survivors.
        // Reconstruct the surviving vector buffer (deletes reordered ids).
        let mut survive: Vec<f32> = Vec::with_capacity(deg.len() * d);
        for i in 0..deg.len() as NodeId {
            survive.extend_from_slice(deg.vector(i));
        }
        let mut hits = 0;
        let trials = 30;
        for i in 0..trials {
            let q = &survive[i * d..(i + 1) * d];
            let truth = brute_force(&survive, q, d, 5);
            let got: Vec<NodeId> = deg
                .query::<L2>(q, 5, 40)
                .into_iter()
                .map(|r| r.id)
                .collect();
            for t in &truth {
                if got.contains(t) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f32 / (trials as f32 * 5.0);
        assert!(recall > 0.7, "post-delete recall {recall} below 0.7");
    }
}
