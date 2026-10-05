//! Online IVF centroid refresh via reservoir sampling.
//!
//! Streaming IVF indexes suffer from centroid drift when the data distribution
//! shifts after initial training. This crate maintains a bounded reservoir of
//! representative samples and periodically re-fits centroids with warm-start
//! k-means, correcting drift without a full retrain.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub type Vector = Vec<f32>;

#[derive(Clone, Copy, Debug)]
pub enum RefreshPolicy {
    /// Never refresh (baseline — classic static IVF).
    Static,
    /// Refresh every N inserts (fixed cadence).
    Periodic(usize),
    /// Refresh when drift score exceeds threshold.
    DriftTriggered(f32),
}

#[derive(Clone, Debug)]
pub struct IvfConfig {
    pub dim: usize,
    pub n_centroids: usize,
    pub reservoir_capacity: usize,
    pub refresh_iters: usize,
    pub policy: RefreshPolicy,
    pub seed: u64,
}

impl IvfConfig {
    pub fn new(dim: usize, n_centroids: usize) -> Self {
        Self {
            dim,
            n_centroids,
            reservoir_capacity: n_centroids * 50,
            refresh_iters: 5,
            policy: RefreshPolicy::DriftTriggered(0.15),
            seed: 0xA5A5_5A5A,
        }
    }
}

pub struct OnlineIvf {
    pub cfg: IvfConfig,
    pub centroids: Vec<Vector>,
    reservoir: Vec<Vector>,
    rng: ChaCha8Rng,
    seen: u64,
    inserts_since_refresh: usize,
    last_drift: f32,
    cached_radius: f32,
    pub refresh_count: u64,
}

impl OnlineIvf {
    pub fn new(cfg: IvfConfig, initial_training: &[Vector]) -> Self {
        assert!(cfg.n_centroids > 0);
        assert!(!initial_training.is_empty());
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let centroids = kmeans_pp_init(initial_training, cfg.n_centroids, &mut rng);
        let centroids = lloyd(&centroids, initial_training, cfg.refresh_iters);
        let mut reservoir = Vec::with_capacity(cfg.reservoir_capacity);
        for v in initial_training.iter().take(cfg.reservoir_capacity) {
            reservoir.push(v.clone());
        }
        let mut this = Self {
            cfg,
            centroids,
            reservoir,
            rng,
            seen: initial_training.len() as u64,
            inserts_since_refresh: 0,
            last_drift: 0.0,
            cached_radius: 1.0,
            refresh_count: 0,
        };
        this.cached_radius = this.mean_radius().max(1e-6);
        this
    }

    /// Insert a vector into the index. Returns the assigned centroid id.
    pub fn insert(&mut self, v: &[f32]) -> usize {
        assert_eq!(v.len(), self.cfg.dim);
        self.seen += 1;
        self.inserts_since_refresh += 1;
        self.reservoir_push(v);
        let cid = self.assign(v);
        if self.should_refresh(v) {
            self.refresh();
        }
        cid
    }

    pub fn assign(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = l2_sq(v, c);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    pub fn drift_score(&self) -> f32 {
        self.last_drift
    }

    pub fn reservoir_len(&self) -> usize {
        self.reservoir.len()
    }

    fn reservoir_push(&mut self, v: &[f32]) {
        if self.reservoir.len() < self.cfg.reservoir_capacity {
            self.reservoir.push(v.to_vec());
            return;
        }
        // Classic Algorithm R: replace with probability k/n.
        let n = self.seen; // 1-indexed count of items seen.
        let j = self.rng.gen_range(0..n);
        if (j as usize) < self.cfg.reservoir_capacity {
            self.reservoir[j as usize].copy_from_slice(v);
        }
    }

    fn should_refresh(&mut self, v: &[f32]) -> bool {
        match self.cfg.policy {
            RefreshPolicy::Static => false,
            RefreshPolicy::Periodic(n) => self.inserts_since_refresh >= n,
            RefreshPolicy::DriftTriggered(thr) => {
                let d = self.min_dist(v).sqrt();
                // O(k) per insert: EMA against the cached mean radius (recomputed
                // after each refresh, not every insert).
                self.last_drift = 0.95 * self.last_drift + 0.05 * d;
                let cooldown =
                    (self.cfg.reservoir_capacity / 2).max(self.cfg.n_centroids * 16);
                if self.inserts_since_refresh < cooldown {
                    return false;
                }
                self.last_drift > (1.0 + thr) * self.cached_radius
            }
        }
    }

    fn min_dist(&self, v: &[f32]) -> f32 {
        let mut best = f32::INFINITY;
        for c in &self.centroids {
            let d = l2_sq(v, c);
            if d < best {
                best = d;
            }
        }
        best
    }

    /// Mean nearest-centroid distance across the reservoir.
    fn mean_radius(&self) -> f32 {
        if self.reservoir.is_empty() {
            return 1.0;
        }
        let mut sum = 0.0;
        for v in &self.reservoir {
            sum += self.min_dist(v).sqrt();
        }
        sum / self.reservoir.len() as f32
    }

    /// Warm-start Lloyd on the reservoir, keeping current centroids as init.
    pub fn refresh(&mut self) {
        self.centroids = lloyd(&self.centroids, &self.reservoir, self.cfg.refresh_iters);
        self.inserts_since_refresh = 0;
        self.last_drift = 0.0;
        self.cached_radius = self.mean_radius().max(1e-6);
        self.refresh_count += 1;
    }

    /// Compute inertia (sum of squared nearest-centroid distances) on a set.
    pub fn inertia(&self, data: &[Vector]) -> f64 {
        let mut s = 0.0f64;
        for v in data {
            s += self.min_dist(v) as f64;
        }
        s
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn kmeans_pp_init(data: &[Vector], k: usize, rng: &mut ChaCha8Rng) -> Vec<Vector> {
    let n = data.len();
    assert!(n >= k);
    let mut centers = Vec::with_capacity(k);
    centers.push(data[rng.gen_range(0..n)].clone());
    let mut d2 = vec![f32::INFINITY; n];
    for _ in 1..k {
        let mut sum = 0.0f64;
        for i in 0..n {
            let dd = l2_sq(&data[i], centers.last().unwrap());
            if dd < d2[i] {
                d2[i] = dd;
            }
            sum += d2[i] as f64;
        }
        let mut target = rng.gen_range(0.0..sum.max(f64::MIN_POSITIVE));
        let mut pick = n - 1;
        for i in 0..n {
            target -= d2[i] as f64;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        centers.push(data[pick].clone());
    }
    centers
}

fn lloyd(init: &[Vector], data: &[Vector], iters: usize) -> Vec<Vector> {
    let k = init.len();
    let dim = init[0].len();
    let mut centers: Vec<Vector> = init.to_vec();
    for _ in 0..iters {
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0u64; k];
        for v in data {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (i, c) in centers.iter().enumerate() {
                let d = l2_sq(v, c);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            counts[best] += 1;
            for j in 0..dim {
                sums[best][j] += v[j] as f64;
            }
        }
        for i in 0..k {
            if counts[i] > 0 {
                let c = counts[i] as f64;
                for j in 0..dim {
                    centers[i][j] = (sums[i][j] / c) as f32;
                }
            }
        }
    }
    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_cluster(seed: u64, n: usize, dim: usize, center: f32, spread: f32) -> Vec<Vector> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| center + (rng.gen::<f32>() - 0.5) * spread)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn l2_sq_basic() {
        assert!((l2_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn reservoir_bounded() {
        let dim = 8;
        let train = gen_cluster(1, 200, dim, 0.0, 1.0);
        let mut cfg = IvfConfig::new(dim, 4);
        cfg.reservoir_capacity = 100;
        cfg.policy = RefreshPolicy::Static;
        let mut idx = OnlineIvf::new(cfg, &train);
        for v in gen_cluster(2, 10_000, dim, 5.0, 1.0) {
            idx.insert(&v);
        }
        assert_eq!(idx.reservoir_len(), 100);
    }

    #[test]
    fn drift_refresh_reduces_inertia() {
        let dim = 16;
        let train = gen_cluster(10, 1_000, dim, 0.0, 1.0);
        let shifted = gen_cluster(11, 5_000, dim, 4.0, 1.0);
        let cfg = IvfConfig {
            dim,
            n_centroids: 8,
            reservoir_capacity: 1000,
            refresh_iters: 10,
            policy: RefreshPolicy::DriftTriggered(0.15),
            seed: 42,
        };
        let mut idx = OnlineIvf::new(cfg.clone(), &train);
        let before = idx.inertia(&shifted);
        for v in &shifted {
            idx.insert(v);
        }
        let after = idx.inertia(&shifted);
        assert!(
            after < before * 0.5,
            "expected refresh to cut inertia; before={before:.1} after={after:.1}"
        );
        assert!(idx.refresh_count > 0);
    }

    #[test]
    fn static_policy_never_refreshes() {
        let dim = 8;
        let train = gen_cluster(3, 400, dim, 0.0, 1.0);
        let mut cfg = IvfConfig::new(dim, 4);
        cfg.policy = RefreshPolicy::Static;
        let mut idx = OnlineIvf::new(cfg, &train);
        for v in gen_cluster(4, 1000, dim, 3.0, 1.0) {
            idx.insert(&v);
        }
        assert_eq!(idx.refresh_count, 0);
    }
}
