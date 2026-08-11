//! Swappable Seeder trait + three implementations.

use crate::{kmeans::KMeans, sqdist, Rng};

pub trait Seeder {
    /// Fit on the dataset (called once, offline).
    fn fit(&mut self, data: &[Vec<f32>]);
    /// Provide entry points for a given query.
    fn entry_points(&self, query: &[f32]) -> Vec<usize>;
    fn name(&self) -> &'static str;
}

/// Baseline: fixed random entry (mirrors classic HNSW default).
pub struct RandomSeeder {
    pub entry: usize,
    pub seed: u64,
}
impl RandomSeeder {
    pub fn new(seed: u64) -> Self {
        Self { entry: 0, seed }
    }
}
impl Seeder for RandomSeeder {
    fn fit(&mut self, data: &[Vec<f32>]) {
        let mut rng = Rng::new(self.seed);
        self.entry = rng.next_usize(data.len());
    }
    fn entry_points(&self, _q: &[f32]) -> Vec<usize> {
        vec![self.entry]
    }
    fn name(&self) -> &'static str {
        "random"
    }
}

/// Centroid-seeded: pick the medoid of the nearest k-means cluster.
pub struct CentroidSeeder {
    pub k: usize,
    pub iters: usize,
    pub seed: u64,
    pub medoids: Vec<usize>,
    pub km: Option<KMeans>,
}
impl CentroidSeeder {
    pub fn new(k: usize, iters: usize, seed: u64) -> Self {
        Self { k, iters, seed, medoids: vec![], km: None }
    }
}
impl Seeder for CentroidSeeder {
    fn fit(&mut self, data: &[Vec<f32>]) {
        let km = KMeans::fit(data, self.k, self.iters, self.seed);
        // Medoid = data point closest to each centroid.
        let mut medoids = vec![0usize; self.k];
        let mut best_d = vec![f32::INFINITY; self.k];
        for (i, v) in data.iter().enumerate() {
            let c = km.assignments[i] as usize;
            let d = sqdist(v, &km.centroids[c]);
            if d < best_d[c] {
                best_d[c] = d;
                medoids[c] = i;
            }
        }
        self.medoids = medoids;
        self.km = Some(km);
    }
    fn entry_points(&self, q: &[f32]) -> Vec<usize> {
        let km = self.km.as_ref().expect("fit first");
        let nc = km.nearest_centroids(q, 1);
        vec![self.medoids[nc[0]]]
    }
    fn name(&self) -> &'static str {
        "centroid1"
    }
}

/// Multi-centroid: take the M nearest cluster medoids as multi-entry seeds.
pub struct MultiCentroidSeeder {
    pub inner: CentroidSeeder,
    pub m: usize,
}
impl MultiCentroidSeeder {
    pub fn new(k: usize, m: usize, iters: usize, seed: u64) -> Self {
        Self { inner: CentroidSeeder::new(k, iters, seed), m }
    }
}
impl Seeder for MultiCentroidSeeder {
    fn fit(&mut self, data: &[Vec<f32>]) {
        self.inner.fit(data);
    }
    fn entry_points(&self, q: &[f32]) -> Vec<usize> {
        let km = self.inner.km.as_ref().expect("fit first");
        km.nearest_centroids(q, self.m)
            .into_iter()
            .map(|c| self.inner.medoids[c])
            .collect()
    }
    fn name(&self) -> &'static str {
        "centroidM"
    }
}
