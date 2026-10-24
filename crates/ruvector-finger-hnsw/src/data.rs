//! Deterministic synthetic dataset + brute-force oracles.

/// splitmix64-based deterministic f32 generator. Returns values in roughly
/// a standard normal shape by summing 12 uniforms (CLT). Fine for ANN
/// benchmarking where the distribution shape matters more than exact
/// Gaussianity.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f32) * (1.0 / ((1u64 << 53) as f32))
    }

    /// Standard-normal-ish via 12-uniform CLT (zero mean, unit variance).
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

/// Generate `n` vectors of dimension `d` with mild clusterable structure:
/// each vector is a cluster centroid plus Gaussian noise. `cluster_sep`
/// controls inter-centroid spread relative to intra-cluster noise (unit
/// variance). A small `cluster_sep` (e.g. 0.8) keeps clusters overlapping
/// so a flat kNN graph stays connected — important because this PoC
/// uses a flat kNN graph instead of layered HNSW.
pub fn gen_dataset(n: usize, d: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    gen_dataset_with_sep(n, d, clusters, 0.8, seed)
}

pub fn gen_dataset_with_sep(n: usize, d: usize, clusters: usize, cluster_sep: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let clusters = clusters.max(1);

    let mut centers = Vec::with_capacity(clusters);
    for _ in 0..clusters {
        let mut c = vec![0.0f32; d];
        for v in c.iter_mut() {
            *v = rng.gauss() * cluster_sep;
        }
        centers.push(c);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % clusters];
        let mut v = c.clone();
        for x in v.iter_mut() {
            *x += rng.gauss();
        }
        out.push(v);
    }
    out
}

/// Generate `n_queries` queries by picking data points at random and
/// perturbing them with Gaussian noise. This produces queries that lie in
/// the manifold of the dataset — the realistic scenario for ANN.
pub fn gen_queries_from_data(data: &[Vec<f32>], n_queries: usize, noise: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let n = data.len();
    let mut out = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let i = (rng.next_u64() as usize) % n;
        let mut q = data[i].clone();
        for v in q.iter_mut() {
            *v += rng.gauss() * noise;
        }
        out.push(q);
    }
    out
}

/// Exact squared L2 distance.
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

/// Brute-force top-k nearest neighbors (by squared L2). Ids returned in
/// ascending-distance order. Oracle used to compute recall.
pub fn brute_force_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = (0..data.len() as u32)
        .map(|i| (i, l2_sq(&data[i as usize], q)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}

/// recall@k against ground truth. Both inputs are id-lists of equal length k.
pub fn recall_at_k(gt: &[u32], got: &[u32]) -> f32 {
    if gt.is_empty() {
        return 0.0;
    }
    let mut hit = 0;
    for g in gt {
        if got.contains(g) {
            hit += 1;
        }
    }
    hit as f32 / gt.len() as f32
}
