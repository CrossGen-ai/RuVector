use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

/// A tiny, flat IVF index backed by contiguous posting lists.
///
/// Layout: `vectors` holds all vectors concatenated, dimension `dim`.
/// `posting[i]` is a Vec of global vector ids assigned to centroid `i`.
pub struct IvfIndex {
    pub dim: usize,
    pub centroids: Vec<f32>,
    pub n_lists: usize,
    pub vectors: Vec<f32>,
    pub n_vecs: usize,
    pub posting: Vec<Vec<u32>>,
}

impl IvfIndex {
    #[inline]
    pub fn vec(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.vectors[s..s + self.dim]
    }
    #[inline]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Build a toy IVF index via k-means++ init and a fixed number of Lloyd
/// iterations. Not production-grade clustering; adequate as a benchmark
/// substrate because every scan strategy sees the same partitioning.
pub fn build_ivf(vectors: Vec<f32>, dim: usize, n_lists: usize, lloyd_iters: usize, seed: u64) -> IvfIndex {
    assert!(vectors.len() % dim == 0, "vectors not aligned to dim");
    let n = vectors.len() / dim;
    assert!(n >= n_lists, "need at least n_lists vectors");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    // k-means++ seeding
    let mut centroid_ids: Vec<usize> = Vec::with_capacity(n_lists);
    centroid_ids.push(rng.gen_range(0..n));
    let mut dists = vec![f32::INFINITY; n];
    while centroid_ids.len() < n_lists {
        let last = *centroid_ids.last().unwrap();
        for i in 0..n {
            let d = l2_sq(
                &vectors[i * dim..(i + 1) * dim],
                &vectors[last * dim..(last + 1) * dim],
            );
            if d < dists[i] {
                dists[i] = d;
            }
        }
        let total: f32 = dists.iter().sum();
        let mut r = rng.gen::<f32>() * total.max(1e-12);
        let mut chosen = n - 1;
        for i in 0..n {
            r -= dists[i];
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroid_ids.push(chosen);
    }
    let mut centroids: Vec<f32> = Vec::with_capacity(n_lists * dim);
    for &id in &centroid_ids {
        centroids.extend_from_slice(&vectors[id * dim..(id + 1) * dim]);
    }

    let mut assign = vec![0u32; n];
    for _it in 0..lloyd_iters {
        for i in 0..n {
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..n_lists {
                let d = l2_sq(
                    &vectors[i * dim..(i + 1) * dim],
                    &centroids[c * dim..(c + 1) * dim],
                );
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            assign[i] = best;
        }
        let mut sums = vec![0f32; n_lists * dim];
        let mut counts = vec![0u32; n_lists];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += vectors[i * dim + d];
            }
        }
        for c in 0..n_lists {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }

    let mut posting: Vec<Vec<u32>> = vec![Vec::new(); n_lists];
    for i in 0..n {
        posting[assign[i] as usize].push(i as u32);
    }

    IvfIndex {
        dim,
        centroids,
        n_lists,
        vectors,
        n_vecs: n,
        posting,
    }
}
