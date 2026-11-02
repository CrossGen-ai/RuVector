//! IVF centroid locality reorder.
//!
//! Permutes IVF cluster ids so that cluster ids that are *close* in centroid
//! space are also *close* in integer id space. When a query probes a sorted
//! list of nearest centroids, consecutive probe ids then land in adjacent
//! memory, raising L1/L2 cache hit rates on the probe-list scan.
//!
//! The crate is intentionally minimal: it provides a `FlatIvf` baseline, two
//! reordering strategies (`GreedyChain`, `MortonProjection`), and `search_topk`.
//! Backend details (distance, allocator) are abstracted behind the `Metric`
//! trait so alternative kernels can be plugged in without re-rolling the index.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub trait Metric: Sync + Send {
    fn dist(a: &[f32], b: &[f32]) -> f32;
}

pub struct L2;
impl Metric for L2 {
    #[inline]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}

/// Flat vectors indexed into contiguous arrays, laid out per-cluster.
///
/// Vectors in a cluster live in `vectors[offsets[c]..offsets[c+1]]`. Reordering
/// is a permutation of cluster ids that rewrites `offsets` and the flat
/// vector buffer so cluster `permuted_id` is physically adjacent to its
/// neighbours.
pub struct FlatIvf {
    pub dim: usize,
    pub n_clusters: usize,
    pub centroids: Vec<f32>, // n_clusters * dim
    pub offsets: Vec<usize>, // n_clusters + 1
    pub vectors: Vec<f32>,   // n_vectors * dim, cluster-grouped
    pub ids: Vec<u32>,       // n_vectors, original ids
}

impl FlatIvf {
    pub fn new<R: Rng>(
        rng: &mut R,
        points: &[f32],
        dim: usize,
        n_clusters: usize,
        kmeans_iters: usize,
    ) -> Self {
        let n = points.len() / dim;
        assert_eq!(points.len(), n * dim);

        // Random init: pick n_clusters distinct points.
        let mut idxs: Vec<usize> = (0..n).collect();
        for i in 0..n_clusters {
            let j = i + (rng.gen::<u32>() as usize) % (n - i);
            idxs.swap(i, j);
        }
        let mut centroids = vec![0f32; n_clusters * dim];
        for (ci, &pi) in idxs.iter().take(n_clusters).enumerate() {
            centroids[ci * dim..(ci + 1) * dim]
                .copy_from_slice(&points[pi * dim..(pi + 1) * dim]);
        }

        let mut assign = vec![0u32; n];
        for _ in 0..kmeans_iters {
            // Assign.
            for pi in 0..n {
                let p = &points[pi * dim..(pi + 1) * dim];
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for ci in 0..n_clusters {
                    let c = &centroids[ci * dim..(ci + 1) * dim];
                    let d = L2::dist(p, c);
                    if d < best_d {
                        best_d = d;
                        best = ci as u32;
                    }
                }
                assign[pi] = best;
            }
            // Update.
            let mut counts = vec![0u32; n_clusters];
            let mut sums = vec![0f32; n_clusters * dim];
            for pi in 0..n {
                let c = assign[pi] as usize;
                counts[c] += 1;
                let p = &points[pi * dim..(pi + 1) * dim];
                let s = &mut sums[c * dim..(c + 1) * dim];
                for k in 0..dim {
                    s[k] += p[k];
                }
            }
            for ci in 0..n_clusters {
                if counts[ci] > 0 {
                    let inv = 1.0 / counts[ci] as f32;
                    let c = &mut centroids[ci * dim..(ci + 1) * dim];
                    let s = &sums[ci * dim..(ci + 1) * dim];
                    for k in 0..dim {
                        c[k] = s[k] * inv;
                    }
                }
            }
        }

        // Group vectors by cluster.
        let mut counts = vec![0usize; n_clusters];
        for &a in &assign {
            counts[a as usize] += 1;
        }
        let mut offsets = vec![0usize; n_clusters + 1];
        for ci in 0..n_clusters {
            offsets[ci + 1] = offsets[ci] + counts[ci];
        }
        let mut cursor = offsets.clone();
        let mut vectors = vec![0f32; n * dim];
        let mut ids = vec![0u32; n];
        for pi in 0..n {
            let c = assign[pi] as usize;
            let dst = cursor[c];
            cursor[c] += 1;
            vectors[dst * dim..(dst + 1) * dim]
                .copy_from_slice(&points[pi * dim..(pi + 1) * dim]);
            ids[dst] = pi as u32;
        }

        FlatIvf {
            dim,
            n_clusters,
            centroids,
            offsets,
            vectors,
            ids,
        }
    }

    pub fn n_vectors(&self) -> usize {
        self.ids.len()
    }

    /// Rank clusters by centroid distance to the query. Returns `nprobe` ids.
    pub fn rank_centroids(&self, q: &[f32], nprobe: usize) -> Vec<u32> {
        let mut cand: Vec<(f32, u32)> = (0..self.n_clusters as u32)
            .map(|ci| {
                let c = &self.centroids[ci as usize * self.dim..(ci as usize + 1) * self.dim];
                (L2::dist(q, c), ci)
            })
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cand.truncate(nprobe);
        // KEY: when the index is locality-reordered, sorting these ids puts
        // physically adjacent clusters next to each other in iteration order,
        // which is where the cache win comes from.
        let mut probes: Vec<u32> = cand.into_iter().map(|(_, i)| i).collect();
        probes.sort_unstable();
        probes
    }

    /// Scan the probed clusters, keep top-k by distance.
    pub fn search_topk(&self, q: &[f32], nprobe: usize, k: usize) -> Vec<(f32, u32)> {
        let probes = self.rank_centroids(q, nprobe);
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for ci in probes {
            let s = self.offsets[ci as usize];
            let e = self.offsets[ci as usize + 1];
            for pi in s..e {
                let v = &self.vectors[pi * self.dim..(pi + 1) * self.dim];
                let d = L2::dist(q, v);
                if heap.len() < k {
                    heap.push((d, self.ids[pi]));
                    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else if d < heap.last().unwrap().0 {
                    heap[k - 1] = (d, self.ids[pi]);
                    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }
        }
        heap
    }
}

/// Reordering strategy trait: produce a permutation `new_from_old[old] = new`.
pub trait Reorder {
    fn permutation(centroids: &[f32], dim: usize, n_clusters: usize) -> Vec<u32>;
}

/// Greedy nearest-neighbor chain: start at cluster 0, each step pick the
/// unvisited centroid closest to the current one. Produces a Hamiltonian path
/// where adjacent-in-id clusters are adjacent-in-space.
pub struct GreedyChain;
impl Reorder for GreedyChain {
    fn permutation(centroids: &[f32], dim: usize, n_clusters: usize) -> Vec<u32> {
        let mut visited = vec![false; n_clusters];
        let mut order = Vec::with_capacity(n_clusters);
        let mut cur = 0usize;
        visited[0] = true;
        order.push(0u32);
        for _ in 1..n_clusters {
            let c_cur = &centroids[cur * dim..(cur + 1) * dim];
            let mut best = usize::MAX;
            let mut best_d = f32::INFINITY;
            for ci in 0..n_clusters {
                if visited[ci] {
                    continue;
                }
                let c = &centroids[ci * dim..(ci + 1) * dim];
                let d = L2::dist(c_cur, c);
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            visited[best] = true;
            order.push(best as u32);
            cur = best;
        }
        // order[new_id] = old_id. Invert to new_from_old[old] = new.
        let mut inv = vec![0u32; n_clusters];
        for (new_id, &old_id) in order.iter().enumerate() {
            inv[old_id as usize] = new_id as u32;
        }
        inv
    }
}

/// Project centroids onto a random 1-D axis, sort by projection, use rank as
/// new id. Cheap O(n log n) alternative to the chain; captures gross locality.
pub struct ProjectionSort;
impl Reorder for ProjectionSort {
    fn permutation(centroids: &[f32], dim: usize, n_clusters: usize) -> Vec<u32> {
        let mut rng = ChaCha20Rng::seed_from_u64(0xC001_D00D);
        let mut axis = vec![0f32; dim];
        for a in axis.iter_mut() {
            *a = rng.gen::<f32>() - 0.5;
        }
        let norm = axis.iter().map(|x| x * x).sum::<f32>().sqrt();
        for a in axis.iter_mut() {
            *a /= norm;
        }
        let mut proj: Vec<(f32, u32)> = (0..n_clusters as u32)
            .map(|ci| {
                let c = &centroids[ci as usize * dim..(ci as usize + 1) * dim];
                let p: f32 = c.iter().zip(axis.iter()).map(|(x, y)| x * y).sum();
                (p, ci)
            })
            .collect();
        proj.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut inv = vec![0u32; n_clusters];
        for (new_id, (_, old_id)) in proj.iter().enumerate() {
            inv[*old_id as usize] = new_id as u32;
        }
        inv
    }
}

/// Apply a cluster-id permutation to a FlatIvf. Produces a new index whose
/// cluster at new_id contains the vectors that used to be at old_id, so
/// sequential probes hit sequential memory.
pub fn apply_permutation(ivf: &FlatIvf, new_from_old: &[u32]) -> FlatIvf {
    let n_c = ivf.n_clusters;
    assert_eq!(new_from_old.len(), n_c);

    let mut old_from_new = vec![0u32; n_c];
    for (old, &new_) in new_from_old.iter().enumerate() {
        old_from_new[new_ as usize] = old as u32;
    }

    let mut centroids = vec![0f32; n_c * ivf.dim];
    for new_ in 0..n_c {
        let old = old_from_new[new_] as usize;
        centroids[new_ * ivf.dim..(new_ + 1) * ivf.dim]
            .copy_from_slice(&ivf.centroids[old * ivf.dim..(old + 1) * ivf.dim]);
    }

    let mut new_counts = vec![0usize; n_c];
    for new_ in 0..n_c {
        let old = old_from_new[new_] as usize;
        new_counts[new_] = ivf.offsets[old + 1] - ivf.offsets[old];
    }
    let mut offsets = vec![0usize; n_c + 1];
    for new_ in 0..n_c {
        offsets[new_ + 1] = offsets[new_] + new_counts[new_];
    }

    let total = ivf.n_vectors();
    let mut vectors = vec![0f32; total * ivf.dim];
    let mut ids = vec![0u32; total];
    for new_ in 0..n_c {
        let old = old_from_new[new_] as usize;
        let os = ivf.offsets[old];
        let oe = ivf.offsets[old + 1];
        let ns = offsets[new_];
        let count = oe - os;
        vectors[ns * ivf.dim..(ns + count) * ivf.dim]
            .copy_from_slice(&ivf.vectors[os * ivf.dim..oe * ivf.dim]);
        ids[ns..ns + count].copy_from_slice(&ivf.ids[os..oe]);
    }

    FlatIvf {
        dim: ivf.dim,
        n_clusters: n_c,
        centroids,
        offsets,
        vectors,
        ids,
    }
}

/// Memory estimate in bytes for a FlatIvf (dense f32 layout).
pub fn estimated_bytes(ivf: &FlatIvf) -> usize {
    ivf.centroids.len() * 4
        + ivf.offsets.len() * std::mem::size_of::<usize>()
        + ivf.vectors.len() * 4
        + ivf.ids.len() * 4
}

/// Measure the "probe stride locality" of a permutation: for a given query
/// probing the `nprobe` nearest clusters, sum |offset[c_i+1] - offset[c_i]|
/// gaps between consecutive probed offsets. Lower = tighter clumping.
pub fn probe_stride(ivf: &FlatIvf, q: &[f32], nprobe: usize) -> i64 {
    let probes = ivf.rank_centroids(q, nprobe);
    let mut total = 0i64;
    for w in probes.windows(2) {
        let a = ivf.offsets[w[0] as usize] as i64;
        let b = ivf.offsets[w[1] as usize] as i64;
        total += (b - a).abs();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn mk(seed: u64, n: usize, dim: usize) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn baseline_search_recovers_known_neighbor() {
        let dim = 8;
        let n = 500;
        let data = mk(1, n, dim);
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let ivf = FlatIvf::new(&mut rng, &data, dim, 16, 5);
        // Query = exact copy of point 100 -> point 100 must be in top-5 with
        // nprobe large enough.
        let q: Vec<f32> = data[100 * dim..(100 + 1) * dim].to_vec();
        let hits = ivf.search_topk(&q, 16, 5);
        assert!(hits.iter().any(|&(_, id)| id == 100));
        assert!(hits[0].0 < 1e-6);
    }

    #[test]
    fn permutation_preserves_contents() {
        let dim = 8;
        let n = 400;
        let data = mk(3, n, dim);
        let mut rng = ChaCha20Rng::seed_from_u64(4);
        let ivf = FlatIvf::new(&mut rng, &data, dim, 12, 5);
        let perm = GreedyChain::permutation(&ivf.centroids, ivf.dim, ivf.n_clusters);
        let reordered = apply_permutation(&ivf, &perm);
        // Same vectors just reshuffled by cluster — total count equal.
        assert_eq!(reordered.n_vectors(), ivf.n_vectors());
        // Same ids present.
        let mut a = ivf.ids.clone();
        let mut b = reordered.ids.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn reorder_matches_baseline_results() {
        let dim = 8;
        let n = 500;
        let data = mk(5, n, dim);
        let mut rng = ChaCha20Rng::seed_from_u64(6);
        let ivf = FlatIvf::new(&mut rng, &data, dim, 16, 5);
        let perm = GreedyChain::permutation(&ivf.centroids, ivf.dim, ivf.n_clusters);
        let reordered = apply_permutation(&ivf, &perm);
        // Compare top-5 sets for 10 queries.
        for qi in 0..10 {
            let q = &data[qi * dim..(qi + 1) * dim];
            let a = ivf.search_topk(q, 16, 5);
            let b = reordered.search_topk(q, 16, 5);
            let mut a_ids: Vec<u32> = a.iter().map(|&(_, i)| i).collect();
            let mut b_ids: Vec<u32> = b.iter().map(|&(_, i)| i).collect();
            a_ids.sort();
            b_ids.sort();
            assert_eq!(a_ids, b_ids, "reorder must not change search results");
        }
    }

    #[test]
    fn probe_stride_lower_after_reorder() {
        let dim = 16;
        let n = 2000;
        let data = mk(7, n, dim);
        let mut rng = ChaCha20Rng::seed_from_u64(8);
        let ivf = FlatIvf::new(&mut rng, &data, dim, 64, 8);
        let perm = GreedyChain::permutation(&ivf.centroids, ivf.dim, ivf.n_clusters);
        let reordered = apply_permutation(&ivf, &perm);
        // Average stride across 50 queries should drop.
        let (mut s_before, mut s_after) = (0i64, 0i64);
        for qi in 0..50 {
            let q = &data[qi * dim..(qi + 1) * dim];
            s_before += probe_stride(&ivf, q, 8);
            s_after += probe_stride(&reordered, q, 8);
        }
        assert!(
            s_after < s_before,
            "reorder should reduce probe stride ({} -> {})",
            s_before,
            s_after
        );
    }
}
