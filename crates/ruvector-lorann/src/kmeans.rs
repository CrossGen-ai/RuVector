//! Tiny deterministic k-means (Lloyd's algorithm, k-means++ init).
//!
//! Used by `IvfIndex` and `LoRannIndex` for partitioning. Kept dependency-free
//! so the crate builds with no transitive deps. Quality is sufficient for
//! research-grade ANN benchmarks; production should swap in a SIMD/parallel
//! k-means (e.g., `linfa-clustering`) via the swappable backend trait.

use crate::{l2_normalize, Lcg};

/// k-means result.
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Centroids, row-major `[K * d]`. L2-normalized (cosine k-means).
    pub centroids: Vec<f32>,
    /// `assignments[i]` = cluster id for point `i`.
    pub assignments: Vec<usize>,
    /// Number of clusters actually used (may be < `k` if init collapsed).
    pub k: usize,
    /// Dimension.
    pub d: usize,
}

impl KMeans {
    /// Train `k` clusters on row-major data `X` of shape `[n × d]`.
    ///
    /// Assumes `X` rows are already L2-normalized (so squared L2 distance and
    /// `2 - 2·dot` agree). The cosine-k-means objective minimizes
    /// `Σ_i ||x_i - c_{a(i)}||²` with centroids re-normalized each iteration.
    pub fn train(x: &[f32], n: usize, d: usize, k: usize, max_iter: usize, seed: u64) -> Self {
        assert!(d > 0 && n >= k && k > 0, "kmeans: bad shape");
        assert_eq!(x.len(), n * d, "kmeans: x length mismatch");
        let mut rng = Lcg::new(seed);

        // k-means++ init.
        let mut centroids = vec![0.0_f32; k * d];
        let first = rng.gen_range(n);
        centroids[..d].copy_from_slice(&x[first * d..(first + 1) * d]);
        let mut d2 = vec![f32::INFINITY; n];
        for c in 1..k {
            // Update min squared distance to nearest chosen centroid.
            let last = &centroids[(c - 1) * d..c * d];
            for i in 0..n {
                let row = &x[i * d..(i + 1) * d];
                let mut s = 0.0_f32;
                for j in 0..d {
                    let v = row[j] - last[j];
                    s += v * v;
                }
                if s < d2[i] {
                    d2[i] = s;
                }
            }
            // Sample proportional to d².
            let sum: f32 = d2.iter().sum();
            if sum <= 0.0 {
                // Degenerate — copy a random point.
                let p = rng.gen_range(n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
                continue;
            }
            let t = rng.next_f32() * sum;
            let mut acc = 0.0_f32;
            let mut pick = n - 1;
            for i in 0..n {
                acc += d2[i];
                if acc >= t {
                    pick = i;
                    break;
                }
            }
            centroids[c * d..(c + 1) * d].copy_from_slice(&x[pick * d..(pick + 1) * d]);
        }

        // Lloyd iterations.
        let mut assignments = vec![0_usize; n];
        let mut new_cent = vec![0.0_f32; k * d];
        let mut counts = vec![0_usize; k];

        for _it in 0..max_iter {
            new_cent.iter_mut().for_each(|v| *v = 0.0);
            counts.iter_mut().for_each(|c| *c = 0);

            // Assign.
            for i in 0..n {
                let row = &x[i * d..(i + 1) * d];
                let mut best = 0_usize;
                let mut best_dot = f32::NEG_INFINITY;
                for c in 0..k {
                    let cent = &centroids[c * d..(c + 1) * d];
                    let mut s = 0.0_f32;
                    for j in 0..d {
                        s += row[j] * cent[j];
                    }
                    if s > best_dot {
                        best_dot = s;
                        best = c;
                    }
                }
                assignments[i] = best;
                counts[best] += 1;
                for j in 0..d {
                    new_cent[best * d + j] += row[j];
                }
            }

            // Update centroids.
            let mut moved = 0.0_f32;
            for c in 0..k {
                if counts[c] == 0 {
                    // Empty cluster — re-seed from a random point.
                    let p = rng.gen_range(n);
                    let src = &x[p * d..(p + 1) * d];
                    for j in 0..d {
                        new_cent[c * d + j] = src[j];
                    }
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    for j in 0..d {
                        new_cent[c * d + j] *= inv;
                    }
                }
                // L2-normalize (cosine k-means).
                let mut sub = &mut new_cent[c * d..(c + 1) * d];
                l2_normalize(&mut sub);
                // Track centroid drift.
                for j in 0..d {
                    let dv = new_cent[c * d + j] - centroids[c * d + j];
                    moved += dv * dv;
                }
            }
            std::mem::swap(&mut centroids, &mut new_cent);
            if moved < 1e-7 {
                break;
            }
        }

        KMeans {
            centroids,
            assignments,
            k,
            d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l2_normalize;

    fn synth_clusters(d: usize, k: usize, per_cluster: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut rng = Lcg::new(seed);
        let mut centers = vec![0.0_f32; k * d];
        for c in 0..k {
            for j in 0..d {
                centers[c * d + j] = rng.next_f32() - 0.5;
            }
            let mut sub = &mut centers[c * d..(c + 1) * d];
            l2_normalize(&mut sub);
        }
        let mut x = Vec::with_capacity(k * per_cluster * d);
        for c in 0..k {
            for _ in 0..per_cluster {
                let mut row = vec![0.0_f32; d];
                for j in 0..d {
                    row[j] = centers[c * d + j] + 0.05 * (rng.next_f32() - 0.5);
                }
                l2_normalize(&mut row);
                x.extend_from_slice(&row);
            }
        }
        (x, k * per_cluster)
    }

    #[test]
    fn recovers_well_separated_clusters() {
        let d = 16;
        let k = 4;
        let per_cluster = 50;
        let (x, n) = synth_clusters(d, k, per_cluster, 7);
        let km = KMeans::train(&x, n, d, k, 50, 7);
        // Check that the true cluster of each point is a majority within its
        // assigned learned cluster (a permutation invariant of recovery).
        let mut conf = vec![vec![0_usize; k]; k];
        for i in 0..n {
            let true_c = i / per_cluster;
            conf[true_c][km.assignments[i]] += 1;
        }
        // For each true cluster, the max overlap should be near per_cluster.
        for c in 0..k {
            let max_overlap = *conf[c].iter().max().unwrap();
            assert!(
                max_overlap as f32 >= 0.9 * per_cluster as f32,
                "true cluster {c} overlap {max_overlap}/{per_cluster}"
            );
        }
    }
}
