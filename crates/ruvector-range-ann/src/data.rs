//! Synthetic data generator: Gaussian clusters with modest per-cluster spread.
//!
//! Cluster-structured data is realistic for embedding workloads
//! (duplicate detection, near-duplicate clustering, fingerprint ANN) and
//! lets us set a radius whose selectivity is predictable.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// A corpus of dense vectors plus the query set drawn from the same mixture.
pub struct Corpus {
    pub points: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dim: usize,
}

/// Build a corpus of `n_points` with `n_queries` in `dim` dimensions.
///
/// The data is laid out as `n_clusters` isotropic Gaussian blobs with
/// standard deviation `std`. Queries are drawn from the same mixture so
/// that a bounded radius has non-trivial recall.
pub fn make_corpus(
    seed: u64,
    n_points: usize,
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    std: f32,
) -> Corpus {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let mut c = Vec::with_capacity(dim);
        for _ in 0..dim {
            // spread cluster centers across [-1, 1]^d
            c.push(rng.gen_range(-1.0f32..1.0f32));
        }
        centers.push(c);
    }

    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        let c = &centers[rng.gen_range(0..n_clusters)];
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            v.push(c[j] + sample_normal(&mut rng) * std);
        }
        points.push(v);
    }

    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let c = &centers[rng.gen_range(0..n_clusters)];
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            v.push(c[j] + sample_normal(&mut rng) * std);
        }
        queries.push(v);
    }

    Corpus { points, queries, dim }
}

/// Sample one value from a standard normal via Box-Muller (`rng` is
/// ChaCha, so this is deterministic for a given seed).
fn sample_normal<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.gen_range(1e-7f32..1.0);
    let u2: f32 = rng.gen();
    let mag = (-2.0 * u1.ln()).sqrt();
    mag * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Pick a radius that yields a target average result-set size against
/// a uniformly-sampled probe set, using the linear-scan ground truth.
///
/// Binary search in log space on squared radius so the caller can run
/// with the same selectivity across configurations.
pub fn calibrate_radius2(
    points: &[Vec<f32>],
    queries: &[Vec<f32>],
    target_hits: usize,
    probe_queries: usize,
) -> f32 {
    use crate::sqr_l2;
    let probe_queries = probe_queries.min(queries.len()).max(1);
    let probe = &queries[..probe_queries];

    let mut lo = 1e-6f32;
    let mut hi = 1.0e2f32;
    for _ in 0..40 {
        let mid = (lo * hi).sqrt(); // geometric midpoint
        let mut total_hits = 0usize;
        for q in probe {
            for p in points {
                if sqr_l2(q, p) <= mid {
                    total_hits += 1;
                }
            }
        }
        let mean = total_hits as f32 / probe_queries as f32;
        if mean > target_hits as f32 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo * hi).sqrt()
}
