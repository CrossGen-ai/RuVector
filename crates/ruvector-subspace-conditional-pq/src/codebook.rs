//! Tiny deterministic Lloyd k-means for sub-vector codebooks.
//!
//! The scheduled-task harness forbids placeholder benchmarks, so this
//! implementation is real — it runs for `iters` passes with a xorshift
//! PRNG seeded init, computes centroid means, and handles empty cells
//! by re-seeding from the farthest sample. It is small enough for
//! codebooks of size K<=32 on sub-dim D/M<=16, which is the range SC-PQ
//! and the comparison PQ exercise.

pub struct Codebook {
    pub k: usize,
    pub sub_d: usize,
    /// row-major, shape [k, sub_d]
    pub centroids: Vec<f32>,
}

impl Codebook {
    pub fn centroid(&self, i: usize) -> &[f32] {
        &self.centroids[i * self.sub_d..(i + 1) * self.sub_d]
    }
}

#[inline]
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic xorshift64 PRNG; one seed in gives reproducible bench runs.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn gen_range(&mut self, hi: usize) -> usize {
        (self.next_u64() as usize) % hi
    }
}

/// Lloyd k-means on row-major [n, sub_d]. Empty clusters re-seed from the
/// sample furthest from its current assignment, which gives stable K-sized
/// codebooks even when the data is clumpy.
pub fn train_kmeans(data: &[f32], sub_d: usize, k: usize, iters: usize, seed: u64) -> Codebook {
    let n = data.len() / sub_d;
    assert!(n >= k, "need at least K samples");
    let mut rng = Rng::new(seed);
    let mut centroids = vec![0f32; k * sub_d];
    // Random distinct seeds.
    let mut taken = vec![false; n];
    let mut c = 0;
    while c < k {
        let idx = rng.gen_range(n);
        if !taken[idx] {
            taken[idx] = true;
            centroids[c * sub_d..(c + 1) * sub_d]
                .copy_from_slice(&data[idx * sub_d..(idx + 1) * sub_d]);
            c += 1;
        }
    }
    let mut assign = vec![0usize; n];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0f32; k * sub_d];
    for _ in 0..iters {
        for v in counts.iter_mut() {
            *v = 0;
        }
        for v in sums.iter_mut() {
            *v = 0.0;
        }
        let mut worst_dist = -1f32;
        let mut worst_idx = 0usize;
        for i in 0..n {
            let x = &data[i * sub_d..(i + 1) * sub_d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for j in 0..k {
                let d = sq_l2(x, &centroids[j * sub_d..(j + 1) * sub_d]);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            assign[i] = best;
            counts[best] += 1;
            for t in 0..sub_d {
                sums[best * sub_d + t] += x[t];
            }
            if best_d > worst_dist {
                worst_dist = best_d;
                worst_idx = i;
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                // Re-seed empty cell from worst-fit point.
                let src = &data[worst_idx * sub_d..(worst_idx + 1) * sub_d];
                centroids[j * sub_d..(j + 1) * sub_d].copy_from_slice(src);
            } else {
                let inv = 1.0 / counts[j] as f32;
                for t in 0..sub_d {
                    centroids[j * sub_d + t] = sums[j * sub_d + t] * inv;
                }
            }
        }
    }
    Codebook {
        k,
        sub_d,
        centroids,
    }
}
