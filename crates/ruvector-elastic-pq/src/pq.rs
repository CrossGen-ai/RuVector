//! The `ElasticPq` quantizer — ties the allocator, per-subspace codebook,
//! and encode/decode together.

use crate::allocator::{
    distortion_iterative_seed, elastic_swap_step, uniform, variance_proportional, Allocator,
    BitBudget,
};
use crate::codebook::Codebook;
use crate::kmeans::sq_l2;
use crate::Quantizer;

/// Errors emitted during training or encoding.
#[derive(Debug, thiserror::Error)]
pub enum ElasticPqError {
    /// Dimensionality is not divisible by the number of subspaces.
    #[error("dim {dim} not divisible by m {m}")]
    NonDivisibleDim {
        /// Full vector dimensionality.
        dim: usize,
        /// Requested number of subspaces.
        m: usize,
    },
    /// Encoded query dimensionality did not match the trained model.
    #[error("expected dim {expected}, got {got}")]
    DimMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Provided dimensionality.
        got: usize,
    },
}

/// Statistics captured during training — reported in the ADR / research
/// doc so the numbers travel with the artifact.
#[derive(Debug, Clone)]
pub struct TrainStats {
    /// Per-subspace bit count actually used after allocation.
    pub bits: Vec<u8>,
    /// Per-subspace training distortion (sum of assigned-centroid sq L2).
    pub distortion: Vec<f64>,
    /// Total training distortion (sum of `distortion`).
    pub total_distortion: f64,
    /// Number of elastic swap iterations run (0 for non-elastic allocators).
    pub swaps: usize,
    /// Total bits per encoded vector.
    pub total_bits: usize,
}

/// Fluent builder for [`ElasticPq`].
pub struct ElasticPqBuilder {
    m: usize,
    allocator: Allocator,
    seed: u64,
}

impl ElasticPqBuilder {
    /// Start a builder with `m` subspaces.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            allocator: Allocator::Uniform { bits: 8 },
            seed: 0xEBA_C0DE,
        }
    }

    /// Pick the bit allocator to use.
    pub fn allocator(mut self, a: Allocator) -> Self {
        self.allocator = a;
        self
    }

    /// Deterministic RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Train on `n` vectors of dimension `dim`. `points` is row-major.
    pub fn train(self, points: &[f32], n: usize, dim: usize) -> Result<ElasticPq, ElasticPqError> {
        if dim % self.m != 0 {
            return Err(ElasticPqError::NonDivisibleDim { dim, m: self.m });
        }
        let sub_dim = dim / self.m;

        // Extract per-subspace point buffers once.
        let sub_points: Vec<Vec<f32>> = (0..self.m)
            .map(|m| {
                let mut buf = Vec::with_capacity(n * sub_dim);
                for i in 0..n {
                    let start = i * dim + m * sub_dim;
                    buf.extend_from_slice(&points[start..start + sub_dim]);
                }
                buf
            })
            .collect();

        // Per-subspace variance for the variance-proportional allocator.
        let sub_var: Vec<f64> = (0..self.m)
            .map(|m| {
                let sp = &sub_points[m];
                let mut mean = vec![0f64; sub_dim];
                for i in 0..n {
                    for d in 0..sub_dim {
                        mean[d] += sp[i * sub_dim + d] as f64;
                    }
                }
                for d in 0..sub_dim {
                    mean[d] /= n as f64;
                }
                let mut var = 0f64;
                for i in 0..n {
                    for d in 0..sub_dim {
                        let x = sp[i * sub_dim + d] as f64 - mean[d];
                        var += x * x;
                    }
                }
                var / n as f64
            })
            .collect();

        // Pick initial budget.
        let (mut budget, mut swaps) = match &self.allocator {
            Allocator::Uniform { bits } => (uniform(self.m, *bits), 0usize),
            Allocator::VarianceProportional {
                total_bits,
                min_bits,
                max_bits,
            } => (
                variance_proportional(&sub_var, *total_bits, *min_bits, *max_bits),
                0,
            ),
            Allocator::DistortionIterative { start_bits, .. } => {
                (distortion_iterative_seed(self.m, *start_bits), 0)
            }
        };

        // Train an initial codebook per subspace.
        let mut codebooks = train_all(&sub_points, sub_dim, n, &budget.bits, self.seed);

        // Elastic swap loop (only for DistortionIterative).
        if let Allocator::DistortionIterative {
            min_bits,
            max_bits,
            max_swaps,
            ..
        } = &self.allocator
        {
            let mut distortion: Vec<f64> = codebooks.iter().map(|c| c.train_distortion).collect();
            for _ in 0..*max_swaps {
                let step = elastic_swap_step(&distortion, &budget, *min_bits, *max_bits);
                let (donor, recv) = match step {
                    Some(pair) => pair,
                    None => break,
                };
                let mut trial_budget = budget.clone();
                trial_budget.bits[donor] -= 1;
                trial_budget.bits[recv] += 1;
                // Retrain only the two touched subspaces.
                let new_donor = Codebook::train(
                    &sub_points[donor],
                    n,
                    sub_dim,
                    trial_budget.bits[donor],
                    self.seed.wrapping_mul(31).wrapping_add(donor as u64),
                );
                let new_recv = Codebook::train(
                    &sub_points[recv],
                    n,
                    sub_dim,
                    trial_budget.bits[recv],
                    self.seed.wrapping_mul(31).wrapping_add(recv as u64),
                );
                let old_pair = distortion[donor] + distortion[recv];
                let new_pair = new_donor.train_distortion + new_recv.train_distortion;
                if new_pair + 1e-9 < old_pair {
                    // Accept.
                    distortion[donor] = new_donor.train_distortion;
                    distortion[recv] = new_recv.train_distortion;
                    codebooks[donor] = new_donor;
                    codebooks[recv] = new_recv;
                    budget = trial_budget;
                    swaps += 1;
                } else {
                    // Reject and stop — the greedy loop hit a fixed point.
                    break;
                }
            }
        }

        // Encode offsets used by the packed byte code (byte-aligned per
        // subspace: one code fits in one u8 because bits <= 8).
        let dim_check = self.m * sub_dim;
        assert_eq!(dim_check, dim);

        let per_sub_dist: Vec<f64> = codebooks.iter().map(|c| c.train_distortion).collect();
        let stats = TrainStats {
            bits: budget.bits.clone(),
            distortion: per_sub_dist.clone(),
            total_distortion: per_sub_dist.iter().sum(),
            swaps,
            total_bits: budget.total_bits(),
        };

        Ok(ElasticPq {
            m: self.m,
            dim,
            sub_dim,
            budget,
            codebooks,
            stats,
        })
    }
}

fn train_all(
    sub_points: &[Vec<f32>],
    sub_dim: usize,
    n: usize,
    bits: &[u8],
    seed: u64,
) -> Vec<Codebook> {
    (0..sub_points.len())
        .map(|m| {
            Codebook::train(
                &sub_points[m],
                n,
                sub_dim,
                bits[m],
                seed.wrapping_mul(1_000_003).wrapping_add(m as u64),
            )
        })
        .collect()
}

/// The trained EBA-PQ model.
pub struct ElasticPq {
    m: usize,
    dim: usize,
    sub_dim: usize,
    budget: BitBudget,
    codebooks: Vec<Codebook>,
    stats: TrainStats,
}

impl ElasticPq {
    /// Number of subspaces.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Full vector dimensionality this model was trained on.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reference to the training stats captured at train time.
    pub fn stats(&self) -> &TrainStats {
        &self.stats
    }

    /// Number of bits used to encode subspace `m`.
    pub fn subspace_bits(&self, m: usize) -> u8 {
        self.budget.bits[m]
    }

    /// Precomputed ADC lookup tables for a query; used by the searcher.
    pub fn adc_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, ElasticPqError> {
        if query.len() != self.dim {
            return Err(ElasticPqError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut tables = Vec::with_capacity(self.m);
        for m in 0..self.m {
            let start = m * self.sub_dim;
            let qsub = &query[start..start + self.sub_dim];
            tables.push(self.codebooks[m].adc_table(qsub));
        }
        Ok(tables)
    }

    /// Encode `n` row-major vectors into packed byte codes.
    pub fn encode_batch(&self, vecs: &[f32], n: usize) -> Vec<u8> {
        let stride = self.m;
        let mut out = vec![0u8; n * stride];
        for i in 0..n {
            let dst = &mut out[i * stride..(i + 1) * stride];
            let src = &vecs[i * self.dim..(i + 1) * self.dim];
            for m in 0..self.m {
                let sub = &src[m * self.sub_dim..(m + 1) * self.sub_dim];
                dst[m] = self.codebooks[m].encode(sub) as u8;
            }
        }
        out
    }

    /// Bytes used by one encoded vector (byte-per-subspace layout).
    pub fn code_bytes(&self) -> usize {
        self.m
    }
}

impl Quantizer for ElasticPq {
    fn encode(&self, vec: &[f32]) -> Vec<u8> {
        self.encode_batch(vec, 1)
    }

    fn asym_l2_sq(&self, query: &[f32], code: &[u8]) -> f32 {
        let tables = self
            .adc_tables(query)
            .expect("query dim matches trained dim");
        let mut acc = 0f32;
        for m in 0..self.m {
            let idx = code[m] as usize;
            acc += tables[m][idx];
        }
        acc
    }

    fn code_bits(&self) -> usize {
        self.budget.total_bits()
    }

    fn name(&self) -> &'static str {
        // Concrete `name` is filled in by the wrapper the benchmark uses;
        // we return the family here.
        "EBA-PQ"
    }
}

/// Convenience: brute-force per-vector distance used by tests for a
/// known-correct baseline.
pub fn brute_force_sq_l2(query: &[f32], base: &[f32], n: usize, dim: usize) -> Vec<f32> {
    (0..n)
        .map(|i| sq_l2(query, &base[i * dim..(i + 1) * dim]))
        .collect()
}
