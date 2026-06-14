//! Recursive random-pivot partition tree.
//!
//! At each node we sample two distinct pivots p, q from the current bucket
//! and assign every point to the closer pivot. We recurse on both sides until
//! the bucket size drops to <= `leaf_size`. Leaves are returned as `Vec<u32>`
//! containing the dataset indices.
//!
//! This is the "Hierarchical Clustering" step in HCNNG — equivalent to a
//! random projection forest with metric-based splits.

use crate::distance::Distance;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub fn partition_tree<D: Distance + ?Sized>(
    vectors: &[Vec<f32>],
    indices: Vec<u32>,
    leaf_size: usize,
    dist: &D,
    rng: &mut SmallRng,
    out_leaves: &mut Vec<Vec<u32>>,
) {
    if indices.len() <= leaf_size {
        out_leaves.push(indices);
        return;
    }

    // Pick two distinct pivots.
    let n = indices.len();
    let a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    let mut tries = 0;
    while b == a && tries < 8 {
        b = rng.gen_range(0..n);
        tries += 1;
    }
    if a == b {
        // Pathological case (e.g., duplicates) — push whole bucket as a leaf.
        out_leaves.push(indices);
        return;
    }
    let pa = indices[a];
    let pb = indices[b];

    let mut left: Vec<u32> = Vec::with_capacity(n / 2 + 1);
    let mut right: Vec<u32> = Vec::with_capacity(n / 2 + 1);
    for &idx in &indices {
        let da = dist.d(&vectors[idx as usize], &vectors[pa as usize]);
        let db = dist.d(&vectors[idx as usize], &vectors[pb as usize]);
        if da <= db {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    // Degenerate split (e.g., many duplicates of one pivot): force a halfway cut
    // so recursion still terminates.
    if left.is_empty() || right.is_empty() {
        let half = indices.len() / 2;
        let (l, r) = indices.split_at(half);
        out_leaves.push(l.to_vec());
        out_leaves.push(r.to_vec());
        return;
    }

    partition_tree(vectors, left, leaf_size, dist, rng, out_leaves);
    partition_tree(vectors, right, leaf_size, dist, rng, out_leaves);
}

pub fn build_tree<D: Distance + ?Sized>(
    vectors: &[Vec<f32>],
    leaf_size: usize,
    dist: &D,
    seed: u64,
) -> Vec<Vec<u32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let indices: Vec<u32> = (0..vectors.len() as u32).collect();
    let mut leaves = Vec::new();
    partition_tree(vectors, indices, leaf_size, dist, &mut rng, &mut leaves);
    leaves
}
