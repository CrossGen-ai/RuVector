use crate::batch::{QueryBatch, TopK};
use crate::ivf::{l2_sq, IvfIndex};

/// Serial baseline: for each query, visit each probed list and compute
/// distances. Posting-list vectors are refetched per query, which is the
/// cache cost the batched variants are attacking.
pub fn serial_scan(index: &IvfIndex, batch: &QueryBatch, probes: &[u32]) -> Vec<TopK> {
    let mut tops: Vec<TopK> = (0..batch.n).map(|_| TopK::new(batch.k)).collect();
    for q in 0..batch.n {
        let qvec = batch.query(q);
        let q_probes = &probes[q * batch.n_probe..(q + 1) * batch.n_probe];
        for &cid in q_probes {
            let list = &index.posting[cid as usize];
            for &vid in list {
                let d = l2_sq(qvec, index.vec(vid));
                tops[q].push(d, vid);
            }
        }
    }
    tops
}

/// Build a `list -> [query ids]` subscription map from the probe plan.
fn build_subscriptions(index: &IvfIndex, batch: &QueryBatch, probes: &[u32]) -> Vec<Vec<u32>> {
    let mut subs: Vec<Vec<u32>> = vec![Vec::new(); index.n_lists];
    for q in 0..batch.n {
        let q_probes = &probes[q * batch.n_probe..(q + 1) * batch.n_probe];
        for &cid in q_probes {
            subs[cid as usize].push(q as u32);
        }
    }
    subs
}

/// Batched list-major scan: visit each probed list exactly once; for every
/// vector in that list compute distances to every subscribing query. The
/// posting-list memory is streamed through the cache a single time per
/// scan, so bandwidth per query drops as the batch gains subscribers.
pub fn batched_list_major_scan(index: &IvfIndex, batch: &QueryBatch, probes: &[u32]) -> Vec<TopK> {
    let subs = build_subscriptions(index, batch, probes);
    let mut tops: Vec<TopK> = (0..batch.n).map(|_| TopK::new(batch.k)).collect();
    for cid in 0..index.n_lists {
        let list = &index.posting[cid];
        let sub_qs = &subs[cid];
        if sub_qs.is_empty() || list.is_empty() {
            continue;
        }
        for &vid in list {
            let v = index.vec(vid);
            for &q in sub_qs {
                let d = l2_sq(batch.query(q as usize), v);
                tops[q as usize].push(d, vid);
            }
        }
    }
    tops
}

/// Batched tiled scan: same correctness as list-major, but the inner loop
/// walks the posting list in tiles of `VEC_TILE` vectors so each vector
/// stays in L1 while all subscribing queries sweep over it. On typical L1
/// sizes this cuts cross-query reload traffic even further.
pub fn batched_tiled_scan(index: &IvfIndex, batch: &QueryBatch, probes: &[u32]) -> Vec<TopK> {
    const VEC_TILE: usize = 32;
    let subs = build_subscriptions(index, batch, probes);
    let mut tops: Vec<TopK> = (0..batch.n).map(|_| TopK::new(batch.k)).collect();
    let dim = index.dim;
    for cid in 0..index.n_lists {
        let list = &index.posting[cid];
        let sub_qs = &subs[cid];
        if sub_qs.is_empty() || list.is_empty() {
            continue;
        }
        let mut i = 0;
        while i < list.len() {
            let hi = (i + VEC_TILE).min(list.len());
            for &q in sub_qs {
                let qvec = batch.query(q as usize);
                for ti in i..hi {
                    let vid = list[ti];
                    let v = &index.vectors[vid as usize * dim..(vid as usize + 1) * dim];
                    let d = l2_sq(qvec, v);
                    tops[q as usize].push(d, vid);
                }
            }
            i = hi;
        }
    }
    tops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::build_ivf;
    use rand::prelude::*;
    use rand_chacha::ChaCha20Rng;

    fn gen_vecs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn all_strategies_agree() {
        let dim = 32;
        let vectors = gen_vecs(2000, dim, 1);
        let index = build_ivf(vectors, dim, 32, 3, 2);
        let queries = gen_vecs(16, dim, 7);
        let batch = QueryBatch::new(queries, dim, 10, 6);
        let probes = batch.probe(&index);
        let a = serial_scan(&index, &batch, &probes);
        let b = batched_list_major_scan(&index, &batch, &probes);
        let c = batched_tiled_scan(&index, &batch, &probes);
        for q in 0..batch.n {
            let sa = a[q].clone().into_sorted();
            let sb = b[q].clone().into_sorted();
            let sc = c[q].clone().into_sorted();
            let ids_a: Vec<u32> = sa.iter().map(|p| p.1).collect();
            let ids_b: Vec<u32> = sb.iter().map(|p| p.1).collect();
            let ids_c: Vec<u32> = sc.iter().map(|p| p.1).collect();
            assert_eq!(ids_a, ids_b, "list-major disagrees with serial on q={}", q);
            assert_eq!(ids_a, ids_c, "tiled disagrees with serial on q={}", q);
        }
    }

    #[test]
    fn top_k_is_correct_subset_of_probed_lists() {
        let dim = 16;
        let vectors = gen_vecs(500, dim, 3);
        let index = build_ivf(vectors, dim, 10, 3, 4);
        let queries = gen_vecs(4, dim, 11);
        let batch = QueryBatch::new(queries, dim, 5, 3);
        let probes = batch.probe(&index);
        let tops = batched_list_major_scan(&index, &batch, &probes);
        for q in 0..batch.n {
            let probed: std::collections::HashSet<u32> = probes
                [q * batch.n_probe..(q + 1) * batch.n_probe]
                .iter()
                .flat_map(|&c| index.posting[c as usize].iter().copied())
                .collect();
            for &(_, id) in &tops[q].items {
                assert!(probed.contains(&id), "top-k id outside probed lists");
            }
        }
    }
}
