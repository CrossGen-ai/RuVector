//! Bipartite-projection graph construction for RoarGraph.
//!
//! ## Algorithm (Chen et al., VLDB 2024)
//!
//! 1. For each training query q_i, find its k_train-NN among base vectors
//!    (brute-force at build time — only done once).
//! 2. For each base vector v, its projected-neighbour set is the union of all
//!    base vectors that co-appear with v in *some* training query's neighbour
//!    list, excluding v itself.
//! 3. Cap the out-degree at `max_degree`, keeping the `max_degree` closest
//!    projected neighbours by L2 distance to v.
//! 4. Connectivity pass: BFS from node 0; for any unreached node u, add an edge
//!    from u to the nearest already-reached node (greedy bridge insertion).

use std::collections::{HashSet, VecDeque};

use crate::error::RoarError;
use crate::graph::{l2sq, RoarGraph};

/// Parameters controlling the RoarGraph build.
#[derive(Debug, Clone)]
pub struct BuildParams {
    /// Number of NN per training query in the bipartite construction.
    pub k_train: usize,
    /// Maximum out-degree of any base node after projection.
    pub max_degree: usize,
}

impl Default for BuildParams {
    fn default() -> Self {
        BuildParams {
            k_train: 20,
            max_degree: 32,
        }
    }
}

/// Build the projected bipartite graph in-place on `graph`.
///
/// `training_queries` is a slice of query vectors drawn from the **query
/// distribution** (i.e., potentially OOD relative to `graph.vectors`).
pub fn build_roargraph(
    graph: &mut RoarGraph,
    training_queries: &[Vec<f32>],
    params: &BuildParams,
) -> Result<(), RoarError> {
    if graph.vectors.is_empty() {
        return Err(RoarError::EmptyIndex);
    }
    if training_queries.is_empty() {
        return Err(RoarError::NoTrainingQueries);
    }

    let n = graph.vectors.len();
    let k_train = params.k_train.min(n);

    // Step 1 & 2: for each training query, compute its k_train-NN among base
    // vectors; then for each base vector record all co-occurring base vectors.
    //
    // `projected[v]` = set of base indices that co-occur with v.
    let mut projected: Vec<HashSet<u32>> = vec![HashSet::new(); n];

    for query in training_queries {
        // Brute-force k-NN of this query over base vectors
        let mut dists: Vec<(u32, f32)> = graph
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, l2sq(query, v)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k_train);

        // For each pair (u, v) in the neighbour list, add v to projected[u]
        // and u to projected[v] (projection is symmetric).
        for i in 0..dists.len() {
            for j in 0..dists.len() {
                if i != j {
                    projected[dists[i].0 as usize].insert(dists[j].0);
                }
            }
        }
    }

    // Step 3: cap out-degree at max_degree by keeping closest neighbours.
    let mut adj: Vec<Vec<u32>> = Vec::with_capacity(n);
    for (v_idx, neighbours) in projected.into_iter().enumerate() {
        let mut nb_list: Vec<(u32, f32)> = neighbours
            .into_iter()
            .map(|nb| {
                let d = l2sq(&graph.vectors[v_idx], &graph.vectors[nb as usize]);
                (nb, d)
            })
            .collect();
        nb_list.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        nb_list.truncate(params.max_degree);
        adj.push(nb_list.into_iter().map(|(id, _)| id).collect());
    }

    // Step 4: connectivity pass — BFS from node 0; bridge any isolated nodes.
    let reached = bfs_reachable(&adj, 0, n);
    connectivity_pass(&mut adj, &reached, &graph.vectors, n);

    graph.adj = adj;
    graph.built = true;
    Ok(())
}

/// BFS from `start`; returns the set of reachable node indices.
fn bfs_reachable(adj: &[Vec<u32>], start: usize, n: usize) -> HashSet<u32> {
    let mut visited: HashSet<u32> = HashSet::with_capacity(n);
    let mut queue: VecDeque<u32> = VecDeque::new();
    visited.insert(start as u32);
    queue.push_back(start as u32);
    while let Some(cur) = queue.pop_front() {
        for &nb in &adj[cur as usize] {
            if visited.insert(nb) {
                queue.push_back(nb);
            }
        }
    }
    visited
}

/// For each unreached node, add a forward edge from it to the nearest node
/// that was reachable at the time of its bridging (or is being bridged).
/// We work iteratively: after each bridge the reached set grows, so later
/// isolated nodes may find a closer bridge.
fn connectivity_pass(
    adj: &mut Vec<Vec<u32>>,
    initial_reached: &HashSet<u32>,
    vectors: &[Vec<f32>],
    n: usize,
) {
    let mut reached = initial_reached.clone();

    for u in 0..n as u32 {
        if reached.contains(&u) {
            continue;
        }
        // Find nearest reached node
        let nearest = reached
            .iter()
            .map(|&r| (r, l2sq(&vectors[u as usize], &vectors[r as usize])))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(r, _)| r);

        if let Some(bridge) = nearest {
            // Forward edge u -> bridge (makes u reachable from bridge during search)
            adj[u as usize].push(bridge);
            // Reverse edge bridge -> u (makes u reachable via BFS from 0)
            adj[bridge as usize].push(u);
            reached.insert(u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_graph_is_connected_after_build() {
        let base = random_vecs(100, 16, 1);
        let queries = random_vecs(30, 16, 2);
        let mut g = RoarGraph::new(16);
        g.add(&base).unwrap();
        let params = BuildParams { k_train: 10, max_degree: 16 };
        build_roargraph(&mut g, &queries, &params).unwrap();

        // BFS from 0 should reach all 100 nodes
        let reached = bfs_reachable(&g.adj, 0, g.len());
        assert_eq!(reached.len(), g.len(), "graph is not fully connected");
    }

    #[test]
    fn test_build_requires_base_vectors() {
        let mut g = RoarGraph::new(4);
        let queries = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let params = BuildParams::default();
        assert!(matches!(
            build_roargraph(&mut g, &queries, &params),
            Err(RoarError::EmptyIndex)
        ));
    }

    #[test]
    fn test_build_requires_training_queries() {
        let base = random_vecs(20, 4, 5);
        let mut g = RoarGraph::new(4);
        g.add(&base).unwrap();
        let params = BuildParams::default();
        assert!(matches!(
            build_roargraph(&mut g, &[], &params),
            Err(RoarError::NoTrainingQueries)
        ));
    }

    /// Recall check: RoarGraph must match or beat the base-to-base k-NN baseline
    /// when queries are OOD (shifted Gaussian means relative to base clusters).
    #[test]
    fn test_roargraph_recall_gte_baseline_on_ood() {
        use crate::baseline::BaselineGraph;
        use crate::dataset::{exact_knn, generate_ood_dataset, DatasetParams};
        use crate::graph::l2sq as _l2sq;
        use crate::AnnIndex;

        let params = DatasetParams {
            n_base: 500,
            dim: 16,
            n_clusters: 4,
            cluster_std: 0.4,
            cluster_range: 4.0,
            ood_shift: 3.0,
            seed: 77,
        };
        let ds = generate_ood_dataset(&params, 100, 50, 5);
        let ef = 30;
        let k = 5;

        // Build baseline (base-to-base)
        let mut baseline = BaselineGraph::new(16, 16);
        baseline.add(&ds.base).unwrap();
        baseline.build(&[]).unwrap();

        // Build RoarGraph with training queries
        let bp = BuildParams { k_train: 10, max_degree: 16 };
        let mut g = RoarGraph::new(16);
        g.add(&ds.base).unwrap();
        build_roargraph(&mut g, &ds.train_queries, &bp).unwrap();

        // Measure recalls
        let mut baseline_recall = 0.0f64;
        let mut roar_recall = 0.0f64;

        for (qi, q) in ds.test_queries.iter().enumerate() {
            let gt = &ds.ground_truth[qi];

            let bl_res = baseline.search(q, k, ef).unwrap();
            let bl_hits = bl_res.iter().filter(|r| gt.contains(&r.id)).count();
            baseline_recall += bl_hits as f64 / gt.len() as f64;

            let rg_res = g.search(q, k, ef).unwrap();
            let rg_hits = rg_res.iter().filter(|r| gt.contains(&r.id)).count();
            roar_recall += rg_hits as f64 / gt.len() as f64;
        }

        baseline_recall /= ds.test_queries.len() as f64;
        roar_recall /= ds.test_queries.len() as f64;

        assert!(
            roar_recall >= baseline_recall,
            "RoarGraph recall@{k} ({:.1}%) < baseline ({:.1}%) — OOD advantage lost",
            roar_recall * 100.0,
            baseline_recall * 100.0
        );
    }
}
