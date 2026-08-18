#[cfg(test)]
mod t {
    use crate::reorder::log_gap_cost;
    use crate::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn data(n: usize, dim: usize) -> Vec<f32> {
        let mut r = StdRng::seed_from_u64(1);
        (0..n * dim).map(|_| r.gen::<f32>()).collect()
    }

    #[test]
    fn build_search_smoke() {
        let dim = 16;
        let n = 500;
        let g = build_hnsw(data(n, dim), dim, 16, 32);
        assert_eq!(g.n(), n);
        let q = vec![0.5f32; dim];
        let (ids, stats) = search_knn(&g, &q, 5, 32);
        assert_eq!(ids.len(), 5);
        assert!(stats.distances_computed > 0);
    }

    #[test]
    fn reorder_preserves_recall() {
        let dim = 24;
        let n = 800;
        let g = build_hnsw(data(n, dim), dim, 16, 48);
        let q = vec![0.3f32; dim];
        let (base_ids, _) = search_knn(&g, &q, 10, 48);
        let base: std::collections::HashSet<u32> = base_ids.iter().copied().collect();

        for strat in [
            ("bfs", bfs_order(&g)),
            ("gorder", gorder(&g, 8)),
            ("rgb", rgb_order(&g, 10)),
        ] {
            let (name, perm) = strat;
            // Permutation validity: bijection over 0..n.
            let mut seen = vec![false; n];
            for &p in &perm {
                assert!(!seen[p as usize], "{name}: duplicate {p}");
                seen[p as usize] = true;
            }
            assert!(seen.iter().all(|&x| x), "{name}: not a permutation");

            let g2 = apply_permutation(&g, &perm);
            let (new_ids, _) = search_knn(&g2, &q, 10, 48);
            // Map new ids back to original.
            let mut inverse = vec![0u32; n];
            for (new_id, &old_id) in perm.iter().enumerate() {
                inverse[old_id as usize] = new_id as u32;
            }
            let recovered: std::collections::HashSet<u32> = new_ids
                .iter()
                .map(|nid| {
                    // find old id = perm[nid]
                    perm[*nid as usize]
                })
                .collect();
            let overlap = base.intersection(&recovered).count();
            // Search over identical graph topology must give the same set.
            assert_eq!(overlap, base.len(), "{name}: recall changed after reorder");
        }
    }

    #[test]
    fn log_gap_improves() {
        let dim = 24;
        let n = 1500;
        let g = build_hnsw(data(n, dim), dim, 16, 48);
        let base_cost = log_gap_cost(&g);
        let g_rgb = apply_permutation(&g, &rgb_order(&g, 12));
        let g_gor = apply_permutation(&g, &gorder(&g, 8));
        let rgb_cost = log_gap_cost(&g_rgb);
        let gor_cost = log_gap_cost(&g_gor);
        println!("log_gap base={base_cost:.3} rgb={rgb_cost:.3} gorder={gor_cost:.3}");
        // Both reorderings should reduce log-gap on random HNSW build.
        assert!(rgb_cost < base_cost, "rgb {rgb_cost} !< base {base_cost}");
        assert!(gor_cost < base_cost, "gorder {gor_cost} !< base {base_cost}");
    }
}
