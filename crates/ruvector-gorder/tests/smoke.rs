use ruvector_gorder::{
    apply_permutation, layout::layout_stats, search::exact_topk, search::search_greedy, BfsLayout,
    GorderLayout, InsertionLayout, Layout, MiniHnsw, MiniHnswParams,
};

fn build() -> MiniHnsw {
    MiniHnsw::build_random(
        2_000,
        MiniHnswParams { dim: 32, m: 12, ef_construction: 48, seed: 7 },
    )
}

#[test]
fn recall_preserved_across_layouts() {
    let g0 = build();
    // Same ef, same k — the top-k id set (after remapping) must be identical
    // across layouts for a large majority of queries, since we only permute IDs.
    let query = g0.vector(123).to_vec();
    let (base_ids, _) = search_greedy(&g0, &query, 64, 10);

    for layout in [
        Box::new(InsertionLayout::default()) as Box<dyn Layout>,
        Box::new(BfsLayout::default()),
        Box::new(GorderLayout { window: 8 }),
    ] {
        let perm = layout.permutation(&g0);
        let g = apply_permutation(&g0, &perm);
        let (ids, _) = search_greedy(&g, &query, 64, 10);
        // Translate result back to original id space.
        let mut inv = vec![0u32; g.len()];
        for (old, &new_id) in perm.iter().enumerate() {
            inv[new_id as usize] = old as u32;
        }
        let translated: Vec<u32> = ids.iter().map(|&i| inv[i as usize]).collect();
        // Intersection should be at least 8/10 with baseline.
        let inter = translated.iter().filter(|id| base_ids.contains(id)).count();
        assert!(inter >= 8, "layout {} lost recall: {}/10", layout.name(), inter);
    }
}

#[test]
fn gorder_improves_locality_over_insertion() {
    let g0 = build();
    let ins = InsertionLayout::default().permutation(&g0);
    let gor = GorderLayout { window: 8 }.permutation(&g0);
    let s_ins = layout_stats(&g0, &ins, "insertion");
    let s_gor = layout_stats(&g0, &gor, "gorder");
    // Real acceptance test: gorder must reduce mean edge span meaningfully.
    assert!(
        s_gor.mean_edge_span < s_ins.mean_edge_span * 0.8,
        "gorder edge span {:.1} not < 0.8 * insertion {:.1}",
        s_gor.mean_edge_span,
        s_ins.mean_edge_span
    );
    // And raise the near-edge fraction.
    assert!(
        s_gor.near_edge_frac > s_ins.near_edge_frac,
        "gorder near-edge frac {:.3} not > insertion {:.3}",
        s_gor.near_edge_frac,
        s_ins.near_edge_frac
    );
}

#[test]
fn permutation_is_bijection() {
    let g0 = build();
    for layout in [
        Box::new(InsertionLayout::default()) as Box<dyn Layout>,
        Box::new(BfsLayout::default()),
        Box::new(GorderLayout { window: 8 }),
    ] {
        let p = layout.permutation(&g0);
        let mut seen = vec![false; g0.len()];
        for &v in &p {
            assert!(!seen[v as usize], "duplicate id in permutation: {}", v);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }
}

#[test]
fn exact_topk_returns_k() {
    let g0 = build();
    let q = g0.vector(0).to_vec();
    let top = exact_topk(&g0, &q, 5);
    assert_eq!(top.len(), 5);
    assert_eq!(top[0], 0); // vector is closest to itself
}
