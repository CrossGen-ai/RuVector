use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_gcsq::{
    backends::{evaluate, BinarySqReranker, GcsqReranker, Reranker, SimHashReranker},
    gray_decode, gray_encode, popcount_xor,
};

fn gaussian_corpus(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let u1: f32 = rng.gen_range(1e-9f32..1.0);
                    let u2: f32 = rng.gen_range(0.0f32..1.0);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect()
        })
        .collect()
}

#[test]
fn gray_code_roundtrip() {
    for n in 0u32..256 {
        assert_eq!(gray_decode(gray_encode(n)), n);
    }
}

#[test]
fn gray_adjacent_hamming_one() {
    // Core theoretical property: successive integers differ by one Hamming bit under Gray.
    for n in 0u32..255 {
        let a = gray_encode(n);
        let b = gray_encode(n + 1);
        let hd = (a ^ b).count_ones();
        assert_eq!(hd, 1, "adjacent bins differ by {hd} bits at n={n}");
    }
}

#[test]
fn popcount_known_values() {
    let a = vec![0u8, 0xFF, 0x0F];
    let b = vec![0u8, 0x0F, 0xFF];
    assert_eq!(popcount_xor(&a, &b), 0 + 4 + 4);
    let c = vec![0xAAu8; 16];
    let d = vec![0x55u8; 16];
    assert_eq!(popcount_xor(&c, &d), 16 * 8);
}

#[test]
fn encode_decode_shapes() {
    let base = gaussian_corpus(1, 64, 32);
    let gc = GcsqReranker::fit(&base, 2);
    let code = gc.encode(&base[0]);
    assert_eq!(code.len(), gc.code_bytes());
    assert_eq!(code.len(), (32 * 2 + 7) / 8);
}

#[test]
fn gcsq_beats_binary_on_hamming_correlation() {
    // Numeric acceptance: on gaussian data, GCSQ's Spearman correlation with
    // true L2 is strictly higher than plain-binary SQ at the same bitrate.
    let base = gaussian_corpus(42, 500, 64);
    let queries = gaussian_corpus(43, 50, 64);

    let gc = GcsqReranker::fit(&base, 2);
    let bn = BinarySqReranker::fit(&base, 2);

    let gc_eval = evaluate(&gc, &base, &queries, 10);
    let bn_eval = evaluate(&bn, &base, &queries, 10);

    println!("GCSQ:   rho={:.4}, recall@10={:.4}", gc_eval.spearman_rho, gc_eval.recall_at_k);
    println!("Binary: rho={:.4}, recall@10={:.4}", bn_eval.spearman_rho, bn_eval.recall_at_k);
    assert!(
        gc_eval.spearman_rho > bn_eval.spearman_rho,
        "GCSQ rho ({:.4}) should exceed binary rho ({:.4}) at same bitrate",
        gc_eval.spearman_rho,
        bn_eval.spearman_rho
    );
    assert!(gc_eval.spearman_rho > 0.3, "GCSQ rho too weak: {}", gc_eval.spearman_rho);
}

#[test]
fn simhash_matches_bitwidth() {
    let base = gaussian_corpus(7, 100, 16);
    let queries = gaussian_corpus(8, 20, 16);
    // Match GCSQ 2-bit code width: 16 dims * 2 = 32 bits.
    let sh = SimHashReranker::fit(16, 32, 99);
    assert_eq!(sh.code_bytes(), 4);
    let res = evaluate(&sh, &base, &queries, 5);
    assert!(res.spearman_rho.is_finite());
}
