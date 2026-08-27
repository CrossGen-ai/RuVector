use ruvector_avq::avq::AvqScoreAware;
use ruvector_avq::avq_norm::AvqNorm;
use ruvector_avq::data::synth_corpus;
use ruvector_avq::pq::PqMse;
use ruvector_avq::{
    brute_top_k, recall_at_k, top_k_scores, AnisotropicConfig, Quantizer, QuantizerConfig,
};

fn make_corpus(dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let c = synth_corpus(dim, 2_000, 50, 2_000, 8, 42);
    (c.train, c.base, c.queries)
}

#[test]
fn encode_shapes_match() {
    let dim = 64;
    let (train, base, _q) = make_corpus(dim);
    let cfg = QuantizerConfig { m: 8, ks: 256, iters: 5, seed: 7 };
    let mut q = PqMse::new(cfg, dim).unwrap();
    q.train(&train).unwrap();
    let codes = q.encode(&base).unwrap();
    assert_eq!(codes.len(), base.len() * cfg.m);
}

#[test]
fn deterministic_seed() {
    let dim = 32;
    let (train, base, queries) = make_corpus(dim);
    let cfg = QuantizerConfig { m: 8, ks: 256, iters: 4, seed: 99 };

    let mut a = PqMse::new(cfg, dim).unwrap();
    let mut b = PqMse::new(cfg, dim).unwrap();
    a.train(&train).unwrap();
    b.train(&train).unwrap();

    let sa = a.adc(&queries[0], &a.encode(&base).unwrap()).unwrap();
    let sb = b.adc(&queries[0], &b.encode(&base).unwrap()).unwrap();
    assert_eq!(sa.len(), sb.len());
    for (x, y) in sa.iter().zip(sb.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn avq_recall_reasonable() {
    let dim = 64;
    let (train, base, queries) = make_corpus(dim);
    let cfg = QuantizerConfig { m: 16, ks: 256, iters: 10, seed: 11 };
    let aniso = AnisotropicConfig { t: 0.2 };
    let k = 10;

    let mut q = AvqScoreAware::new(cfg, aniso, dim).unwrap();
    q.train(&train).unwrap();
    let codes = q.encode(&base).unwrap();

    let mut r = 0.0f32;
    for query in &queries {
        let truth = brute_top_k(query, &base, k);
        let scores = q.adc(query, &codes).unwrap();
        let preds = top_k_scores(&scores, k);
        r += recall_at_k(&preds, &truth);
    }
    let mean = r / queries.len() as f32;
    // Sanity: on a small mixture-of-gaussians corpus with M=16 codes
    // per 64-D vector, Recall@10 should be substantially above chance
    // (which is 10/2000 = 0.005). Very loose lower bound so this test
    // stays deterministic and not flaky across platforms.
    assert!(mean > 0.10, "AVQ recall too low: {mean}");
}

#[test]
fn avq_norm_side_channel_is_four_bytes() {
    let dim = 32;
    let (train, base, _q) = make_corpus(dim);
    let cfg = QuantizerConfig { m: 8, ks: 256, iters: 4, seed: 3 };
    let mut q = AvqNorm::new(cfg, AnisotropicConfig::default(), dim).unwrap();
    q.train(&train).unwrap();
    let _codes = q.encode_with_norms(&base).unwrap();
    assert_eq!(q.side_bytes(), 4);
    assert_eq!(q.norms.len(), base.len());
    for n in &q.norms {
        assert!(n.is_finite());
    }
}
