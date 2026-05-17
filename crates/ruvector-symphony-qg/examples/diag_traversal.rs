// Diagnose: compare graph traversal vs full ADT scan recall on same data.
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{AnnIndex, FlatIndex, SymphonyQgIndex, ProductQuantizer};

fn gauss(n: usize, d: usize, kc: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..kc).map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = &centers[i % kc];
        for j in 0..d { out.push(c[j] + normal.sample(&mut rng) as f32); }
    }
    out
}

fn recall(truth: &[(u32, f32)], got: &[u32], k: usize) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().take(k).map(|x| x.0).collect();
    let hit = got.iter().take(k).filter(|x| t.contains(x)).count();
    hit as f32 / k as f32
}

fn main() {
    let n = 8000; let d = 128; let nq = 50; let k = 10;
    let data = gauss(n, d, 32, 1.2, 17);
    let qs = gauss(nq, d, 32, 1.2, 99);
    let flat = FlatIndex::new(d, data.clone());
    let truths: Vec<_> = (0..nq).map(|i| flat.search(&qs[i*d..(i+1)*d], k)).collect();

    let pq = ProductQuantizer::train(&data, d, 32, 256, 12, 7).unwrap();
    let codes = pq.encode_all(&data).unwrap();
    // Full ADT scan recall — upper bound for any ADT-based traversal.
    let mut adt_scan_recall_total = 0.0_f32;
    for qi in 0..nq {
        let q = &qs[qi*d..(qi+1)*d];
        let lut = pq.build_adt(q);
        let mut all: Vec<(u32, f32)> = (0..n).map(|i| (i as u32, pq.adc_distance(&lut, codes.code(i)))).collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let got: Vec<u32> = all.into_iter().take(k).map(|x| x.0).collect();
        adt_scan_recall_total += recall(&truths[qi], &got, k);
    }
    println!("ADT-only full-scan top-{k} recall = {:.4}", adt_scan_recall_total / nq as f32);

    let sym = SymphonyQgIndex::build(d, data, 32, 256, 12, 32, 200, 1.2, 7).with_ef_search(200);
    println!("entries: {}", sym.entries.len());
    let degs: Vec<usize> = sym.adj.iter().map(|v| v.len()).collect();
    println!("avg deg = {:.1}", degs.iter().sum::<usize>() as f32 / degs.len() as f32);

    let mut g_total = 0.0_f32;
    for qi in 0..nq {
        let q = &qs[qi*d..(qi+1)*d];
        let got = sym.search(q, k);
        let ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        g_total += recall(&truths[qi], &ids, k);
    }
    println!("Symphony graph top-{k} recall = {:.4}", g_total / nq as f32);
}
