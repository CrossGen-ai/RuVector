use std::time::Instant;
use ruvector_imi::data::{make_clustered, make_queries};
use ruvector_imi::kmeans::l2_sq;
use ruvector_imi::{AnnIndex, FlatIvf, ImiIndex};

fn exact(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / d;
    let mut pairs: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, l2_sq(q, &data[i as usize*d..(i as usize+1)*d]))).collect();
    pairs.select_nth_unstable_by(k-1, |a,b| a.1.partial_cmp(&b.1).unwrap()); pairs.truncate(k);
    pairs.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    pairs.into_iter().map(|(i,_)| i).collect()
}
fn recall(got: &[(u32,f32)], t: &[u32]) -> f32 { let h = t.iter().filter(|x| got.iter().any(|(g,_)| g==*x)).count(); h as f32 / t.len() as f32 }
fn sweep(idx: &dyn AnnIndex, queries: &[f32], truth: &[Vec<u32>], d: usize, nq: usize, kt: usize, probe: usize, sl: usize) {
    let t = Instant::now(); let mut rs = 0.0f32;
    for q in 0..nq { let got = idx.search(&queries[q*d..(q+1)*d], probe, kt); rs += recall(&got, &truth[q]); }
    let ms = t.elapsed().as_secs_f64()*1e3;
    let tag = if idx.name() == "flat-ivf" { format!("nprobe={probe:>3}") } else { format!("shortlist={sl:>5}") };
    println!("  {} {:<16} recall@{kt}={:.3}  lat={:>6.2} ms/q", idx.name(), tag, rs/nq as f32, ms/nq as f64);
}
fn main() {
    let n = 300_000usize; let d = 128usize; let nc = 2048usize; let j = 0.07f32; let nq = 200usize; let kt = 10usize;
    println!("# scaled: n={n}, d={d}, clusters={nc}, nq={nq}, k={kt}");
    let data = make_clustered(n, d, nc, j, 0xC0FFEE);
    let queries = make_queries(nq, d, nc, j*1.1, 0xBEEF);
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = (0..nq).map(|q| exact(&data, d, &queries[q*d..(q+1)*d], kt)).collect();
    println!("gt: {:.1} ms/q", t.elapsed().as_secs_f64()*1e3/nq as f64);

    let kf = (n as f64).sqrt() as usize;
    let t = Instant::now();
    let flat = FlatIvf::build(data.clone(), d, kf, 10, 0xA);
    println!("\n[A] FlatIVF K={kf} build={:.1}s mem={:.1}MiB", t.elapsed().as_secs_f64(), flat.memory_bytes() as f64/1024.0/1024.0);
    for &p in &[4usize,8,16,32,64,128] { sweep(&flat, &queries, &truth, d, nq, kt, p, 0); }

    let ki = 96usize; // K²=9216 cells, ~33 pts/cell
    let t = Instant::now();
    let mut imi = ImiIndex::build(data.clone(), d, ki, 18, 0xB);
    println!("\n[B] IMI K={ki}² build={:.1}s mem={:.1}MiB", t.elapsed().as_secs_f64(), imi.memory_bytes() as f64/1024.0/1024.0);
    for &sl in &[512usize,1024,2048,4096,8192,16384] { imi.shortlist_cap = sl; sweep(&imi, &queries, &truth, d, nq, kt, 1<<20, sl); }
}
