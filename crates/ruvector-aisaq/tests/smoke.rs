//! Smoke tests: small dataset, all three backends must return
//! non-empty, in-range top-k answers, and PQ variants must
//! agree with each other on the same graph.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_aisaq::backends::{DistanceBackend, FlatF32Ram, PqDisk, PqRam};
use ruvector_aisaq::graph::{BeamSearcher, KnnGraph};
use ruvector_aisaq::pq::ProductQuantizer;

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0f32; n * d];
    for v in out.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
    out
}

#[test]
fn end_to_end_all_backends() {
    let n = 512;
    let d = 32;
    let m = 8;
    let r = 16;
    let beam = 32;
    let k = 5;

    let base = gaussian(n, d, 1);
    let query = gaussian(1, d, 2);
    let pq = ProductQuantizer::train(&base, d, m, 3);
    let codes = pq.encode_all(&base);

    let graph = KnnGraph::build_bruteforce(&base, d, r);

    // Variant 1
    let mut s1 = BeamSearcher::new(&graph, FlatF32Ram::new(base.clone(), d), beam);
    let r1 = s1.search(&query, k);
    assert_eq!(r1.len(), k);
    for id in &r1 { assert!((*id as usize) < n); }

    // Variant 2
    let pq_a = ProductQuantizer::train(&base, d, m, 3);
    let codes_a = pq_a.encode_all(&base);
    let mut s2 = BeamSearcher::new(&graph, PqRam::new(pq_a, codes_a), beam);
    let r2 = s2.search(&query, k);
    assert_eq!(r2.len(), k);

    // Variant 3: AISAQ
    let tmp = std::env::temp_dir().join(format!("aisaq-smoke-{}.bin", std::process::id()));
    let pq_b = ProductQuantizer::train(&base, d, m, 3);
    let backend = PqDisk::create(pq_b, &codes, tmp.clone()).unwrap();
    let mut s3 = BeamSearcher::new(&graph, backend, beam);
    let r3 = s3.search(&query, k);
    assert_eq!(r3.len(), k);
    let _ = std::fs::remove_file(tmp);

    // PQ-RAM and PQ-DISK should produce identical results (same PQ, same
    // graph, same beam) — this is a key correctness invariant of AISAQ:
    // storage location must not perturb search behavior.
    assert_eq!(r2, r3, "AISAQ (disk) must match in-RAM PQ ordering");

    // Flat-f32 recall vs PQ: overlap of at least one id.
    let overlap = r1.iter().filter(|id| r3.contains(id)).count();
    assert!(overlap >= 1, "PQ should share >=1 neighbour with exact search on random data");
}

#[test]
fn ram_footprint_ordering() {
    // Codebooks are ~M*K*dsub*4 bytes and dominate for very small N.
    // Use N large enough that codes >> codebooks so ordering reflects
    // the asymptotic AISAQ story rather than the fixed-cost prologue.
    let n = 16_384;
    let d = 32;
    let m = 8;

    let base = gaussian(n, d, 5);
    let pq = ProductQuantizer::train(&base, d, m, 5);
    let codes = pq.encode_all(&base);

    let flat = FlatF32Ram::new(base.clone(), d);
    let pq_a = ProductQuantizer::train(&base, d, m, 5);
    let pq_ram = PqRam::new(pq_a, codes.clone());

    let tmp = std::env::temp_dir().join(format!("aisaq-ram-{}.bin", std::process::id()));
    let pq_b = ProductQuantizer::train(&base, d, m, 5);
    let pq_disk = PqDisk::create(pq_b, &codes, tmp.clone()).unwrap();

    // The whole point of AISAQ: disk backend's heap footprint should
    // be dramatically smaller than the flat-f32 baseline.
    assert!(flat.ram_bytes() > pq_ram.ram_bytes());
    assert!(pq_ram.ram_bytes() > pq_disk.ram_bytes());
    let _ = std::fs::remove_file(tmp);
}
