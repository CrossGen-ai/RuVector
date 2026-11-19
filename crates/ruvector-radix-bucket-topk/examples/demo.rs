//! Walkthrough: push 10k random distances into each collector, show the
//! top-5 and confirm they agree.

use ruvector_radix_bucket_topk::{
    extract::run_beam, BinaryHeapTopK, Candidate, RadixBucketTopK, SortedVecTopK, TopKCollector,
};

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as u32 as f32) / (u32::MAX as f32)
}

fn main() {
    let n: u32 = 10_000;
    let k: usize = 5;
    let mut rng = 0xC0FFEEu64;
    let events: Vec<Candidate> = (0..n)
        .map(|i| {
            // Distances in [0, 10): covers ~3 binary exponent bands.
            let d = lcg(&mut rng) * 10.0;
            Candidate::new(i, d)
        })
        .collect();

    let (heap, _) = run_beam(BinaryHeapTopK::new(k), &events);
    let (radix, _) = run_beam(RadixBucketTopK::new(k), &events);
    let sorted = {
        let mut c = SortedVecTopK::new(k);
        for e in &events {
            c.push(*e);
        }
        c.finalize()
    };

    println!("top-{k} by binary-heap:");
    for c in &heap {
        println!("  id={:>5} dist={:.6}", c.id, c.dist);
    }
    println!("top-{k} by radix-bucket:");
    for c in &radix {
        println!("  id={:>5} dist={:.6}", c.id, c.dist);
    }
    println!("top-{k} by sorted-vec:");
    for c in &sorted {
        println!("  id={:>5} dist={:.6}", c.id, c.dist);
    }

    // All three collectors must agree on the exact top-K distances up to ties.
    let d_heap: Vec<f32> = heap.iter().map(|c| c.dist).collect();
    let d_radix: Vec<f32> = radix.iter().map(|c| c.dist).collect();
    let d_sorted: Vec<f32> = sorted.iter().map(|c| c.dist).collect();
    assert_eq!(d_heap, d_sorted, "heap vs sorted mismatch");
    assert_eq!(d_heap, d_radix, "heap vs radix mismatch");
    println!("all three collectors agree on top-{k}");
}
