//! Small demo: train an elastic PQ over an anisotropic toy corpus and
//! print the chosen bit budget.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_elastic_pq::{Allocator, ElasticPqBuilder};

fn main() {
    let n = 800;
    let dim = 16;
    let m = 4;
    let mut rng = StdRng::seed_from_u64(0xDE_C0_DE);
    let mut base = vec![0f32; n * dim];
    for i in 0..n {
        for s in 0..m {
            let sigma = 1.0 / (1.0 + s as f32);
            for d in 0..dim / m {
                base[i * dim + s * (dim / m) + d] = (rng.gen::<f32>() * 2.0 - 1.0) * sigma;
            }
        }
    }
    let pq = ElasticPqBuilder::new(m)
        .allocator(Allocator::DistortionIterative {
            start_bits: 4,
            min_bits: 2,
            max_bits: 6,
            max_swaps: 16,
        })
        .train(&base, n, dim)
        .unwrap();
    println!("bits per subspace: {:?}", pq.stats().bits);
    println!("total bits       : {}", pq.stats().total_bits);
    println!("total distortion : {:.4}", pq.stats().total_distortion);
    println!("swaps performed  : {}", pq.stats().swaps);
}
