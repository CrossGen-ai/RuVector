use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_streamvbyte_hnsw::{
    AdjacencyCodec, DeltaVarintCodec, PlainCodec, StreamVByteCodec,
};

fn graph(n: usize, degree: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<u32> =
                (0..degree).map(|_| rng.gen_range(0..n as u32)).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect()
}

fn bench_decode(c: &mut Criterion) {
    let n = 100_000;
    let degree = 32;
    let lists = graph(n, degree, 2026);
    let total_edges: usize = lists.iter().map(|l| l.len()).sum();

    let codecs: Vec<Box<dyn AdjacencyCodec>> = vec![
        Box::new(PlainCodec),
        Box::new(DeltaVarintCodec),
        Box::new(StreamVByteCodec),
    ];

    for codec in &codecs {
        let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(lists.len());
        let mut lens: Vec<usize> = Vec::with_capacity(lists.len());
        for list in &lists {
            let mut buf = Vec::new();
            codec.encode(list, &mut buf);
            encoded.push(buf);
            lens.push(list.len());
        }

        let mut group = c.benchmark_group("decode");
        group.throughput(Throughput::Elements(total_edges as u64));
        group.bench_function(codec.name(), |b| {
            b.iter(|| {
                let mut scratch = Vec::with_capacity(64);
                for (i, buf) in encoded.iter().enumerate() {
                    codec.decode(buf, lens[i], &mut scratch);
                    black_box(&scratch);
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
