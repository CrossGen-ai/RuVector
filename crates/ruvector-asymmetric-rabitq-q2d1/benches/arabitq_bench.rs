use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_asymmetric_rabitq_q2d1::{
    encode_bit, encode_ternary, est_asym, est_sym, gen_corpus, HadamardRotation,
};

fn bench_estimators(c: &mut Criterion) {
    let dim = 128;
    let rot = HadamardRotation::new(dim, 11);
    let corpus = gen_corpus(1, dim, 42);
    let mut v = corpus[0].clone();
    rot.apply(&mut v);
    let dcode = encode_bit(&v);

    let mut q = gen_corpus(1, dim, 99)[0].clone();
    rot.apply(&mut q);
    let qbit = encode_bit(&q);
    let qter = encode_ternary(&q, 0.5);

    c.bench_function("est_sym_1b1b_d128", |b| {
        b.iter(|| black_box(est_sym(&qbit, &dcode, dim)))
    });
    c.bench_function("est_asym_q2d1_d128", |b| {
        b.iter(|| black_box(est_asym(&qter, &dcode)))
    });
}

criterion_group!(benches, bench_estimators);
criterion_main!(benches);
