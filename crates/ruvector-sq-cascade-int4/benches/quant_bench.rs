use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_sq_cascade_int4::dataset::gaussian_dataset;
use ruvector_sq_cascade_int4::quant::{CascadeInt8Int4, Int8Sq, Quantizer};

fn bench_encode(c: &mut Criterion) {
    let data = gaussian_dataset(5000, 128, 1);
    let mut i8 = Int8Sq::new(128);
    i8.train(&data);
    let mut casc = CascadeInt8Int4::new(128);
    casc.train(&data);

    c.bench_function("int8_encode_d128", |b| {
        b.iter(|| black_box(i8.encode(black_box(&data[0]))))
    });
    c.bench_function("cascade_encode_d128", |b| {
        b.iter(|| black_box(casc.encode(black_box(&data[0]))))
    });

    let i8_code = i8.encode(&data[0]);
    let casc_code = casc.encode(&data[0]);
    c.bench_function("int8_decode_d128", |b| {
        b.iter(|| black_box(i8.decode(black_box(&i8_code))))
    });
    c.bench_function("cascade_decode_d128", |b| {
        b.iter(|| black_box(casc.decode(black_box(&casc_code))))
    });
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
