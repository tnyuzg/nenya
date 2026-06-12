//! Per-op arithmetic throughput for the three Galois-ring levels.
//!
//! Format: one op per criterion iter, both operands `black_box`'d.
//! Consecutive iters have no data dependency, so this measures per-op
//! throughput (+ a small harness floor), not single-op critical-path
//! latency. See `galois_ring.md` for why throughput is the right metric.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galois_ring::{GR32_1, GR32_128, GR32_4};

fn bench_gr1(c: &mut Criterion) {
    let a = GR32_1(0xDEAD_BEEF);
    let b = GR32_1(0x1234_5678);

    let mut g = c.benchmark_group("gr1");
    g.bench_function("add", |bencher| {
        bencher.iter(|| black_box(a) + black_box(b))
    });
    g.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(a) * black_box(b))
    });
    g.finish();
}

fn bench_gr4(c: &mut Criterion) {
    let a = GR32_4([0xDEAD_BEEF, 0x1234_5678, 0x9ABC_DEF0, 0x0FED_CBA9]);
    let b = GR32_4([0x0123_4567, 0x89AB_CDEF, 0xFEDC_BA98, 0x7654_3210]);

    let mut g = c.benchmark_group("gr4");
    g.bench_function("add", |bencher| {
        bencher.iter(|| black_box(a) + black_box(b))
    });
    g.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(a) * black_box(b))
    });
    g.finish();
}

fn bench_gr128(c: &mut Criterion) {
    // Build two nontrivial GR32_128 elements via algebraic combinations of β = Y.
    let mut beta_arr = [GR32_4::ZERO; 32];
    beta_arr[1] = GR32_4::ONE;
    let beta = GR32_128(beta_arr);
    let a = beta * beta + GR32_128::ONE;
    let b = beta + GR32_128::from_gr4(GR32_4([3, 1, 0, 0]));
    let small = GR32_1(0xDEAD_BEEF);
    let small_gr4 = GR32_4([0xDEAD_BEEF, 0x1234_5678, 0x9ABC_DEF0, 0x0FED_CBA9]);

    let mut g = c.benchmark_group("gr128");
    g.bench_function("add", |bencher| {
        bencher.iter(|| black_box(a) + black_box(b))
    });
    g.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(a) * black_box(b))
    });
    g.bench_function("mul_by_gr1", |bencher| {
        bencher.iter(|| black_box(a).mul_by_gr1(black_box(small)))
    });
    g.bench_function("mul_by_gr4", |bencher| {
        bencher.iter(|| black_box(a).mul_by_gr4(black_box(small_gr4)))
    });
    g.finish();
}

criterion_group!(benches, bench_gr1, bench_gr4, bench_gr128);
criterion_main!(benches);
