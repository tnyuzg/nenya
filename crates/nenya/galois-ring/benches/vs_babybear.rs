//! Field-vs-ring arithmetic comparison at the two size tiers where a direct
//! pairing exists.
//!
//! Tiers:
//!   ~32 bits     BabyBear    vs  GR32_1    (Z/2^32)
//!   ~128 bits    BabyBear^4  vs  GR32_128  (GR(2^32, 128))
//!
//! GR32_4 has no field analog — it is only the inner ring used to build
//! GR32_128 — so it does not appear here.
//!
//! Format matches `arithmetic.rs`: one op per criterion iter, both operands
//! `black_box`'d. Consecutive iters have no data dependency, so this
//! measures per-op throughput (+ a small harness floor), not single-op
//! critical-path latency. See `galois_ring.md` for why throughput is the
//! right metric here.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use field::{AbstractField, BB, BB4};
use galois_ring::{GR32_1, GR32_128, GR32_4};

fn bench_base(c: &mut Criterion) {
    let bb_a = BB::from_canonical_u32(0xDEAD_BEEF);
    let bb_b = BB::from_canonical_u32(0x1234_5678);
    let gr1_a = GR32_1(0xDEAD_BEEF);
    let gr1_b = GR32_1(0x1234_5678);

    let mut g = c.benchmark_group("base_32bit");
    g.bench_function("BabyBear/add", |bencher| {
        bencher.iter(|| black_box(bb_a) + black_box(bb_b))
    });
    g.bench_function("GR32_1/add", |bencher| {
        bencher.iter(|| black_box(gr1_a) + black_box(gr1_b))
    });
    g.bench_function("BabyBear/mul", |bencher| {
        bencher.iter(|| black_box(bb_a) * black_box(bb_b))
    });
    g.bench_function("GR32_1/mul", |bencher| {
        bencher.iter(|| black_box(gr1_a) * black_box(gr1_b))
    });
    g.finish();
}

fn bench_ext(c: &mut Criterion) {
    let bb4_a = BB4::from_canonical_u32(0xDEAD_BEEF);
    let bb4_b = BB4::from_canonical_u32(0x1234_5678);

    // Build two nontrivial GR32_128 elements via algebraic combinations of β = Y.
    let mut beta_arr = [GR32_4::ZERO; 32];
    beta_arr[1] = GR32_4::ONE;
    let beta = GR32_128(beta_arr);
    let gr128_a = beta * beta + GR32_128::ONE;
    let gr128_b = beta + GR32_128::from_gr4(GR32_4([3, 1, 0, 0]));

    let mut g = c.benchmark_group("ext_128bit");
    g.bench_function("BabyBear4/add", |bencher| {
        bencher.iter(|| black_box(bb4_a) + black_box(bb4_b))
    });
    g.bench_function("GR32_128/add", |bencher| {
        bencher.iter(|| black_box(gr128_a) + black_box(gr128_b))
    });
    g.bench_function("BabyBear4/mul", |bencher| {
        bencher.iter(|| black_box(bb4_a) * black_box(bb4_b))
    });
    g.bench_function("GR32_128/mul", |bencher| {
        bencher.iter(|| black_box(gr128_a) * black_box(gr128_b))
    });
    g.finish();
}

criterion_group!(benches, bench_base, bench_ext);
criterion_main!(benches);
