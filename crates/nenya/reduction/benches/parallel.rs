//! Parallel reduction overhead (rayon `par_iter` across constraint rows).
//!
//! `mul_with_quotients` has no cross-row dependencies, so the per-row loop
//! parallelizes with a single `par_iter().map().unzip()`. This bench mirrors
//! `benches/reduce.rs` — same inputs, same measurement — but with the
//! parallel driver, so `cargo bench -p reduction` gives directly comparable
//! serial and parallel numbers at each `n`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::{GR32_1, GR32_128};
use rayon::prelude::*;
use reduction::embedding::{mul_with_quotients, rect_from_flat};

fn pseudo_random_vec(seed: u64, n: usize) -> Vec<GR32_1> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            GR32_1(x as u32)
        })
        .collect()
}

fn bench_parallel_reduction(c: &mut Criterion) {
    let mut g = c.benchmark_group("parallel_reduction");
    g.sample_size(10);
    for e in 10..=15 {
        let n: usize = 1 << e;
        let aw: [Vec<GR32_1>; 32] =
            std::array::from_fn(|k| pseudo_random_vec(0x9E37_79B9_7F4A_7C15 ^ (k as u64), n));
        let bw: [Vec<GR32_1>; 32] =
            std::array::from_fn(|k| pseudo_random_vec(0xBB67_AE85_84CA_A73B ^ (k as u64), n));

        g.bench_with_input(
            BenchmarkId::from_parameter(format!("2^{e}")),
            &(aw, bw),
            |bencher, (aw, bw)| {
                bencher.iter(|| {
                    let (q1, q2): (Vec<GR32_128>, Vec<GR32_128>) = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let mut x_row = [GR32_1::ZERO; 32];
                            let mut y_row = [GR32_1::ZERO; 32];
                            for k in 0..32 {
                                x_row[k] = aw[k][i];
                                y_row[k] = bw[k][i];
                            }
                            let (_c_phi, q1_i, q2_i) =
                                mul_with_quotients(rect_from_flat(&x_row), rect_from_flat(&y_row));
                            (q1_i, q2_i)
                        })
                        .unzip();
                    black_box((q1, q2))
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_parallel_reduction);
criterion_main!(benches);
