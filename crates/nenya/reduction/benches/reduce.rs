//! Reduction overhead as a function of constraint count `n`.
//!
//! The only *new* work `simd_to_packed` adds beyond what the prover already
//! does (witness packing → needed by the downstream IOP; per-batch matvec →
//! produced during satisfaction checking) is the `n` calls to
//! `mul_with_quotients` that compute the per-row quotient pairs
//! `(q1[i], q2[i])`. This bench times exactly that loop.
//!
//! Witness length `m` is amortized and never enters the measurement —
//! fixed at any value (in a real circuit `m = 2^20` is typical) it would be
//! invisible here.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::{GR32_1, GR32_128};
use reduction::embedding::{mul_with_quotients, rect_from_flat};

/// Tiny xorshift to fill a length-`n` GR32_1 vector with reproducible noise.
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

fn bench_reduction_overhead(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduction_overhead");
    g.sample_size(10);
    for e in 10..=13 {
        let n: usize = 1 << e;
        // Stand in for `aw[k] = A·w_k`, `bw[k] = B·w_k` — vectors the prover
        // already has from satisfaction checking. Built once, outside timing.
        let aw: [Vec<GR32_1>; 32] =
            std::array::from_fn(|k| pseudo_random_vec(0x9E37_79B9_7F4A_7C15 ^ (k as u64), n));
        let bw: [Vec<GR32_1>; 32] =
            std::array::from_fn(|k| pseudo_random_vec(0xBB67_AE85_84CA_A73B ^ (k as u64), n));

        g.bench_with_input(
            BenchmarkId::from_parameter(format!("2^{e}")),
            &(aw, bw),
            |bencher, (aw, bw)| {
                bencher.iter(|| {
                    let mut q1: Vec<GR32_128> = Vec::with_capacity(n);
                    let mut q2: Vec<GR32_128> = Vec::with_capacity(n);
                    for i in 0..n {
                        let mut x_row = [GR32_1::ZERO; 32];
                        let mut y_row = [GR32_1::ZERO; 32];
                        for k in 0..32 {
                            x_row[k] = aw[k][i];
                            y_row[k] = bw[k][i];
                        }
                        let (_c_phi, q1_i, q2_i) =
                            mul_with_quotients(rect_from_flat(&x_row), rect_from_flat(&y_row));
                        q1.push(q1_i);
                        q2.push(q2_i);
                    }
                    black_box((q1, q2))
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_reduction_overhead);
criterion_main!(benches);
