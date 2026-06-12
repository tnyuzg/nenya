//! End-to-end `mimd_to_packed` reduction overhead.
//!
//! Sweeps the constraint count `n` over `2^e` for several `e`, with a
//! fixed witness length `m = n` (square shape, sparse with one nonzero
//! per row in each of `A_i, B_i, C_i`). Compared to the SIMD reduction
//! benched in `reduce.rs` / `parallel.rs`, MIMD pays for: (a) packing
//! 3 sets of 32 matrices via `pack_r2`, (b) three additional linear-
//! relation kernel-quotient sweeps over the matrix supports, and
//! (c) the same per-row `mul_with_quotients_only` for the mult relation.
//! The witness packing is identical to SIMD.
//!
//! The instance is constructed outside the timed region so we measure
//! exactly the reduction call.
//!
//! Run: `cargo bench -p reduction --bench mimd`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::GR32_1;
use reduction::instance::MimdR1csInstance;
use reduction::matrix::SparseMatrix;
use reduction::reduce::mimd_to_packed;

fn pseudo_random_u32(seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed as u32
}

/// Build a satisfied 32-instance MIMD with `n` constraints and witness
/// length `m = n`. Per row `i` of instance `k`, place one nonzero in
/// `A_k`, `B_k`, `C_k` so the support is exactly `n` triples per matrix.
/// Per-row scalars vary in `(k, i)` so matrices truly differ across the
/// 32 instances.
fn build_mimd(n: usize) -> MimdR1csInstance {
    let m = n;
    let scalar_a = |k: u32, i: u32| -> GR32_1 {
        GR32_1((k.wrapping_mul(0x9E37_79B9) ^ i.wrapping_mul(0xDEAD_BEEF)) | 1)
    };
    let scalar_b = |k: u32, i: u32| -> GR32_1 {
        GR32_1((k.wrapping_mul(0xBB67_AE85) ^ i.wrapping_mul(0xCAFE_BABE)) | 1)
    };

    let mut a_per_instance: [SparseMatrix<GR32_1>; 32] =
        std::array::from_fn(|_| SparseMatrix::<GR32_1>::new(n, m));
    let mut b_per_instance: [SparseMatrix<GR32_1>; 32] =
        std::array::from_fn(|_| SparseMatrix::<GR32_1>::new(n, m));
    let mut c_per_instance: [SparseMatrix<GR32_1>; 32] =
        std::array::from_fn(|_| SparseMatrix::<GR32_1>::new(n, m));

    // Fixed witness: column 0 holds 1; per row i we use three witness slots
    // (x_i, y_i, z_i) cycled mod m to keep the witness length at m and the
    // sparsity at one nonzero per matrix per row.
    let x_col = |i: usize| 1 + (3 * i) % (m - 1);
    let y_col = |i: usize| 1 + (3 * i + 1) % (m - 1);
    let z_col = |i: usize| 1 + (3 * i + 2) % (m - 1);

    for k in 0..32 {
        for i in 0..n {
            a_per_instance[k].push(i, x_col(i), scalar_a(k as u32, i as u32));
            b_per_instance[k].push(i, y_col(i), scalar_b(k as u32, i as u32));
            c_per_instance[k].push(i, z_col(i), GR32_1::ONE);
        }
    }

    let mut witnesses: [Vec<GR32_1>; 32] = std::array::from_fn(|_| vec![]);
    for k in 0..32 {
        let mut w = vec![GR32_1::ZERO; m];
        w[0] = GR32_1::ONE;
        let mut seed = 0xA5A5_A5A5_5A5A_5A5Au64 ^ (k as u64);
        // Fill with random `x_i`, `y_i`, then set `z_i = a*x*b*y` to satisfy
        // the constraint `(a·x) · (b·y) == z`.
        let mut x_vals = vec![GR32_1::ZERO; n];
        let mut y_vals = vec![GR32_1::ZERO; n];
        for i in 0..n {
            x_vals[i] = GR32_1(pseudo_random_u32(&mut seed));
            y_vals[i] = GR32_1(pseudo_random_u32(&mut seed));
        }
        // Apply x then y to the witness slots (later constraints may
        // overwrite earlier ones since slots cycle; satisfaction needs us
        // to lock z to the *final* x and y values at each slot).
        for i in 0..n {
            w[x_col(i)] = x_vals[i];
            w[y_col(i)] = y_vals[i];
        }
        for i in 0..n {
            let lhs = scalar_a(k as u32, i as u32) * w[x_col(i)];
            let rhs = scalar_b(k as u32, i as u32) * w[y_col(i)];
            w[z_col(i)] = lhs * rhs;
        }
        witnesses[k] = w;
    }

    MimdR1csInstance {
        a_per_instance,
        b_per_instance,
        c_per_instance,
        witnesses,
    }
}

fn bench_mimd_reduction(c: &mut Criterion) {
    let mut g = c.benchmark_group("mimd_reduction");
    g.sample_size(10);
    for e in 10..=13 {
        let n: usize = 1 << e;
        let mimd = build_mimd(n);

        g.bench_with_input(
            BenchmarkId::from_parameter(format!("2^{e}")),
            &mimd,
            |bencher, mimd| {
                bencher.iter(|| {
                    let packed = mimd_to_packed(black_box(mimd));
                    black_box(packed)
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_mimd_reduction);
criterion_main!(benches);
