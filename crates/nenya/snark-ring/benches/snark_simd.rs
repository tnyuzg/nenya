//! Benchmarks for the SIMD SNARK at R1CS sizes 2^15 and 2^20.
//!
//! Builds a large satisfied SIMD R1CS (32 batched `x*x = y` instances with
//! a square `n × m = m` constraint matrix), runs `prove_simd` and
//! `verify_simd`, and reports prover/verifier wall-clock times plus proof
//! size.
//!
//! Note: at `2^20` constraints, each commitment encodes `~2^10 × 2^12`
//! `R_2` words of intermediate state — this is a multi-GB working set and
//! takes minutes per iteration. The bench is wired with `sample_size(10)`
//! and a 10-minute measurement budget for that case.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use galois_ring::GR32_1;
use pcs_ring::Pcs;
use reduction::matrix::SparseMatrix;
use reduction::SimdR1csInstance;
use snark_ring::{prove_simd, verify_simd, R2Pcs, SimdSnarkParams};
use std::time::Duration;

/// Build a satisfied SIMD instance whose `n × m` matrices have `n = m`
/// (square). Encodes `n` independent `x*x = y` constraints per witness;
/// with witness layout `[1, x_0, y_0, x_1, y_1, …]` of length `1 + 2·k`,
/// we set the leading 1 + zero-pad to land at length `m`.
fn build_square_simd(log_size: usize) -> SimdR1csInstance {
    let n = 1usize << log_size;
    let m = n;
    let k = (m - 1) / 2; // number of squaring constraints we can fit
    assert!(k <= n, "must have constraints <= rows");

    let mut a = SparseMatrix::<GR32_1>::new(n, m);
    let mut b = SparseMatrix::<GR32_1>::new(n, m);
    let mut c = SparseMatrix::<GR32_1>::new(n, m);
    for i in 0..k {
        let x_col = 1 + 2 * i;
        let y_col = 1 + 2 * i + 1;
        a.push(i, x_col, GR32_1::ONE);
        b.push(i, x_col, GR32_1::ONE);
        c.push(i, y_col, GR32_1::ONE);
    }

    let mut witnesses: [Vec<GR32_1>; 32] = std::array::from_fn(|_| vec![]);
    for kk in 0..32 {
        let mut w = vec![GR32_1::ZERO; m];
        w[0] = GR32_1::ONE;
        for i in 0..k {
            let x = GR32_1((((kk * 0x9E37) ^ (i * 0x85EB)) as u32).wrapping_add(1));
            w[1 + 2 * i] = x;
            w[1 + 2 * i + 1] = x * x;
        }
        witnesses[kk] = w;
    }

    SimdR1csInstance { a, b, c, witnesses }
}

fn pick_pcs_dims(log_size: usize) -> (usize, usize) {
    // Aim for k1 ≈ k2 (square reshape, balances commit / open cost).
    let half = log_size / 2;
    (1 << half, 1 << (log_size - half))
}

fn bench_at(c: &mut Criterion, log_size: usize, group_label: &str) {
    let mut group = c.benchmark_group(group_label);
    group.sample_size(10);
    if log_size >= 18 {
        group.measurement_time(Duration::from_secs(600));
    }
    group.throughput(Throughput::Elements(1u64 << log_size));

    let (k1_w, k2_w) = pick_pcs_dims(log_size);
    let (k1_q, k2_q) = pick_pcs_dims(log_size);
    let pcs_w = Pcs::new(k1_w, k2_w, 4, 16, 0xDEAD_BEEF);
    let pcs_q = R2Pcs::new(k1_q, k2_q, 4, 16, 0xCAFE_BABE);
    let params = SimdSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w,
        pcs_q,
    };

    let simd = build_square_simd(log_size);
    let proof = prove_simd(&simd, &params);
    let proof_bytes = proof.size_bytes();
    println!(
        "[snark_simd 2^{log_size}] proof_size = {} bytes",
        proof_bytes
    );

    group.bench_with_input(BenchmarkId::new("prove", log_size), &log_size, |b, _| {
        b.iter(|| {
            let p = prove_simd(&simd, &params);
            criterion::black_box(p);
        });
    });

    group.bench_with_input(BenchmarkId::new("verify", log_size), &log_size, |b, _| {
        b.iter(|| {
            let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
            criterion::black_box(r.is_ok());
        });
    });

    group.finish();
}

fn bench_2pow15(c: &mut Criterion) {
    bench_at(c, 15, "snark_simd_2pow15");
}
fn bench_2pow20(c: &mut Criterion) {
    bench_at(c, 20, "snark_simd_2pow20");
}

criterion_group!(benches, bench_2pow15, bench_2pow20);
criterion_main!(benches);
