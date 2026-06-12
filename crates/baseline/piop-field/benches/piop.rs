//! Criterion benchmarks for the piop-field sumchecks.
//!
//! Two benches at n = 20 (matching ring n = 15 under the φ₂ packing factor
//! of 32):
//!   - `mult_check/20`: prove `∑_h eq(r_target, h)·a(h)·b(h)`.
//!   - `linear_check/20`: prove `∑_h A_row[h]·w(h)` with sparse density 3.

use criterion::{criterion_group, criterion_main, Criterion};
use field::{AbstractExtensionField, AbstractField, BB};
use piop_field::{
    build_eq_table, prove_linear_check, prove_mult_check, BB4UniformSampler, Mle, SparseMatrix, BB4,
};

fn rand_bb4(state: &mut u64) -> BB4 {
    let mut next = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    };
    let bases: [BB; 4] = [
        BB::from_canonical_u32(next() % 0x78000001),
        BB::from_canonical_u32(next() % 0x78000001),
        BB::from_canonical_u32(next() % 0x78000001),
        BB::from_canonical_u32(next() % 0x78000001),
    ];
    <BB4 as AbstractExtensionField<BB>>::from_base_slice(&bases)
}

fn rand_mle(num_vars: usize, state: &mut u64) -> Mle {
    let len = 1usize << num_vars;
    let mut evals = Vec::with_capacity(len);
    for _ in 0..len {
        evals.push(rand_bb4(state));
    }
    Mle::new(evals)
}

fn rand_target(num_vars: usize, state: &mut u64) -> Vec<BB4> {
    (0..num_vars).map(|_| rand_bb4(state)).collect()
}

fn rand_sparse(
    log_rows: usize,
    log_cols: usize,
    nnz_per_row: usize,
    state: &mut u64,
) -> SparseMatrix {
    let nrows = 1usize << log_rows;
    let ncols = 1usize << log_cols;
    let mut m = SparseMatrix::new(log_rows, log_cols);
    for r in 0..nrows {
        for _ in 0..nnz_per_row {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let c = ((*state >> 32) as usize) % ncols;
            let v = rand_bb4(state);
            m.push(r, c, v);
        }
    }
    m
}

fn bench_mult_check(c: &mut Criterion) {
    let n = 20;
    let mut state: u64 = 0xC0FFEE_u64;
    let a = rand_mle(n, &mut state);
    let b = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut claim = BB4::zero();
    for i in 0..a.len() {
        claim += eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    c.bench_function("mult_check/20", |bencher| {
        bencher.iter(|| {
            let mut sampler = BB4UniformSampler::new(0xBEEF);
            let _ = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);
        });
    });
}

fn bench_linear_check(c: &mut Criterion) {
    let n = 20;
    let mut state: u64 = 0xFEEDFACE_u64;
    let matrix = rand_sparse(n, n, 3, &mut state);
    let w = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut a_row = vec![BB4::zero(); 1usize << n];
    for &(row, col, val) in &matrix.entries {
        a_row[col] += eq.evals()[row] * val;
    }
    let mut claim = BB4::zero();
    for i in 0..(1usize << n) {
        claim += a_row[i] * w.evals()[i];
    }

    c.bench_function("linear_check/20", |bencher| {
        bencher.iter(|| {
            let mut sampler = BB4UniformSampler::new(0xCAFE);
            let _ = prove_linear_check(&matrix, &w, &r_target, claim, &mut sampler);
        });
    });
}

criterion_group!(benches, bench_mult_check, bench_linear_check);
criterion_main!(benches);
