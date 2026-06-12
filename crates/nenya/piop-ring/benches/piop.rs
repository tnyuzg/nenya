//! Criterion benchmarks for the piop-ring sumchecks.
//!
//! Full-prover benches:
//!   - `mult_check/15`: prove `∑_h eq(r_target, h)·a(h)·b(h)` at n = 15.
//!   - `linear_check/15`: prove `∑_h A_row[h]·w(h)` at n = 15 with a sparse
//!     R₀-valued matrix at ~3 nonzeros per row (R1CS-typical density).
//!
//! Fine-grained mult-check phase breakdown (both at round 0 of n = 15, i.e.
//! tables of size 2¹⁵ with 2¹⁴ fold-pairs):
//!   - `mult_check/compute_round_poly/15`: one call to the deg-3 round-poly
//!     builder (10 muls/pair × 2¹⁴ pairs = ~164 K GR32_128 muls).
//!   - `mult_check/fold_3tables/15`: one round of `fold_in_place` applied to
//!     all three tables (eq, a, b). 3 muls/pair × 2¹⁴ pairs = ~49 K muls.
//!   - Inferring total prover cost: the fold-style geometric series
//!     `2^{n-1} + 2^{n-2} + … + 1 = 2^n − 1 ≈ 2 · 2^{n-1}` means the full
//!     prover does roughly **2× the round-0 work** across all rounds, plus
//!     `n` sampler calls (~0.5 ms each for the Teichmüller lift). See
//!     `mult_check/compute_round_poly_total/15` and
//!     `mult_check/fold_3tables_total/15` for all-rounds variants that
//!     confirm this.
//!
//! n = 15 packs to 2^20 u32 witnesses under φ₂, matching the field-side
//! bench at n = 20 BB4 elements.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use galois_ring::{GR32_1, GR32_128, GR32_4};
use piop_ring::{
    build_eq_table, mult_check, prove_linear_check_simd, prove_mult_check, Mle,
    R2TeichmullerSampler, SparseMatrix,
};

fn rand_gr128(state: &mut u64) -> GR32_128 {
    let mut next = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    };
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([next(), next(), next(), next()]);
    }
    GR32_128(arr)
}

fn rand_mle(num_vars: usize, state: &mut u64) -> Mle {
    let len = 1usize << num_vars;
    let mut evals = Vec::with_capacity(len);
    for _ in 0..len {
        evals.push(rand_gr128(state));
    }
    Mle::new(evals)
}

fn rand_target(num_vars: usize, state: &mut u64) -> Vec<GR32_128> {
    (0..num_vars).map(|_| rand_gr128(state)).collect()
}

fn rand_sparse(
    log_rows: usize,
    log_cols: usize,
    nnz_per_row: usize,
    state: &mut u64,
) -> SparseMatrix {
    let mut next = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    };
    let nrows = 1usize << log_rows;
    let ncols = 1usize << log_cols;
    let mut m = SparseMatrix::new(log_rows, log_cols);
    for r in 0..nrows {
        for _ in 0..nnz_per_row {
            let c = (next() as usize) % ncols;
            let v = GR32_1(next());
            m.push(r, c, v);
        }
    }
    m
}

fn bench_mult_check(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a = rand_mle(n, &mut state);
    let b = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    // Compute the honest claim once.
    let eq = build_eq_table(&r_target);
    let mut claim = GR32_128::ZERO;
    for i in 0..a.len() {
        claim = claim + eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    c.bench_function("mult_check/15", |bencher| {
        bencher.iter(|| {
            let mut sampler = R2TeichmullerSampler::new(0xBEEF);
            let _ = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);
        });
    });
}

/// Single-call Gruen-optimized round-message build on 2¹⁵-size tables.
/// Captures the deg-2 inner-factor hot loop in isolation — no folding, no
/// sampling, no setup. Uses the *slim* eq table (length 2^{n-1}): the
/// active-slot `eq_one` factor is handled analytically by the verifier, not
/// materialized (see [`mult_check::compute_round_poly_gruen`]).
fn bench_mult_check_compute_round_poly(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a = rand_mle(n, &mut state);
    let b = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);
    let eq_gruen = build_eq_table(&r_target[1..]);

    c.bench_function("mult_check/compute_round_poly_gruen/15", |bencher| {
        bencher.iter(|| {
            black_box(mult_check::compute_round_poly_gruen(eq_gruen.evals(), &a, &b));
        });
    });
}

/// One round of `fold_in_place` applied to all three mult-check tables.
/// Tables start at size 2¹⁵; after the call each is halved to 2¹⁴. Uses
/// `iter_batched` with `LargeInput` so the per-iteration clone cost of
/// the fresh starting tables is excluded from the measurement.
fn bench_mult_check_fold_3tables(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a_base = rand_mle(n, &mut state);
    let b_base = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);
    let eq_base = build_eq_table(&r_target);
    let r = rand_gr128(&mut state);

    c.bench_function("mult_check/fold_3tables/15", |bencher| {
        bencher.iter_batched(
            || (eq_base.clone(), a_base.clone(), b_base.clone()),
            |(mut eq, mut a, mut b)| {
                eq.fold_in_place(r);
                a.fold_in_place(r);
                b.fold_in_place(r);
                black_box((eq, a, b));
            },
            BatchSize::LargeInput,
        );
    });
}

/// Sum of Gruen-optimized round-message builds across all n rounds, with the
/// real per-round table updates interleaved: `a`/`b` fold by `r`, and the slim
/// `eq_gruen` table is updated by `marginalize_and_scale` (the Gruen analogue
/// of the eq fold). The updates are inside the timed region, so this measures
/// the full per-round prover loop minus sampling. Geometric-series shaped, so
/// it lands near `2 ×` the single-call bench plus the table-update cost.
fn bench_mult_check_compute_round_poly_total(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a_base = rand_mle(n, &mut state);
    let b_base = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);
    let eq_gruen_base: Vec<GR32_128> = build_eq_table(&r_target[1..]).evals().to_vec();
    let r = rand_gr128(&mut state);
    let scale = rand_gr128(&mut state);

    c.bench_function("mult_check/compute_round_poly_gruen_total/15", |bencher| {
        bencher.iter_batched(
            || (eq_gruen_base.clone(), a_base.clone(), b_base.clone()),
            |(mut eq_gruen, mut a, mut b)| {
                for i in 0..n {
                    black_box(mult_check::compute_round_poly_gruen(&eq_gruen, &a, &b));
                    a.fold_in_place(r);
                    b.fold_in_place(r);
                    // Gruen eq update (skipped on the last round, where the slim
                    // table is already length 1).
                    if i + 1 < n {
                        mult_check::marginalize_and_scale(&mut eq_gruen, scale);
                    }
                }
                black_box((eq_gruen, a, b));
            },
            BatchSize::LargeInput,
        );
    });
}

/// Sum of `fold_in_place` calls across all n rounds on all three tables.
/// Same geometric-series shape as the prover's actual fold sequence, but
/// with no round-poly work interleaved. Subtract this from
/// `compute_round_poly_total` to get the pure compute-only time.
fn bench_mult_check_fold_3tables_total(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a_base = rand_mle(n, &mut state);
    let b_base = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);
    let eq_base = build_eq_table(&r_target);
    let r = rand_gr128(&mut state);

    c.bench_function("mult_check/fold_3tables_total/15", |bencher| {
        bencher.iter_batched(
            || (eq_base.clone(), a_base.clone(), b_base.clone()),
            |(mut eq, mut a, mut b)| {
                for _ in 0..n {
                    eq.fold_in_place(r);
                    a.fold_in_place(r);
                    b.fold_in_place(r);
                }
                black_box((eq, a, b));
            },
            BatchSize::LargeInput,
        );
    });
}

/// Full-prover bench of the Gruen-optimized mult check at n = 15. Drop-in
/// comparison to `mult_check/15`: same inputs, same claim, same challenge
/// sequence, different prover internals (slim eq table + 6-mul-per-pair
/// inner loop instead of full eq + 10-mul-per-pair). See
/// `src/mult_check.rs` for the derivation.
fn bench_mult_check_optimized(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xC0FFEE_u64;
    let a = rand_mle(n, &mut state);
    let b = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut claim = GR32_128::ZERO;
    for i in 0..a.len() {
        claim = claim + eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    c.bench_function("mult_check_optimized/15", |bencher| {
        bencher.iter(|| {
            let mut sampler = R2TeichmullerSampler::new(0xBEEF);
            let _ = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);
        });
    });
}

fn bench_linear_check(c: &mut Criterion) {
    let n = 15;
    let mut state: u64 = 0xFEEDFACE_u64;
    let matrix = rand_sparse(n, n, 3, &mut state);
    let w = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    // Honest claim via the same matvec the prover does.
    let eq = build_eq_table(&r_target);
    let mut a_row = vec![GR32_128::ZERO; 1usize << n];
    for &(row, col, val) in &matrix.entries {
        a_row[col] = a_row[col] + eq.evals()[row].mul_by_gr1(val);
    }
    let mut claim = GR32_128::ZERO;
    for i in 0..(1usize << n) {
        claim = claim + a_row[i] * w.evals()[i];
    }

    c.bench_function("linear_check/15", |bencher| {
        bencher.iter(|| {
            let mut sampler = R2TeichmullerSampler::new(0xCAFE);
            let _ = prove_linear_check_simd(&matrix, &w, &r_target, claim, &mut sampler);
        });
    });
}

criterion_group!(
    benches,
    bench_mult_check,
    bench_mult_check_compute_round_poly,
    bench_mult_check_fold_3tables,
    bench_mult_check_compute_round_poly_total,
    bench_mult_check_fold_3tables_total,
    bench_mult_check_optimized,
    bench_linear_check,
);
criterion_main!(benches);
