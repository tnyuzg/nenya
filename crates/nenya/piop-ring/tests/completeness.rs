//! End-to-end completeness round-trips for both PIOP sumchecks at
//! `num_vars ∈ {6, 10}`. Honest prover, honest verifier, honest binding
//! point. Each test runs the prover, verifies the proof, and asserts that
//! the verifier's challenge sequence equals the prover's.

use galois_ring::{GR32_1, GR32_128, GR32_4};
use piop_ring::{
    build_eq_table, prove_linear_check_simd, prove_mult_check, verify_linear_check_simd,
    verify_mult_check, Mle, R2TeichmullerSampler, SparseMatrix,
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

fn mult_check_claim(a: &Mle, b: &Mle, r_target: &[GR32_128]) -> GR32_128 {
    let eq = build_eq_table(r_target);
    let mut acc = GR32_128::ZERO;
    for i in 0..a.len() {
        acc = acc + eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }
    acc
}

fn linear_check_claim(matrix: &SparseMatrix, w: &Mle, r_target: &[GR32_128]) -> GR32_128 {
    let eq = build_eq_table(r_target);
    let mut a_row = vec![GR32_128::ZERO; matrix.num_cols()];
    for &(row, col, val) in &matrix.entries {
        a_row[col] = a_row[col] + eq.evals()[row].mul_by_gr1(val);
    }
    let mut acc = GR32_128::ZERO;
    for i in 0..w.len() {
        acc = acc + a_row[i] * w.evals()[i];
    }
    acc
}

fn mult_check_round_trip(num_vars: usize, seed: u64) {
    let mut state = seed;
    let a = rand_mle(num_vars, &mut state);
    let b = rand_mle(num_vars, &mut state);
    let r_target = rand_target(num_vars, &mut state);
    let claim = mult_check_claim(&a, &b, &r_target);

    let mut prover_sampler = R2TeichmullerSampler::new(0xABCDEF);
    let (proof, prover_challenges) =
        prove_mult_check(&a, &b, &r_target, claim, &mut prover_sampler);

    let mut verifier_sampler = R2TeichmullerSampler::new(0xABCDEF);
    let verifier_challenges = verify_mult_check(&proof, &r_target, &mut verifier_sampler)
        .expect("honest mult-check proof must verify");

    assert_eq!(prover_challenges, verifier_challenges);
}

fn mult_check_optimized_round_trip(num_vars: usize, seed: u64) {
    let mut state = seed;
    let a = rand_mle(num_vars, &mut state);
    let b = rand_mle(num_vars, &mut state);
    let r_target = rand_target(num_vars, &mut state);
    let claim = mult_check_claim(&a, &b, &r_target);

    let mut prover_sampler = R2TeichmullerSampler::new(0xC0FFEE);
    let (proof, prover_challenges) =
        prove_mult_check(&a, &b, &r_target, claim, &mut prover_sampler);

    let mut verifier_sampler = R2TeichmullerSampler::new(0xC0FFEE);
    let verifier_challenges = verify_mult_check(&proof, &r_target, &mut verifier_sampler)
        .expect("honest mult-check (optimized) proof must verify");

    assert_eq!(prover_challenges, verifier_challenges);
}

fn linear_check_round_trip(num_vars: usize, seed: u64) {
    let mut state = seed;
    let matrix = rand_sparse(num_vars, num_vars, 3, &mut state);
    let w = rand_mle(num_vars, &mut state);
    let r_target = rand_target(num_vars, &mut state);
    let claim = linear_check_claim(&matrix, &w, &r_target);

    let mut prover_sampler = R2TeichmullerSampler::new(0xFEEDBEEF);
    let (proof, prover_challenges) =
        prove_linear_check_simd(&matrix, &w, &r_target, claim, &mut prover_sampler);

    let mut verifier_sampler = R2TeichmullerSampler::new(0xFEEDBEEF);
    let verifier_challenges = verify_linear_check_simd(&proof, &mut verifier_sampler)
        .expect("honest linear-check proof must verify");

    assert_eq!(prover_challenges, verifier_challenges);
}

#[test]
fn mult_check_round_trip_6_vars() {
    mult_check_round_trip(6, 1);
}

#[test]
fn mult_check_round_trip_10_vars() {
    mult_check_round_trip(10, 2);
}

#[test]
fn mult_check_optimized_round_trip_6_vars() {
    mult_check_optimized_round_trip(6, 21);
}

#[test]
fn mult_check_optimized_round_trip_10_vars() {
    mult_check_optimized_round_trip(10, 22);
}

#[test]
fn linear_check_round_trip_6_vars() {
    linear_check_round_trip(6, 3);
}

#[test]
fn linear_check_round_trip_10_vars() {
    linear_check_round_trip(10, 4);
}
