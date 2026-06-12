//! Field-side soundness sanity tests.

use field::{AbstractExtensionField, AbstractField, BB};
use piop_field::{
    build_eq_table, prove_linear_check, prove_mult_check, verify_linear_check, verify_mult_check,
    BB4UniformSampler, Mle, SparseMatrix, VerifyError, BB4,
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

#[test]
fn tampered_mult_check_first_round_rejected() {
    let n = 6;
    let mut state = 0xC0FFEE_u64;
    let a = rand_mle(n, &mut state);
    let b = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut claim = BB4::zero();
    for i in 0..a.len() {
        claim += eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    let mut sampler = BB4UniformSampler::new(7);
    let (mut proof, _) = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);

    proof.round_polys[0][0] += BB4::one();

    let mut verifier_sampler = BB4UniformSampler::new(7);
    let err = verify_mult_check(&proof, &r_target, &mut verifier_sampler)
        .expect_err("tampered mult-check proof must be rejected");
    assert_eq!(err, VerifyError::RoundCheckFailed { round: 0 });
}

#[test]
fn tampered_linear_check_first_round_rejected() {
    let n = 6;
    let mut state = 0xBADBEEF_u64;
    let matrix = rand_sparse(n, n, 3, &mut state);
    let w = rand_mle(n, &mut state);
    let r_target = rand_target(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut a_row = vec![BB4::zero(); matrix.num_cols()];
    for &(row, col, val) in &matrix.entries {
        a_row[col] += eq.evals()[row] * val;
    }
    let mut claim = BB4::zero();
    for i in 0..w.len() {
        claim += a_row[i] * w.evals()[i];
    }

    let mut sampler = BB4UniformSampler::new(13);
    let (mut proof, _) = prove_linear_check(&matrix, &w, &r_target, claim, &mut sampler);

    proof.round_polys[0][2] += BB4::one();

    let mut verifier_sampler = BB4UniformSampler::new(13);
    let err = verify_linear_check(&proof, &mut verifier_sampler)
        .expect_err("tampered linear-check proof must be rejected");
    assert_eq!(err, VerifyError::RoundCheckFailed { round: 0 });
}
