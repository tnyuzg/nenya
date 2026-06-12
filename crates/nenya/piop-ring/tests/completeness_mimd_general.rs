//! Completeness round-trips for `linear_check_mimd` and `linear_check_general`.
//! Honest prover, honest verifier, identically seeded samplers. Each test
//! constructs an honest claim, proves it, verifies, and asserts the
//! prover/verifier challenge sequences match.

use galois_ring::{R2TeichmullerSampler, GR32_128, GR32_4};
use piop_ring::{
    build_eq_table, linear_check_general::LAYERS, prove_linear_check_general,
    prove_linear_check_mimd, verify_linear_check_general, verify_linear_check_mimd, Mle,
};
use reduction::SparseMatrix;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn rand_gr128(state: &mut u64) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([lcg(state), lcg(state), lcg(state), lcg(state)]);
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

// ---------------------------------------------------------------------------
// MIMD
// ---------------------------------------------------------------------------

fn rand_sparse_r2(
    rows: usize,
    cols: usize,
    nnz_per_row: usize,
    state: &mut u64,
) -> SparseMatrix<GR32_128> {
    let mut m = SparseMatrix::<GR32_128>::new(rows, cols);
    for r in 0..rows {
        for _ in 0..nnz_per_row {
            let c = (lcg(state) as usize) % cols;
            m.push(r, c, rand_gr128(state));
        }
    }
    m
}

fn mimd_linear_check_claim(
    matrix: &SparseMatrix<GR32_128>,
    w: &Mle,
    r_target: &[GR32_128],
) -> GR32_128 {
    let eq = build_eq_table(r_target);
    let mut m_row = vec![GR32_128::ZERO; matrix.cols];
    for &(row, col, val) in &matrix.triples {
        m_row[col] = m_row[col] + eq.evals()[row] * val;
    }
    let mut acc = GR32_128::ZERO;
    for i in 0..w.len() {
        acc = acc + m_row[i] * w.evals()[i];
    }
    acc
}

fn mimd_round_trip(log_rows: usize, log_cols: usize, seed: u64) {
    let mut state = seed | 1;
    let matrix = rand_sparse_r2(1 << log_rows, 1 << log_cols, 3, &mut state);
    let w = rand_mle(log_cols, &mut state);
    let r_target = rand_target(log_rows, &mut state);
    let claim = mimd_linear_check_claim(&matrix, &w, &r_target);

    let mut prover_sampler = R2TeichmullerSampler::new(0xC0FFEE);
    let (proof, prover_challenges) =
        prove_linear_check_mimd(&matrix, &w, &r_target, claim, &mut prover_sampler);

    let mut verifier_sampler = R2TeichmullerSampler::new(0xC0FFEE);
    let verifier_challenges = verify_linear_check_mimd(&proof, &mut verifier_sampler)
        .expect("honest MIMD linear-check proof must verify");

    assert_eq!(prover_challenges, verifier_challenges);
}

#[test]
fn mimd_linear_check_round_trip_6_vars() {
    mimd_round_trip(6, 6, 1);
}

#[test]
fn mimd_linear_check_round_trip_10_vars() {
    mimd_round_trip(10, 10, 2);
}

#[test]
fn mimd_linear_check_round_trip_rect() {
    mimd_round_trip(8, 6, 3);
}

// ---------------------------------------------------------------------------
// General-circuit
// ---------------------------------------------------------------------------

fn general_linear_check_claim(rows: &[Mle; LAYERS], rhos: &[Mle; LAYERS]) -> GR32_128 {
    let len = rows[0].len();
    let mut acc = GR32_128::ZERO;
    for ell in 0..LAYERS {
        for i in 0..len {
            acc = acc + rows[ell].evals()[i] * rhos[ell].evals()[i];
        }
    }
    acc
}

fn build_layer_array(num_vars: usize, seed_base: u64, state: &mut u64) -> [Mle; LAYERS] {
    std::array::from_fn(|ell| rand_mle(num_vars, &mut (*state ^ seed_base ^ (ell as u64))))
}

fn general_round_trip(num_vars: usize, seed: u64) {
    let mut state = seed | 1;
    let rows = build_layer_array(num_vars, 0xA5A5_A5A5_A5A5_A5A5, &mut state);
    let rhos = build_layer_array(num_vars, 0x5A5A_5A5A_5A5A_5A5A, &mut state);
    let claim = general_linear_check_claim(&rows, &rhos);

    let mut prover_sampler = R2TeichmullerSampler::new(0xFEEDBEEF);
    let (proof, prover_challenges) =
        prove_linear_check_general(rows.clone(), rhos.clone(), claim, &mut prover_sampler);

    let mut verifier_sampler = R2TeichmullerSampler::new(0xFEEDBEEF);
    let verifier_challenges = verify_linear_check_general(&proof, &mut verifier_sampler)
        .expect("honest general linear-check proof must verify");

    assert_eq!(prover_challenges, verifier_challenges);

    // Independent verifier-side reconstruction of the final claim equality
    // (matches what the snark-ring composition layer will do later).
    let mut final_sum = GR32_128::ZERO;
    for ell in 0..LAYERS {
        final_sum = final_sum + proof.final_row[ell] * proof.final_rho[ell];
    }
    let mut c = claim;
    for (i, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        c = a0 + verifier_challenges[i] * (a1 + verifier_challenges[i] * a2);
    }
    assert_eq!(
        c, final_sum,
        "final claim must match Σ_ℓ final_row · final_rho"
    );
}

#[test]
fn general_linear_check_round_trip_4_vars() {
    general_round_trip(4, 11);
}

#[test]
fn general_linear_check_round_trip_8_vars() {
    general_round_trip(8, 22);
}

#[test]
fn general_negative_corrupted_final_rho() {
    // Tamper with one final_rho entry: round consistency still holds, but the
    // final-check sum no longer matches c_n.
    let num_vars = 6;
    let mut state = 33u64;
    let rows = build_layer_array(num_vars, 0xA5A5_A5A5_A5A5_A5A5, &mut state);
    let rhos = build_layer_array(num_vars, 0x5A5A_5A5A_5A5A_5A5A, &mut state);
    let claim = general_linear_check_claim(&rows, &rhos);

    let mut prover_sampler = R2TeichmullerSampler::new(0xBEEFCAFE);
    let (mut proof, _) = prove_linear_check_general(rows.clone(), rhos.clone(), claim, &mut prover_sampler);

    // Corrupt one final_rho.
    proof.final_rho[7] = proof.final_rho[7] + GR32_128::ONE;

    let mut verifier_sampler = R2TeichmullerSampler::new(0xBEEFCAFE);
    let err = verify_linear_check_general(&proof, &mut verifier_sampler)
        .expect_err("corrupted final_rho must be rejected");
    assert_eq!(err, piop_ring::VerifyError::FinalCheckFailed);
}
