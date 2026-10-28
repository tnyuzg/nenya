//! Field SNARK verifier. Mirror of `snark_ring::simd`, minus q1/q2.

use field::{AbstractField, BB4};
use piop_field::{build_eq_table, verify_linear_check, verify_mult_check};
use rayon::prelude::*;

use crate::instance::SparseMatrixBB;
use crate::proof::FieldSnarkProof;
use crate::prove::{absorb_lin_finals, absorb_mult_finals, build_combined_row, FieldSnarkParams};
use crate::transcript::Transcript;

#[derive(Debug, PartialEq, Eq)]
pub enum FieldVerifyError {
    MultSumcheck,
    LinSumcheck,
    LinClaimMismatch,
    LinARowMismatch,
    PcsW,
}

#[derive(Clone, Debug, Default)]
pub struct FieldVerifyTiming {
    pub mult_sumcheck_ms: f64,
    pub linear_sumcheck_ms: f64,
    pub linear_arow_recompute_ms: f64,
    pub pcs_w_ms: f64,
    pub total_ms: f64,
}

pub fn verify_field(
    a: &SparseMatrixBB,
    b: &SparseMatrixBB,
    c: &SparseMatrixBB,
    params: &FieldSnarkParams,
    proof: &FieldSnarkProof,
) -> Result<(), FieldVerifyError> {
    verify_field_with_timing(a, b, c, params, proof).map(|_| ())
}

pub fn verify_field_with_timing(
    a: &SparseMatrixBB,
    b: &SparseMatrixBB,
    c: &SparseMatrixBB,
    params: &FieldSnarkParams,
    proof: &FieldSnarkProof,
) -> Result<FieldVerifyTiming, FieldVerifyError> {
    let log_n = params.log_n;
    let mut timing = FieldVerifyTiming::default();
    let total_t = std::time::Instant::now();
    let mut t;

    let mut transcript = Transcript::new(b"field");
    transcript.absorb_root(&proof.root_w);
    let r_target = transcript.challenge_target(b"mult_check_target", log_n);

    // c(r) = claim_mult (exact over a field; no q1/q2 correction).
    let claim_mult = proof.mult_proof.claimed_sum;
    let cw_at_r = claim_mult;
    transcript.absorb_bb4_slice(b"claim_mult", &[claim_mult]);

    // Mult check (round consistency + final c_n == eq(r,s1)·a·b are internal).
    t = std::time::Instant::now();
    transcript.absorb_label(b"mult_check_sumcheck");
    let s_1 = verify_mult_check(&proof.mult_proof, &r_target, &mut transcript)
        .map_err(|_| FieldVerifyError::MultSumcheck)?;
    timing.mult_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    absorb_mult_finals(&mut transcript, &proof.mult_proof);

    // γ; claim_lin.
    let gamma = transcript.challenge_bb4(b"linear_check_gamma");
    let gamma2 = gamma * gamma;
    let claim_lin = proof.mult_proof.final_a + gamma * proof.mult_proof.final_b + gamma2 * cw_at_r;
    if claim_lin != proof.lin_proof.claimed_sum {
        return Err(FieldVerifyError::LinClaimMismatch);
    }
    transcript.absorb_bb4_slice(b"claim_lin", &[claim_lin]);

    // Linear check (round consistency + final c_n == a_row·w internal).
    t = std::time::Instant::now();
    transcript.absorb_label(b"linear_check_sumcheck");
    let s_2 = verify_linear_check(&proof.lin_proof, &mut transcript)
        .map_err(|_| FieldVerifyError::LinSumcheck)?;
    timing.linear_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;

    // Cross-check final_a_row against verifier-computed M_γ_row(s_2). eq-table
    // builds belong to the linear/piop bucket, not the matrix-eval component, so
    // piop_mxeval stays comparable across systems.
    let teq = std::time::Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    let eq_r = build_eq_table(&r_target);
    let eq_s2 = build_eq_table(&s_2);
    timing.linear_sumcheck_ms += teq.elapsed().as_secs_f64() * 1e3;
    t = std::time::Instant::now();
    // Reuse the prover's parallel row build, then a parallel inner product with
    // eq_s2, which equals M_γ_row(s_2).
    let m_row = build_combined_row(
        a,
        b,
        c,
        eq_s1.evals(),
        eq_r.evals(),
        gamma,
        gamma2,
        params.m(),
    );
    let m_gamma_at_s2 = m_row
        .evals()
        .par_iter()
        .zip(eq_s2.evals().par_iter())
        .map(|(&x, &y)| x * y)
        .reduce(|| BB4::zero(), |u, v| u + v);
    if m_gamma_at_s2 != proof.lin_proof.final_a_row {
        return Err(FieldVerifyError::LinARowMismatch);
    }
    timing.linear_arow_recompute_ms = t.elapsed().as_secs_f64() * 1e3;
    absorb_lin_finals(&mut transcript, &proof.lin_proof);

    // PCS open of w at s_2 must match lin_proof.final_w.
    t = std::time::Instant::now();
    if !params
        .pcs_w
        .verify(proof.root_w, &s_2, proof.lin_proof.final_w, &proof.open_w)
    {
        return Err(FieldVerifyError::PcsW);
    }
    timing.pcs_w_ms = t.elapsed().as_secs_f64() * 1e3;

    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    Ok(timing)
}
