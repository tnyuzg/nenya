//! Sumcheck verifiers (deg-2 and deg-3 single product).

use galois_ring::{R2TeichmullerSampler, GR32_128};

use crate::proof::{SumcheckProof, SumcheckProofDeg2};

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Round `i` failed `s_i(0) + s_i(1) == c_{i-1}`.
    RoundCheckFailed { round: usize },
    /// Final evaluation did not match the product of claimed finals.
    FinalCheckFailed,
}

/// Verify a deg-3 sumcheck proof. Returns the binding point on success.
pub fn verify(
    proof: &SumcheckProof,
    num_vars: usize,
    sampler: &mut R2TeichmullerSampler,
) -> Result<Vec<GR32_128>, VerifyError> {
    assert_eq!(
        proof.round_polys.len(),
        num_vars,
        "proof has wrong number of rounds"
    );

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, [a0, a1, a2, a3]) in proof.round_polys.iter().copied().enumerate() {
        // s_i(0) + s_i(1) = 2·a0 + a1 + a2 + a3.
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2 + a3;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        let r = sampler.sample();
        challenges.push(r);
        c = a0 + r * (a1 + r * (a2 + r * a3));
    }

    if c != proof.final_f1 * proof.final_f2 * proof.final_f3 {
        return Err(VerifyError::FinalCheckFailed);
    }

    Ok(challenges)
}

/// Verify a deg-2 sumcheck proof. Returns the binding point on success.
pub fn verify_deg2(
    proof: &SumcheckProofDeg2,
    num_vars: usize,
    sampler: &mut R2TeichmullerSampler,
) -> Result<Vec<GR32_128>, VerifyError> {
    assert_eq!(
        proof.round_polys.len(),
        num_vars,
        "proof has wrong number of rounds"
    );

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        let r = sampler.sample();
        challenges.push(r);
        c = a0 + r * (a1 + r * a2);
    }

    if c != proof.final_f1 * proof.final_f2 {
        return Err(VerifyError::FinalCheckFailed);
    }

    Ok(challenges)
}
