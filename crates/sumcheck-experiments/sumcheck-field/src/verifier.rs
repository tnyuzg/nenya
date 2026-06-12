//! Sumcheck verifier for the deg-2 single-product sumcheck over BB4.
//!
//! Mirror of `sumcheck-ring`'s `verifier.rs`.

use field::{AbstractField, BB4UniformSampler, BB4};

use crate::proof::SumcheckProof;

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    RoundCheckFailed { round: usize },
    FinalCheckFailed,
}

pub fn verify(
    proof: &SumcheckProof,
    num_vars: usize,
    sampler: &mut BB4UniformSampler,
) -> Result<Vec<BB4>, VerifyError> {
    assert_eq!(
        proof.round_polys.len(),
        num_vars,
        "proof has wrong number of rounds"
    );

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        // s_i(0) + s_i(1) = 2·a0 + a1 + a2.
        let two = BB4::one() + BB4::one();
        let lhs = two * a0 + a1 + a2;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        let r = sampler.sample();
        challenges.push(r);
        c = a0 + a1 * r + a2 * (r * r);
    }

    if c != proof.final_f * proof.final_g {
        return Err(VerifyError::FinalCheckFailed);
    }

    Ok(challenges)
}
