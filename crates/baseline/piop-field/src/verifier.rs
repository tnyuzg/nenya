//! Verifiers — field-side mirror of [`piop_ring::verifier`].

use field::{AbstractField, BB4};

use crate::fs::Challenger;
use crate::proof::{LinearCheckProof, MultCheckProof};

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    RoundCheckFailed { round: usize },
    FinalCheckFailed,
}

pub fn verify_mult_check(
    proof: &MultCheckProof,
    r_target: &[BB4],
    challenger: &mut impl Challenger,
) -> Result<Vec<BB4>, VerifyError> {
    let n = r_target.len();
    assert_eq!(
        proof.round_polys.len(),
        n,
        "mult-check proof has wrong number of rounds"
    );

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(n);

    for (round, [a0, a1, a2, a3]) in proof.round_polys.iter().copied().enumerate() {
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2 + a3;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        // Per-round Fiat–Shamir: observe before sampling (prover's order).
        challenger.observe(&[a0, a1, a2, a3]);
        let r = challenger.sample();
        challenges.push(r);

        c = a0 + r * (a1 + r * (a2 + r * a3));
    }

    let mut eq_final = BB4::one();
    for (rt, &rc) in r_target.iter().zip(challenges.iter()) {
        eq_final *= *rt * rc + (BB4::one() - *rt) * (BB4::one() - rc);
    }

    if c != eq_final * proof.final_a * proof.final_b {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}

pub fn verify_linear_check(
    proof: &LinearCheckProof,
    challenger: &mut impl Challenger,
) -> Result<Vec<BB4>, VerifyError> {
    let n = proof.round_polys.len();
    assert!(n > 0, "linear-check proof has zero rounds");

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(n);

    for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        // Per-round Fiat–Shamir: observe before sampling (prover's order).
        challenger.observe(&[a0, a1, a2]);
        let r = challenger.sample();
        challenges.push(r);

        c = a0 + r * (a1 + r * a2);
    }

    if c != proof.final_a_row * proof.final_w {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}
