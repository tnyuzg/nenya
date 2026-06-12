//! Sumcheck proof container.
//!
//! Mirror of `sumcheck-ring`'s `proof.rs` — same coefficient-form representation
//! `s_i(Z) = a0 + a1·Z + a2·Z²`. Over BB4 we *could* store evaluations and have
//! the verifier interpolate (since 2 is a unit in a prime field), but using
//! coefficient form here makes the field-vs-ring file diff smaller, keeps the
//! verifier hot path identical, and costs nothing.

use field::BB4;

#[derive(Clone, Debug)]
pub struct SumcheckProof {
    pub claimed_sum: BB4,
    pub round_polys: Vec<[BB4; 3]>,
    pub final_f: BB4,
    pub final_g: BB4,
}

impl SumcheckProof {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}
