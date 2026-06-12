//! Proof containers — field-side mirror of [`piop_ring::proof`].
//!
//! Round messages are stored in coefficient form. Over a field this is just
//! a stylistic choice (we could equally well send Lagrange evaluations and
//! interpolate on the verifier), but matching the ring side's encoding makes
//! the bench numbers easier to compare and the inner-loop arithmetic line up.

use crate::BB4;

#[derive(Clone, Debug, PartialEq)]
pub struct MultCheckProof {
    pub claimed_sum: BB4,
    /// `s_i(Z) = a0 + a1·Z + a2·Z² + a3·Z³`.
    pub round_polys: Vec<[BB4; 4]>,
    pub final_a: BB4,
    pub final_b: BB4,
}

impl MultCheckProof {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCheckProof {
    pub claimed_sum: BB4,
    /// `s_i(Z) = a0 + a1·Z + a2·Z²`.
    pub round_polys: Vec<[BB4; 3]>,
    pub final_a_row: BB4,
    pub final_w: BB4,
}

impl LinearCheckProof {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}
