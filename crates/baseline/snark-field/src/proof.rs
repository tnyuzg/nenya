//! Field SNARK proof container.

use pcs_field::{FieldOpening, Hash};
use piop_field::{LinearCheckProof, MultCheckProof};

/// Full field SNARK proof for a satisfied R1CS instance.
#[derive(PartialEq)]
pub struct FieldSnarkProof {
    /// Witness commitment root. (Only one — no q1/q2.)
    pub root_w: Hash,
    /// Mult-check (deg-3) proof: `∑_h eq(r,h)·(A·w)(h)·(B·w)(h) = c(r)`.
    pub mult_proof: MultCheckProof,
    /// Linear-check (deg-2) proof for the γ-combined row.
    pub lin_proof: LinearCheckProof,
    /// PCS opening of `w` at `s_2`.
    pub open_w: FieldOpening,
}

impl FieldSnarkProof {
    /// Approximate proof size in bytes (`BB4` = 16 bytes).
    pub fn size_bytes(&self) -> usize {
        const E: usize = 16; // sizeof(BB4)
        let roots = 32;
        // mult: claim + 2 finals + 4 coeffs/round.
        let mult = 3 * E + self.mult_proof.round_polys.len() * 4 * E;
        // lin: claim + 2 finals + 3 coeffs/round.
        let lin = 3 * E + self.lin_proof.round_polys.len() * 3 * E;
        roots + mult + lin + self.open_w.size_bytes()
    }
}
