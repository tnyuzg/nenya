//! Sumcheck proof containers.
//!
//! Two proof types for two reference provers:
//!
//! - [`SumcheckProofDeg2`] for `∑ f1(h)·f2(h)` — 3-coefficient round messages.
//! - [`SumcheckProof`] (deg-3) for `∑ f1(h)·f2(h)·f3(h)` — 4-coefficient.
//!
//! Both store round polys in **monomial-coefficient form**. Lagrange-on-
//! `{0,1,…,d}` would require dividing by a multiple of `2`, and `2` is not
//! a unit in `R₀ = Z/2³²`, so coefficient form is the only division-free
//! encoding that works over arbitrary degrees.

use galois_ring::GR32_128;

/// Proof of `sum = ∑_h f1(h) · f2(h) · f3(h)` (deg-3 single product).
///
/// Round messages: `s_i(Z) = a0 + a1·Z + a2·Z² + a3·Z³`.
///
/// The verifier's challenge sequence is *not* stored: it is re-derived by
/// stepping a [`galois_ring::R2TeichmullerSampler`] in lock-step with the
/// prover. Both sides must use identically seeded samplers.
#[derive(Clone, Debug)]
pub struct SumcheckProof {
    pub claimed_sum: GR32_128,
    pub round_polys: Vec<[GR32_128; 4]>,
    pub final_f1: GR32_128,
    pub final_f2: GR32_128,
    pub final_f3: GR32_128,
}

impl SumcheckProof {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}

/// Proof of `sum = ∑_h f1(h) · f2(h)` (deg-2 single product).
///
/// Round messages: `s_i(Z) = a0 + a1·Z + a2·Z²`.
#[derive(Clone, Debug)]
pub struct SumcheckProofDeg2 {
    pub claimed_sum: GR32_128,
    pub round_polys: Vec<[GR32_128; 3]>,
    pub final_f1: GR32_128,
    pub final_f2: GR32_128,
}

impl SumcheckProofDeg2 {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}
