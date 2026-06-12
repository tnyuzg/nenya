//! Sumcheck protocol over R₂ = GR(2³², 128).
//!
//! Two reference provers:
//!
//! - [`prove`] — **deg-3 single product** `∑ f1·f2·f3`. Cross-checked
//!   against `piop-ring::prove_mult_check`.
//! - [`prove_deg2`] — **deg-2 single product** `∑ f1·f2`. Cross-checked
//!   against `piop-ring::prove_linear_check`.
//!
//! Both use monomial-coefficient-form round messages (Lagrange-on-
//! `{0,…,d}` would divide by a multiple of `2`, which is not a unit in
//! `Z/2³²`). Challenges come from the Teichmüller set `T_2 ⊂ R_2` via
//! [`galois_ring::R2TeichmullerSampler`] — unit-separated, size `2¹²⁸`,
//! the canonical exceptional set (the paper).
//!
//! Reference for protocol shape: SLOP `slop-sumcheck` 6.0.2
//! (`reduce_sumcheck_to_evaluation`). We use the same round structure but
//! drop the RLC-of-many-polys generality and the `DuplexChallenger`
//! plumbing.

pub mod forms;
pub mod mle;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use forms::{dot, prove_form1, prove_form2, Form1Proof};
pub use mle::Mle;
pub use proof::{SumcheckProof, SumcheckProofDeg2};
pub use prover::{prove, prove_deg2};
pub use verifier::{verify, verify_deg2, VerifyError};
