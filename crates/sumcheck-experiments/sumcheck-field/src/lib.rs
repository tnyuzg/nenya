//! Sumcheck protocol over BabyBear^4 = `BinomialExtensionField<BabyBear, 4>`.
//!
//! Implements the same degree-2 single-product sumcheck as `sumcheck-ring`:
//!
//! ```text
//! sum = ∑_{x ∈ {0,1}^n} f(x) · g(x)
//! ```
//!
//! using the *same* file layout (`mle.rs`, `proof.rs`, `prover.rs`,
//! `verifier.rs`, `sampler.rs`) so a side-by-side diff against `sumcheck-ring`
//! makes the structural sameness obvious — only the element type and the
//! sampler differ. The field side is the **comparison reference** for the ring
//! side; we don't plan to micro-optimize this beyond what `p3-field` already
//! gives us.
//!
//! Challenges are sampled uniformly from BB4 by [`sampler::BB4UniformSampler`];
//! over a field every difference is automatically a unit, so any subset is
//! "unit-separated" and there's no Teichmüller plumbing to worry about.
//! Soundness per round: `2 / |BB4| ≈ 2 / 2^124`.
//!
//! Reference for protocol shape: SLOP `slop-sumcheck` 6.0.2
//! (`reduce_sumcheck_to_evaluation`).

pub mod mle;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use field::{BB4UniformSampler, BB4};
pub use mle::Mle;
pub use proof::SumcheckProof;
pub use prover::prove;
pub use verifier::{verify, VerifyError};
