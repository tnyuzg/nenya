//! Fiat–Shamir challenger abstraction for the ring sumchecks.
//!
//! Per-round absorb-then-squeeze, mirroring Plonky3's `CanObserve` / `CanSample`
//! split (`Plonky3/challenger/src/lib.rs`): each sumcheck round `observe`s its
//! round-polynomial message and *then* `sample`s the next fold challenge, so the
//! challenge is bound to that round's message and all prior ones. This is the
//! standard **per-round** Fiat–Shamir discipline — without it (challenges drawn
//! from a single seed fixed at the start of the sumcheck) a malicious prover
//! could pick round messages after seeing the challenges.
//!
//! The real implementor is `snark-ring`'s `Transcript` (a Blake3 transcript).
//! [`R2Sampler`] also implements the trait in a degenerate **lock-step** mode
//! where `observe` is a no-op and challenges come straight from the seeded
//! PRNG. That mode is used only by the crate-internal sumcheck unit tests and
//! the `sumcheck-ring` cross-checks, which drive prover and verifier from the
//! same seed out of band and so need no transcript binding. It is **not** sound
//! for the real SNARK — there, always pass a `Transcript`.

use galois_ring::{Gr128, Limb, R2Sampler};

/// Per-round Fiat–Shamir challenger for the sumcheck provers and verifiers.
///
/// Contract: in every sumcheck round, call [`observe`](Challenger::observe) on
/// the round message before [`sample`](Challenger::sample)ing that round's
/// challenge. Prover and verifier must `observe`/`sample` in the same order so
/// their challenge sequences agree.
pub trait Challenger<L: Limb> {
    /// Absorb a round message (the round polynomial's coefficients) into the
    /// transcript.
    fn observe(&mut self, msg: &[Gr128<L>]);

    /// Squeeze one Teichmüller challenge in `T_2 ⊂ R_2`, bound to all prior
    /// `observe`s. Challenges must lie in the unit-separated set `T_2` for the
    /// Schwartz–Zippel-over-rings soundness bound to hold (paper §2.1).
    fn sample(&mut self) -> Gr128<L>;
}

/// Lock-step mode: `observe` is a no-op; challenges come straight from the
/// seeded PRNG. Sound only when prover and verifier share the seed out of band
/// (test / cross-check harnesses), **never** for the real SNARK — use
/// `snark_ring::Transcript` there.
impl<L: Limb> Challenger<L> for R2Sampler<L> {
    #[inline]
    fn observe(&mut self, _msg: &[Gr128<L>]) {}

    #[inline]
    fn sample(&mut self) -> Gr128<L> {
        R2Sampler::sample(self)
    }
}
