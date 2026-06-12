//! Per-round Fiat–Shamir challenger for the field sumchecks.
//!
//! Field-baseline mirror of [`piop_ring::Challenger`] (the ring stack). Each
//! sumcheck round `observe`s its round-polynomial message and *then* `sample`s
//! that round's challenge, so the challenge is bound to every prior message —
//! the standard per-round Fiat–Shamir discipline. Kept identical to the ring so
//! the two stacks are compared under the same FS regime.
//!
//! The real implementor is `snark-field`'s `Transcript`; [`BB4UniformSampler`]
//! also implements it in a lock-step mode (no-op `observe`) for the
//! crate-internal tests, which drive prover and verifier from the same seed.

use field::{BB4UniformSampler, BB4};

/// Per-round Fiat–Shamir challenger for the field sumcheck provers/verifiers.
///
/// Contract: in every round, `observe` the round message before `sample`ing
/// that round's challenge; prover and verifier must call them in the same order.
pub trait Challenger {
    /// Absorb a round message (the round polynomial's coefficients).
    fn observe(&mut self, msg: &[BB4]);
    /// Squeeze one field challenge, bound to all prior `observe`s.
    fn sample(&mut self) -> BB4;
}

/// Lock-step mode: `observe` is a no-op; challenges come straight from the
/// seeded PRNG. Sound only when prover and verifier share the seed out of band
/// (tests), **never** for the real SNARK — use `snark_field::Transcript` there.
impl Challenger for BB4UniformSampler {
    #[inline]
    fn observe(&mut self, _msg: &[BB4]) {}

    #[inline]
    fn sample(&mut self) -> BB4 {
        BB4UniformSampler::sample(self)
    }
}
