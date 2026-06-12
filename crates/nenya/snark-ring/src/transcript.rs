//! Blake3-backed Fiat–Shamir transcript, generic over the base-limb width `L`.
//!
//! The transcript is the single source of randomness for the entire SNARK.
//! It absorbs commitment roots, sumcheck round messages, and PCS-bound
//! values, then squeezes per-sumcheck seeds for [`R2Sampler<L>`] and
//! per-callsite Teichmüller challenges (mult-check target, linear-check
//! batching coefficient).
//!
//! ## Per-round Fiat–Shamir
//!
//! The transcript implements [`piop_ring::Challenger`]: each sumcheck round
//! `observe`s its round message (absorbed into the Blake3 state) and then
//! `sample`s that round's challenge, so every challenge is bound to all prior
//! round messages — the standard per-round FS discipline. The sumcheck provers
//! and verifiers drive the challenger directly; the orchestrators
//! ([`crate::simd`], [`crate::mimd`], [`crate::general`]) absorb the inter-check
//! glue (commitment roots, claims, final evaluations, batching points).

use blake3::Hasher;
use galois_ring::{Gr128, Limb, R2Sampler};
use pcs_ring::Hash;

pub struct Transcript {
    h: Hasher,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        let mut h = Hasher::new();
        h.update(b"snark-ring/v0");
        h.update(&(label.len() as u32).to_le_bytes());
        h.update(label);
        Self { h }
    }

    pub fn absorb_label(&mut self, label: &[u8]) {
        self.h.update(b"L");
        self.h.update(&(label.len() as u32).to_le_bytes());
        self.h.update(label);
    }

    pub fn absorb_root(&mut self, root: &Hash) {
        self.h.update(b"R");
        self.h.update(root);
    }

    /// Absorb one `R_2` element, width-correct (low `BITS/8` bytes per limb).
    pub fn absorb_gr128<L: Limb>(&mut self, x: Gr128<L>) {
        let nbytes = (L::BITS / 8) as usize;
        self.h.update(b"E");
        for c in x.0.iter() {
            for w in c.0.iter() {
                self.h.update(&w.to_u64().to_le_bytes()[..nbytes]);
            }
        }
    }

    pub fn absorb_gr128_slice<L: Limb>(&mut self, label: &[u8], xs: &[Gr128<L>]) {
        self.absorb_label(label);
        self.h.update(&(xs.len() as u32).to_le_bytes());
        for x in xs {
            self.absorb_gr128(*x);
        }
    }

    /// Squeeze 8 bytes and return a fresh `R2Sampler<L>`. Mixes the squeezed
    /// bytes back into the transcript so subsequent reseeds differ.
    pub fn challenge_sampler<L: Limb>(&mut self, label: &[u8]) -> R2Sampler<L> {
        self.absorb_label(label);
        let mut xof = self.h.clone().finalize_xof();
        let mut bytes = [0u8; 8];
        xof.fill(&mut bytes);
        let seed = u64::from_le_bytes(bytes);
        self.h.update(b"S");
        self.h.update(&bytes);
        R2Sampler::new(seed)
    }

    /// One Teichmüller `R_2` element. Bound to current transcript state.
    pub fn challenge_gr128<L: Limb>(&mut self, label: &[u8]) -> Gr128<L> {
        let mut sampler = self.challenge_sampler::<L>(label);
        sampler.sample()
    }

    /// `n` Teichmüller `R_2` elements (e.g. mult-check target r). All drawn
    /// from one reseed.
    pub fn challenge_target<L: Limb>(&mut self, label: &[u8], n: usize) -> Vec<Gr128<L>> {
        let mut sampler = self.challenge_sampler::<L>(label);
        (0..n).map(|_| sampler.sample()).collect()
    }
}

/// Per-round Fiat–Shamir for the `piop-ring` sumchecks: `observe` absorbs the
/// round message, `sample` squeezes that round's `T_2` challenge bound to it.
impl<L: Limb> piop_ring::Challenger<L> for Transcript {
    #[inline]
    fn observe(&mut self, msg: &[Gr128<L>]) {
        self.absorb_gr128_slice(b"sc/round", msg);
    }

    #[inline]
    fn sample(&mut self) -> Gr128<L> {
        self.challenge_gr128(b"sc/challenge")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::GR32_128;
    use piop_ring::Challenger;

    fn poly(seed: u32) -> [GR32_128; 3] {
        let mk = |k: u32| {
            let mut a = [galois_ring::GR32_4::ZERO; 32];
            a[0] = galois_ring::GR32_4([seed.wrapping_add(k), 0, 0, 0]);
            GR32_128(a)
        };
        [mk(1), mk(2), mk(3)]
    }

    /// Per-round binding: from the *same* prior state, observing different
    /// round messages must yield different challenges. This is the property
    /// per-round Fiat–Shamir provides and per-sumcheck FS does not — a guard
    /// against silently regressing the sumcheck challenger to a single seed.
    #[test]
    fn challenge_binds_observed_round_message() {
        let mut t1 = Transcript::new(b"fs-test");
        let mut t2 = Transcript::new(b"fs-test");
        Challenger::observe(&mut t1, &poly(0xAA));
        Challenger::observe(&mut t2, &poly(0xBB));
        let c1: GR32_128 = Challenger::sample(&mut t1);
        let c2: GR32_128 = Challenger::sample(&mut t2);
        assert_ne!(c1, c2, "challenge must depend on the observed round message");
    }

    /// Determinism: identical observe sequences yield identical challenges
    /// (prover and verifier must stay in lock-step).
    #[test]
    fn challenge_is_deterministic() {
        let mut t1 = Transcript::new(b"fs-test");
        let mut t2 = Transcript::new(b"fs-test");
        Challenger::observe(&mut t1, &poly(0xAA));
        Challenger::observe(&mut t2, &poly(0xAA));
        let c1: GR32_128 = Challenger::sample(&mut t1);
        let c2: GR32_128 = Challenger::sample(&mut t2);
        assert_eq!(c1, c2);
    }

    /// A challenge also depends on the round *index* / running state: two
    /// successive samples after the same message differ, so per-round
    /// challenges across a sumcheck are not all equal.
    #[test]
    fn successive_challenges_differ() {
        let mut t = Transcript::new(b"fs-test");
        Challenger::observe(&mut t, &poly(0xAA));
        let c1: GR32_128 = Challenger::sample(&mut t);
        Challenger::observe(&mut t, &poly(0xAA));
        let c2: GR32_128 = Challenger::sample(&mut t);
        assert_ne!(c1, c2);
    }
}
