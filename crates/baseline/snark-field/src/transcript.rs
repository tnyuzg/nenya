//! Blake3-backed Fiat–Shamir transcript over `BB4`. Field mirror of
//! `snark_ring::transcript`. **Per-round** Fiat–Shamir: implements
//! [`piop_field::Challenger`], so each sumcheck round `observe`s its message and
//! then `sample`s that round's challenge.

use blake3::Hasher;
use field::{AbstractExtensionField, BB4UniformSampler, BB, BB4};
use p3_field::PrimeField32;
use pcs_field::Hash;

pub struct Transcript {
    h: Hasher,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        let mut h = Hasher::new();
        h.update(b"snark-field/v0");
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

    pub fn absorb_bb4(&mut self, x: BB4) {
        self.h.update(b"E");
        for c in <BB4 as AbstractExtensionField<BB>>::as_base_slice(&x) {
            self.h.update(&c.as_canonical_u32().to_le_bytes());
        }
    }

    pub fn absorb_bb4_slice(&mut self, label: &[u8], xs: &[BB4]) {
        self.absorb_label(label);
        self.h.update(&(xs.len() as u32).to_le_bytes());
        for x in xs {
            self.absorb_bb4(*x);
        }
    }

    /// Squeeze 8 bytes → a fresh `BB4UniformSampler`; mix the bytes back in.
    pub fn challenge_sampler(&mut self, label: &[u8]) -> BB4UniformSampler {
        self.absorb_label(label);
        let mut xof = self.h.clone().finalize_xof();
        let mut bytes = [0u8; 8];
        xof.fill(&mut bytes);
        let seed = u64::from_le_bytes(bytes);
        self.h.update(b"S");
        self.h.update(&bytes);
        BB4UniformSampler::new(seed)
    }

    pub fn challenge_bb4(&mut self, label: &[u8]) -> BB4 {
        self.challenge_sampler(label).sample()
    }

    pub fn challenge_target(&mut self, label: &[u8], n: usize) -> Vec<BB4> {
        let mut s = self.challenge_sampler(label);
        (0..n).map(|_| s.sample()).collect()
    }
}

/// Per-round Fiat–Shamir for the `piop-field` sumchecks: `observe` absorbs the
/// round message, `sample` squeezes that round's challenge bound to it.
impl piop_field::Challenger for Transcript {
    #[inline]
    fn observe(&mut self, msg: &[BB4]) {
        self.absorb_bb4_slice(b"sc/round", msg);
    }

    #[inline]
    fn sample(&mut self) -> BB4 {
        self.challenge_bb4(b"sc/challenge")
    }
}
