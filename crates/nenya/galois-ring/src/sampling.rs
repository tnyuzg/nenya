//! Uniform sampler for the Teichmüller set `T_2 ⊂ R_2`.
//!
//! `T_2 = { x ∈ R_2 : x^{2^{128}} = x }` is the unique multiplicative lift
//! of `F_{2^{128}}` into `R_2` and has size `2^{128}`. It is the canonical
//! unit-separated challenge set for sumcheck over `R_2` (the paper) —
//! pairwise differences are all units in `R_2`, which is what Schwartz–Zippel
//! needs to stay honest over a ring with zero divisors.
//!
//! This module lives next to [`crate::teichmuller_lift_r2`] because the
//! sampler is just "fill a natural lift with 128 bits of entropy, then Hensel-
//! lift." Keeping the wrapper adjacent to the primitive means the one thing
//! the sampler has to get right — spreading 128 independent bits into the
//! natural-lift slots before lifting — is a one-file concern.
//!
//! ## Entropy layout
//!
//! `R_2 = R_1[β] / (f(β))` with `R_1 = (ℤ/2³²)[α] / (α⁴ + α + 1)`. The mod-2
//! image of an `R_2` element is a polynomial in β of degree < 32 with
//! coefficients in `F_{16} = F_2[α] / (α⁴+α+1)` — exactly `32 · 4 = 128`
//! independent bits. Filling all 128 `u32` coefficients with random bits
//! gives 4096 bits of entropy total; only the LSB of each is load-bearing
//! for uniqueness, but the extra precision is collapsed for free by the
//! Hensel lift.

use core::marker::PhantomData;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::limb::Limb;
use crate::{teichmuller_lift_r2, Gr128, Gr4};

/// Seed-reproducible sampler from `T_2 ⊂ R_2 = GR(2^n, 128)`, generic over the
/// base limb width `L`.
///
/// Prover and verifier must use identically seeded samplers — the sumcheck
/// challenge sequences diverge otherwise. This is a prototype-level
/// primitive; a real protocol plugs in a hash-based challenger upstream.
///
/// Note: one `L::random` draw per limb. The byte stream consumed per sample
/// therefore depends on the width, so a given seed yields different (but
/// equally valid) challenges across widths — expected and harmless.
pub struct R2Sampler<L: Limb> {
    rng: StdRng,
    _l: PhantomData<L>,
}

impl<L: Limb> R2Sampler<L> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            _l: PhantomData,
        }
    }

    /// Sample one Teichmüller element of `R_2`. Cost is dominated by the
    /// Hensel lift; the random fill is negligible.
    pub fn sample(&mut self) -> Gr128<L> {
        let mut arr = [Gr4::<L>::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = Gr4([
                L::random(&mut self.rng),
                L::random(&mut self.rng),
                L::random(&mut self.rng),
                L::random(&mut self.rng),
            ]);
        }
        teichmuller_lift_r2(Gr128(arr))
    }
}
