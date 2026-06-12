//! Uniform sampler for `BB4`.
//!
//! Over a field every nonzero element is a unit, so any subset is
//! "unit-separated" by construction — no Teichmüller plumbing needed. We just
//! draw four uniform `BabyBear` words and assemble them into a `BB4` element
//! via `from_base_slice`. `Distribution<BabyBear> for Standard` lives in
//! `p3-baby-bear`, so `rng.gen::<BabyBear>()` is uniform on the prime field.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractExtensionField;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type BB4 = BinomialExtensionField<BabyBear, 4>;

pub struct BB4UniformSampler {
    rng: StdRng,
}

impl BB4UniformSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one uniform BB4 element. Same seed reproduces the same sequence;
    /// prover and verifier must use identically seeded samplers.
    pub fn sample(&mut self) -> BB4 {
        let bases: [BabyBear; 4] = [
            self.rng.gen(),
            self.rng.gen(),
            self.rng.gen(),
            self.rng.gen(),
        ];
        <BB4 as AbstractExtensionField<BabyBear>>::from_base_slice(&bases)
    }
}
