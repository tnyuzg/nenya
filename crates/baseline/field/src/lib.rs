//! Shared field types for the BabyBear side of the workspace.
//!
//! - `BB`  = `BabyBear` (prime field, 2^31 − 2^27 + 1, ~31 bits)
//! - `BB4` = `BinomialExtensionField<BabyBear, 4>` (~124 bits, the sumcheck
//!    challenge field)
//! - [`BB4UniformSampler`] draws uniform `BB4` challenges from a seeded RNG.
//!
//! Trait re-exports from `p3_field` (`AbstractField`, `AbstractExtensionField`,
//! `Field`) are surfaced here so downstream crates can stay on a single
//! `field::*` import surface.

pub mod sampler;

pub use p3_baby_bear::BabyBear;
pub use p3_field::extension::BinomialExtensionField;
pub use p3_field::{AbstractExtensionField, AbstractField, Field};

pub use sampler::BB4UniformSampler;

/// BabyBear prime field. Alias over `p3_baby_bear::BabyBear`.
pub type BB = BabyBear;

/// Degree-4 binomial extension of BabyBear. ~124-bit sumcheck challenge field.
pub type BB4 = BinomialExtensionField<BabyBear, 4>;
