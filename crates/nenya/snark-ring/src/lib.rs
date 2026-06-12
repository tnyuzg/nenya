//! End-to-end zkSNARKs over `R_2 = GR(2^n, 128)`, generic over the base-limb
//! width `L` (`u8`/`u16`/`u32`/`u64`).
//!
//! Three reduction variants, each a self-contained module with a parallel
//! structure (`prove_*` / `verify_*` / `*SnarkParams` / `*SnarkProof`):
//!
//! - [`simd`] — SIMD R1CS: 32 copies of one R1CS sharing the matrices
//!   (paper §3, SIMD path).
//! - [`mimd`] — MIMD R1CS: 32 *independent* R1CS packed into one (paper §3.2.2).
//! - [`general`] — a single large R1CS over `R_2`.
//!
//! Shared infrastructure: the `R_2` PCS ([`r2_pcs`]), the Fiat–Shamir
//! [`transcript`], and the common verifier [`error::VerifyError`].
//!
//! The `sumcheck-ring` crate holds the *unoptimized reference* kernels; it is
//! used only for benchmarking, never on these prover paths.

pub mod error;
pub mod general;
pub mod mimd;
pub mod r2_pcs;
pub mod simd;
pub mod transcript;

pub use error::VerifyError;
pub use general::{
    measure_linear_decomp, prove_general, prove_general_with_timing, verify_general,
    verify_general_with_timing, GeneralProveTiming, GeneralSnarkParams, GeneralSnarkProof,
    GeneralVerifyTiming, LinearDecomp,
};
pub use mimd::{
    prove_mimd, prove_mimd_with_timing, verify_mimd, verify_mimd_with_timing, MimdProveTiming,
    MimdSnarkParams, MimdSnarkProof, MimdVerifyTiming,
};
pub use r2_pcs::{R2OpeningProof, R2Pcs};
pub use simd::{
    prove_simd, prove_simd_with_timing, verify_simd, verify_simd_with_timing, SimdProveTiming,
    SimdSnarkParams, SimdSnarkProof, SimdVerifyTiming,
};
