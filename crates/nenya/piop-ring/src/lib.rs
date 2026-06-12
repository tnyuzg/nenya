//! PIOP sumcheck instances over `R₂ = GR(2³², 128)`.
//!
//! Two specialized sumchecks, each with its own prover/verifier pair:
//!
//! 1. **Multiplication check** ([`mult_check`]):
//!    ```text
//!    sum = ∑_{h ∈ {0,1}^n} eq(r, h) · a(h) · b(h)
//!    ```
//!    Three multilinear factors, degree 3 per round. `r ∈ T_2^n` is the
//!    target point fixed by the upstream protocol; `a` and `b` are GR(2³², 128)
//!    multilinears in `n` variables.
//!
//! 2. **Linear check** ([`linear_check_simd`]):
//!    ```text
//!    sum = ∑_{h ∈ {0,1}^n} A_row[h] · w(h)
//!    ```
//!    where `A_row[h] = ∑_y eq(r, y) · A[y, h]` is built once from a sparse
//!    `R_0`-valued matrix `A` (entries are `u32`s). Two factors, degree 2 per
//!    round, but the matrix-vector pre-step exploits the u32-ness of `A` via
//!    [`galois_ring::GR32_128::mul_by_gr1`] (~15× cheaper per matvec entry than
//!    the generic schoolbook).
//!
//! ## Two general optimizations
//!
//! Both internal sumchecks apply two ring-specific tricks beyond the
//! [`sumcheck-ring`](../sumcheck_ring/index.html) baseline:
//!
//! - **`{s(0), s(1), s(∞)}` round message** trades 1 GR32_128 mul for ~2
//!   adds per fold-pair (linear check) — adds are ~30× cheaper than muls,
//!   so this is a clear win. The [`mult_check`] uses Gruen's optimization:
//!   factors the active-slot `eq_one` out and applies `{0, 1, ∞}` to the
//!   remaining deg-2 inner factor (6 muls/pair instead of 10).
//! - **`mul_by_gr1`** in the linear-check matvec stage skips ~95% of
//!   the schoolbook's inner work whenever one operand is a u32 lifted to
//!   the constant term of `R₂`.
//!
//! ## Challenger (per-round Fiat–Shamir)
//!
//! Every sumcheck prover/verifier takes a [`Challenger`] (the [`fs`] module):
//! each round `observe`s its round message then `sample`s a `T_2` challenge
//! bound to it. `snark-ring`'s `Transcript` is the real (Blake3) implementor;
//! [`galois_ring::R2TeichmullerSampler`] implements it in a lock-step mode
//! (no-op `observe`, PRNG `sample`) for the crate-internal unit tests and the
//! `sumcheck-ring` cross-checks only.
//!
//! ## What this crate is *not*
//!
//! - **Not wired to `reduction::PackedSimdR1csInstance`** (or the MIMD/general
//!   variants) yet — the PIOP takes raw sparse matrices and witness vectors.
//!   Wiring R1CS into the PIOP is a follow-up.

pub mod eq;
pub mod error;
pub mod fs;
pub mod linear_check_general;
pub mod linear_check_mimd;
pub mod linear_check_simd;
pub mod mle;
pub mod mult_check;
pub mod sparse_matrix;

pub use eq::build_eq_table;
pub use error::VerifyError;
pub use fs::Challenger;
pub use galois_ring::R2TeichmullerSampler;
pub use linear_check_general::{
    prove_linear_check_general, verify_linear_check_general, LinearCheckGeneralProof, LAYERS,
};
pub use linear_check_mimd::{
    prove_linear_check_mimd, prove_linear_check_mimd_sumcheck_only, verify_linear_check_mimd,
    LinearCheckMimdProof,
};
pub use linear_check_simd::{
    prove_linear_check_simd, prove_linear_check_simd_sumcheck_only, verify_linear_check_simd,
    LinearCheckSimdProof,
};
pub use mle::Mle;
pub use mult_check::{
    prove_mult_check, prove_mult_check_sub, verify_mult_check, verify_mult_check_sub,
    MultCheckProof, MultCheckSubProof,
};
pub use sparse_matrix::SparseMatrix;
