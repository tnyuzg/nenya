//! PIOP sumcheck instances over `BB4 = BinomialExtensionField<BabyBear, 4>`.
//!
//! Field-side mirror of [`piop-ring`](../piop_ring/index.html). Same module
//! layout, same protocol shape, same `{s(0), s(1), s(∞)}` round-message
//! trick on both internal sumchecks. The element type is the only difference,
//! and a few ring-specific tricks (the u32-aware matvec via `mul_by_gr1`)
//! intentionally have no field analogue.
//!
//! Two checks:
//!
//! 1. **Multiplication check** ([`mult_check`]):
//!    `∑_h eq(r, h) · a(h) · b(h)` — deg-3, naive 3-table.
//!
//! 2. **Linear check** ([`linear_check`]):
//!    `∑_h A_row[h] · w(h)` where `A_row[h] = ∑_y eq(r, y) · A[y, h]`.
//!    Two factors, deg-2. The matrix entries are BB4 (not "small"); the
//!    matvec uses the generic `*`. We don't pretend we have a structural
//!    win here — that's the whole point of the comparison.
//!
//! ## Challenger (per-round Fiat–Shamir)
//!
//! Every sumcheck prover/verifier takes a [`Challenger`] (the [`fs`] module):
//! each round `observe`s its message then `sample`s a challenge bound to it,
//! mirroring `piop_ring`. `snark-field`'s `Transcript` is the real implementor;
//! [`BB4UniformSampler`] implements it in a lock-step mode (no-op `observe`) for
//! the crate-internal tests only.

pub mod eq;
pub mod fs;
pub mod linear_check;
pub mod mle;
pub mod mult_check;
pub mod proof;
pub mod sparse_matrix;
pub mod verifier;

pub use eq::build_eq_table;
pub use field::{BB4UniformSampler, BB4};
pub use fs::Challenger;
pub use linear_check::{prove_linear_check, prove_linear_check_sumcheck_only};
pub use mle::Mle;
pub use mult_check::prove_mult_check;
pub use proof::{LinearCheckProof, MultCheckProof};
pub use sparse_matrix::SparseMatrix;
pub use verifier::{verify_linear_check, verify_mult_check, VerifyError};
