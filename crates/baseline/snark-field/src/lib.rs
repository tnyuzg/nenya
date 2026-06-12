//! End-to-end **field** zkSNARK over BabyBear — apples-to-apples baseline for
//! [`snark_ring`](../snark_ring/index.html).
//!
//! Same protocol skeleton as the ring SNARK, minus everything packing-related:
//!
//! 1. Commit the witness `w ∈ BB^m` once via [`pcs_field::FieldPcs`].
//!    (No φ-packing ⇒ **no `q1, q2` quotient witnesses and no extra
//!    commitments** — the ring SNARK's two extra PCS commits have no analogue.)
//! 2. Mult check at `r ∈ BB4^{log n}`:
//!    `∑_h eq(r,h)·(A·w)(h)·(B·w)(h) = (C·w)(r)` — exact over a field, so the
//!    claim is just `c(r)` (no `v_α/v_β` corrections).
//! 3. Linear check batched with `γ`:
//!    `a(s₁)+γ·b(s₁)+γ²·c(r) = ∑_h (A_row(s₁,h)+γ·B_row(s₁,h)+γ²·C_row(r,h))·w(h)`.
//! 4. PCS-open `w` at `s₂`.
//!
//! Challenges are uniform `BB4` (every nonzero field element is a unit, so no
//! Teichmüller machinery). Same per-sumcheck Fiat–Shamir caveat as `snark-ring`.

pub mod instance;
pub mod proof;
pub mod prove;
pub mod transcript;
pub mod verify;

pub use instance::{FieldR1csInstance, SparseMatrixBB};
pub use pcs_field::FieldPcs;
pub use proof::FieldSnarkProof;
pub use prove::{prove_field, prove_field_with_timing, FieldProveTiming, FieldSnarkParams};
pub use verify::{verify_field, verify_field_with_timing, FieldVerifyError, FieldVerifyTiming};
