//! Verifier error type, shared by all three SNARK variants (`simd`, `mimd`,
//! `general`). Kept in a neutral module so no variant owns it.

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    MultSumcheck,
    LinSumcheck,
    LinClaimMismatch,
    LinARowMismatch,
    PcsW,
    PcsE,
    /// An opened error column lies outside `ker ψ` — the free-subspace check
    /// of [`crate::r2_pcs::columns_in_kernel`]. Without this, a prover with
    /// `a∘b ≠ c` could pass by committing a non-kernel `ê`.
    KernelCheck,
}
