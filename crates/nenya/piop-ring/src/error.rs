//! Shared verifier error type for all PIOP sumchecks in this crate.

/// Returned by the `verify_*` functions in [`crate::mult_check`],
/// [`crate::linear_check_simd`], [`crate::linear_check_mimd`], and
/// [`crate::linear_check_general`].
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Round `i` failed `s_i(0) + s_i(1) == c_{i-1}`.
    RoundCheckFailed { round: usize },
    /// Final evaluation didn't match the protocol-specific final equation.
    FinalCheckFailed,
}
