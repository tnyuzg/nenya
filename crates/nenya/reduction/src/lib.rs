//! Packing embedding φ_2 and the three R1CS reductions of paper §4:
//! SIMD (data-parallel), MIMD (circuit-parallel), and general single-circuit.
//!
//! - Two-level Lagrange packing `φ_2: R_0^32 → R_2` with low-degree
//!   representatives.
//! - Mul-with-quotients identity
//!   `φ_2(a)·φ_2(b) = φ_2(a∘b) + q_1·v_α + q_2·v_β`.
//! - SIMD: shared `(A, B, C)`, 32 witnesses → one packed instance with
//!   one mult-relation kernel pair.
//! - MIMD: 32 distinct `(A_i, B_i, C_i)` triples, 32 witnesses → packed
//!   matrices + four kernel pairs (linear A/B/C + mult).
//! - General: single R1CS, witness length divisible by 32, packed via the
//!   Halevi–Shoup cyclic-diagonal decomposition; 32 packed rotation
//!   witnesses + four kernel pairs.

pub mod constants;
pub mod embedding;
pub mod instance;
pub mod matrix;
pub mod reduce;

pub use embedding::{
    in_kernel, in_kernel_with_t1, mul_with_quotients, mul_with_quotients_only, pack_r1, pack_r2,
    pack_r2_matrix, psi_r2, psi_r2_with_t1, unpack_r2,
};
pub use instance::{
    GeneralR1csInstance, MimdR1csInstance, PackedGeneralR1csInstance, PackedMimdR1csInstance,
    PackedSimdR1csInstance, SimdR1csInstance,
};
pub use matrix::{
    matvec_gr1, matvec_gr128, matvec_gr128_csr, matvec_gr128_dense, matvec_gr128_dense_csr,
    matvec_gr1_csr, CsrMatrix, SparseMatrix,
};
pub use reduce::{
    general_reduce_witness, general_to_packed, mimd_reduce_witness, mimd_to_packed,
    pack_general_matrices, pack_mimd_matrices, simd_to_packed, GeneralDiagonals,
    GeneralWitnessParts, MimdMatrices, MimdWitnessParts,
};
