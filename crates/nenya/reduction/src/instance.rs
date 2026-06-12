//! R1CS instance data structures for the three reductions of paper §3.2:
//! SIMD (data-parallel), MIMD (circuit-parallel), and general single-circuit.

use crate::constants::{constants, HasPackingConstants};
use crate::embedding::{pack_r2, unpack_r2};
use crate::matrix::{matvec_gr1, matvec_gr128, matvec_gr128_dense, SparseMatrix};
use galois_ring::{Gr1, Gr128, Limb};

/// SIMD R1CS instance (paper §3.2.1): 32 R1CS instances over R_0 = Z/2^n
/// sharing one (A, B, C). `L` defaults to `u32`.
#[derive(Clone, Debug)]
pub struct SimdR1csInstance<L: Limb = u32> {
    /// Constraint matrix `A` of shape `(n × m)`, `R_0`-valued.
    pub a: SparseMatrix<Gr1<L>>,
    /// Constraint matrix `B`, same shape as `A`.
    pub b: SparseMatrix<Gr1<L>>,
    /// Constraint matrix `C`, same shape as `A`.
    pub c: SparseMatrix<Gr1<L>>,
    /// 32 witnesses, each of length `m = a.cols`.
    pub witnesses: [Vec<Gr1<L>>; 32],
}

impl<L: Limb> SimdR1csInstance<L> {
    /// True iff every one of the 32 witnesses satisfies `(A·w)∘(B·w) == C·w`.
    pub fn is_satisfied(&self) -> bool {
        for w in &self.witnesses {
            let aw = matvec_gr1(&self.a, w);
            let bw = matvec_gr1(&self.b, w);
            let cw = matvec_gr1(&self.c, w);
            for i in 0..aw.len() {
                if aw[i] * bw[i] != cw[i] {
                    return false;
                }
            }
        }
        true
    }
}

/// SIMD-reduced packed R1CS instance over R_2 = GR(2^32, 128).
///
/// Produced by `simd_to_packed`. The matrices are still `R_0`-valued and are
/// lifted to `R_2` at mat·vec time; the per-row quotient pair
/// `(q1[i], q2[i])` corrects the `(A·w_phi)_i · (B·w_phi)_i` product back to
/// `(C·w_phi)_i` modulo `v_α`, `v_β`.
#[derive(Clone, Debug)]
pub struct PackedSimdR1csInstance<L: Limb = u32> {
    /// Same `R_0`-valued `A` as the source `SimdR1csInstance`.
    pub a: SparseMatrix<Gr1<L>>,
    /// Same `R_0`-valued `B`.
    pub b: SparseMatrix<Gr1<L>>,
    /// Same `R_0`-valued `C`.
    pub c: SparseMatrix<Gr1<L>>,
    /// Column-wise φ_2-packed witness: `w_phi[j] = φ_2((w_0[j], …, w_31[j]))`.
    pub w_phi: Vec<Gr128<L>>,
    /// Per-row **combined kernel error** `e[i] = q1[i]·v_α + q2[i]·v_β ∈ ker(ψ)`.
    /// The verifier only ever uses `q1,q2` through this fixed combination, so the
    /// SNARK commits the single `e` (one PCS / one query). `q1,q2` are the
    /// reduce-internal low-degree decomposition that produces `e` and remain its
    /// conceptual membership witness for `e ∈ ker(ψ)`.
    pub e: Vec<Gr128<L>>,
}

impl<L: HasPackingConstants> PackedSimdR1csInstance<L> {
    /// True iff `(A·w_phi)∘(B·w_phi) == C·w_phi + e`.
    pub fn is_satisfied(&self) -> bool {
        let aw = matvec_gr128(&self.a, &self.w_phi);
        let bw = matvec_gr128(&self.b, &self.w_phi);
        let cw = matvec_gr128(&self.c, &self.w_phi);
        for i in 0..aw.len() {
            let lhs = aw[i] * bw[i];
            let rhs = cw[i] + self.e[i];
            if lhs != rhs {
                return false;
            }
        }
        true
    }
}

/// MIMD R1CS instance (paper §3.2.2): 32 R1CS instances over R_0 with
/// independent (A_i, B_i, C_i) matrices.
#[derive(Clone, Debug)]
pub struct MimdR1csInstance<L: Limb = u32> {
    /// 32 constraint matrices `A_i`, all of shape `(n × m)`.
    pub a_per_instance: [SparseMatrix<Gr1<L>>; 32],
    /// 32 constraint matrices `B_i`.
    pub b_per_instance: [SparseMatrix<Gr1<L>>; 32],
    /// 32 constraint matrices `C_i`.
    pub c_per_instance: [SparseMatrix<Gr1<L>>; 32],
    /// 32 witnesses, each of length `m`.
    pub witnesses: [Vec<Gr1<L>>; 32],
}

impl<L: Limb> MimdR1csInstance<L> {
    /// True iff every instance satisfies `(A_i·w_i)∘(B_i·w_i) == C_i·w_i`.
    pub fn is_satisfied(&self) -> bool {
        for i in 0..32 {
            let w = &self.witnesses[i];
            let aw = matvec_gr1(&self.a_per_instance[i], w);
            let bw = matvec_gr1(&self.b_per_instance[i], w);
            let cw = matvec_gr1(&self.c_per_instance[i], w);
            for j in 0..aw.len() {
                if aw[j] * bw[j] != cw[j] {
                    return false;
                }
            }
        }
        true
    }
}

/// MIMD-reduced packed R1CS instance over R_2 = GR(2^32, 128).
///
/// Produced by `mimd_to_packed`. Matrices are now `R_2`-valued (each entry
/// in `Im(φ)`) — the entry-wise packing across the 32 instances. Compared
/// to the SIMD output we carry three additional kernel-witness pairs
/// `(q1^M, q2^M)` for the linear relations `Â·ŵ`, `B̂·ŵ`, `Ĉ·ŵ`, on top
/// of the multiplication-relation pair.
#[derive(Clone, Debug)]
pub struct PackedMimdR1csInstance<L: Limb = u32> {
    /// Packed `Â`, with `Â[j,k] = pack_r2((A_0[j,k], …, A_31[j,k]))`.
    pub a_hat: SparseMatrix<Gr128<L>>,
    /// Packed `B̂`.
    pub b_hat: SparseMatrix<Gr128<L>>,
    /// Packed `Ĉ`.
    pub c_hat: SparseMatrix<Gr128<L>>,
    /// Column-wise φ_2-packed witness, length `m`.
    pub w_phi: Vec<Gr128<L>>,
    /// Linear-relation kernel witness for `A`: `Â·w_phi - ã = q1_a·v_α + q2_a·v_β`.
    pub q1_a: Vec<Gr128<L>>,
    pub q2_a: Vec<Gr128<L>>,
    /// Linear-relation kernel witness for `B`.
    pub q1_b: Vec<Gr128<L>>,
    pub q2_b: Vec<Gr128<L>>,
    /// Linear-relation kernel witness for `C`.
    pub q1_c: Vec<Gr128<L>>,
    pub q2_c: Vec<Gr128<L>>,
    /// Multiplication-relation kernel witness: `ã ∘ b̃ - ĉ = q1_mult·v_α + q2_mult·v_β`.
    pub q1_mult: Vec<Gr128<L>>,
    pub q2_mult: Vec<Gr128<L>>,
}

impl<L: HasPackingConstants> PackedMimdR1csInstance<L> {
    /// True iff
    /// `(Â·w_phi - ê^A) ∘ (B̂·w_phi - ê^B) == (Ĉ·w_phi - ê^C) + ê^mult`,
    /// where each `ê^M = q1^M·v_α + q2^M·v_β`.
    pub fn is_satisfied(&self) -> bool {
        let cst = constants::<L>();
        let v_alpha = Gr128::<L>::from_gr4(cst.v_alpha);
        let v_beta = cst.v_beta;
        let aw = matvec_gr128_dense(&self.a_hat, &self.w_phi);
        let bw = matvec_gr128_dense(&self.b_hat, &self.w_phi);
        let cw = matvec_gr128_dense(&self.c_hat, &self.w_phi);
        let n = aw.len();
        debug_assert_eq!(self.q1_a.len(), n);
        debug_assert_eq!(self.q1_mult.len(), n);
        for i in 0..n {
            let a_tilde = aw[i] - (self.q1_a[i] * v_alpha + self.q2_a[i] * v_beta);
            let b_tilde = bw[i] - (self.q1_b[i] * v_alpha + self.q2_b[i] * v_beta);
            let c_tilde = cw[i] - (self.q1_c[i] * v_alpha + self.q2_c[i] * v_beta);
            let lhs = a_tilde * b_tilde;
            let rhs = c_tilde + self.q1_mult[i] * v_alpha + self.q2_mult[i] * v_beta;
            if lhs != rhs {
                return false;
            }
        }
        true
    }
}

/// General-circuit R1CS instance (paper §3.2.3): a single R1CS over R_0
/// whose witness length `n` is divisible by 32.
#[derive(Clone, Debug)]
pub struct GeneralR1csInstance<L: Limb = u32> {
    /// Square constraint matrix `A` of shape `(n × n)`, `R_0`-valued.
    pub a: SparseMatrix<Gr1<L>>,
    pub b: SparseMatrix<Gr1<L>>,
    pub c: SparseMatrix<Gr1<L>>,
    /// Witness, length `n` (must be a multiple of 32).
    pub w: Vec<Gr1<L>>,
}

impl<L: Limb> GeneralR1csInstance<L> {
    /// True iff `(A·w) ∘ (B·w) == C·w`.
    pub fn is_satisfied(&self) -> bool {
        let aw = matvec_gr1(&self.a, &self.w);
        let bw = matvec_gr1(&self.b, &self.w);
        let cw = matvec_gr1(&self.c, &self.w);
        for i in 0..aw.len() {
            if aw[i] * bw[i] != cw[i] {
                return false;
            }
        }
        true
    }
}

/// General-circuit packed R1CS instance.
///
/// Produced by `general_to_packed`, length `n_blocks = n / 32`.
///
/// The Halevi–Shoup decomposition stores each input matrix as 32 cyclic
/// diagonal layers indexed by `ℓ ∈ [32]`, each itself a sparse
/// `n_blocks × n_blocks` matrix over `R_2` whose `(p, q)` entry is the
/// `pack_r2` of the `ℓ`-th cyclic diagonal of the `(p, q)` block of the
/// original `R_0`-valued matrix.
#[derive(Clone, Debug)]
pub struct PackedGeneralR1csInstance<L: Limb = u32> {
    /// 32 cyclic-diagonal layers of `A`, each `(n_blocks × n_blocks)` packed.
    pub d_hat_a: [SparseMatrix<Gr128<L>>; 32],
    pub d_hat_b: [SparseMatrix<Gr128<L>>; 32],
    pub d_hat_c: [SparseMatrix<Gr128<L>>; 32],
    /// Block-packed witness: `w_phi[q] = pack_r2(w[32q .. 32(q+1)])`.
    pub w_phi: Vec<Gr128<L>>,
    /// Packed rotations: `rho[ℓ][q] = pack_r2(rot_ℓ(w[32q .. 32(q+1)]))`.
    pub rho: [Vec<Gr128<L>>; 32],
    /// Linear-relation kernel witnesses.
    pub q1_a: Vec<Gr128<L>>,
    pub q2_a: Vec<Gr128<L>>,
    pub q1_b: Vec<Gr128<L>>,
    pub q2_b: Vec<Gr128<L>>,
    pub q1_c: Vec<Gr128<L>>,
    pub q2_c: Vec<Gr128<L>>,
    /// Multiplication-relation kernel witness.
    pub q1_mult: Vec<Gr128<L>>,
    pub q2_mult: Vec<Gr128<L>>,
}

impl<L: HasPackingConstants> PackedGeneralR1csInstance<L> {
    /// Two-part check:
    ///
    /// 1. Rotation relation: every `rho[ℓ][q]` unpacks to `rot_ℓ` of the
    ///    unpacked `w_phi[q]`.
    /// 2. Combined linear + multiplication identity:
    ///    `(Σ_ℓ D̂_a^{(ℓ)}·rho[ℓ] - ê^A) ∘ (Σ_ℓ D̂_b^{(ℓ)}·rho[ℓ] - ê^B)
    ///       == (Σ_ℓ D̂_c^{(ℓ)}·rho[ℓ] - ê^C) + ê^mult`.
    ///
    /// Uses `unpack_r2` (test-only path); `is_satisfied` is itself a sanity check.
    pub fn is_satisfied(&self) -> bool {
        // (1) Rotation relation.
        for ell in 0..32 {
            for q in 0..self.w_phi.len() {
                let w_block: [Gr1<L>; 32] = unpack_flat(self.w_phi[q]);
                let mut rotated = [Gr1::<L>::ZERO; 32];
                for i in 0..32 {
                    rotated[i] = w_block[(i + ell) % 32];
                }
                let rho_block: [Gr1<L>; 32] = unpack_flat(self.rho[ell][q]);
                if rotated != rho_block {
                    return false;
                }
                // Cross-check that rho[ℓ][q] equals pack_r2(rotated). This is
                // strictly stronger than the rotation relation R^rot_S
                // (it forces rho ∈ Im(φ)) and matches what general_to_packed
                // produces by construction.
                if pack_r2(rotated) != self.rho[ell][q] {
                    return false;
                }
            }
        }

        // (2) Linear + mult.
        let cst = constants::<L>();
        let v_alpha = Gr128::<L>::from_gr4(cst.v_alpha);
        let v_beta = cst.v_beta;
        let n_blocks = self.w_phi.len();
        let mut sum_a = vec![Gr128::<L>::ZERO; n_blocks];
        let mut sum_b = vec![Gr128::<L>::ZERO; n_blocks];
        let mut sum_c = vec![Gr128::<L>::ZERO; n_blocks];
        for ell in 0..32 {
            let part_a = matvec_gr128_dense(&self.d_hat_a[ell], &self.rho[ell]);
            let part_b = matvec_gr128_dense(&self.d_hat_b[ell], &self.rho[ell]);
            let part_c = matvec_gr128_dense(&self.d_hat_c[ell], &self.rho[ell]);
            for p in 0..n_blocks {
                sum_a[p] = sum_a[p] + part_a[p];
                sum_b[p] = sum_b[p] + part_b[p];
                sum_c[p] = sum_c[p] + part_c[p];
            }
        }
        for p in 0..n_blocks {
            let a_tilde = sum_a[p] - (self.q1_a[p] * v_alpha + self.q2_a[p] * v_beta);
            let b_tilde = sum_b[p] - (self.q1_b[p] * v_alpha + self.q2_b[p] * v_beta);
            let c_tilde = sum_c[p] - (self.q1_c[p] * v_alpha + self.q2_c[p] * v_beta);
            let lhs = a_tilde * b_tilde;
            let rhs = c_tilde + self.q1_mult[p] * v_alpha + self.q2_mult[p] * v_beta;
            if lhs != rhs {
                return false;
            }
        }
        true
    }
}

/// Helper: unpack a `pack_r2`'d `Im(φ)` element back to its 32-vector pre-image.
/// Wraps `unpack_r2` (which returns the 16×2 matrix view) to a flat 32-array.
fn unpack_flat<L: HasPackingConstants>(x: Gr128<L>) -> [Gr1<L>; 32] {
    let m = unpack_r2(x);
    let mut out = [Gr1::<L>::ZERO; 32];
    for i in 0..16 {
        out[2 * i] = m[i][0];
        out[2 * i + 1] = m[i][1];
    }
    out
}
