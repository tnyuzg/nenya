//! End-to-end MIMD SNARK test (paper §4.2): 32 independent R1CS instances,
//! each with `x·x = y` constraints (distinct witnesses, here identical
//! matrices, but the prover treats them as independent per-instance matrices).
//! Instantiated per base-limb width below (u8/u16/u32/u64).

use galois_ring::{Gr1, Gr128, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, pack_mimd_matrices, MimdR1csInstance};
use snark_ring::{prove_mimd, verify_mimd, MimdSnarkParams, R2Pcs};

fn build_mimd<L: Limb>(log_size: usize) -> MimdR1csInstance<L> {
    let n = 1usize << log_size; // rows
    let m = n; // cols
    let k = (m - 1) / 2;
    let mut a_per: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    let mut b_per: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    let mut c_per: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    for inst in 0..32 {
        for i in 0..k {
            a_per[inst].push(i, 1 + 2 * i, Gr1::<L>::ONE);
            b_per[inst].push(i, 1 + 2 * i, Gr1::<L>::ONE);
            c_per[inst].push(i, 2 + 2 * i, Gr1::<L>::ONE);
        }
    }
    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![Gr1::<L>::ZERO; m]);
    for inst in 0..32 {
        witnesses[inst][0] = Gr1::<L>::ONE;
        for i in 0..k {
            let x = Gr1(L::from_u64(
                (((inst * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1),
            ));
            witnesses[inst][1 + 2 * i] = x;
            witnesses[inst][2 + 2 * i] = x * x;
        }
    }
    MimdR1csInstance {
        a_per_instance: a_per,
        b_per_instance: b_per,
        c_per_instance: c_per,
        witnesses,
    }
}

/// **INSECURE — unit-test params only** (`ell = 8`, ~4-bit, small for speed).
/// Real proofs use [`MimdSnarkParams::secure`] (δ=0.19, λ=100 ⇒ `ell = 1060`).
fn make_params<L: HasPackingConstants>(
    mimd: &MimdR1csInstance<L>,
    log_size: usize,
) -> MimdSnarkParams<L> {
    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    MimdSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w: Pcs::new(k1, k2, 4, 8, 0xDEAD_BEEF),
        pcs_abc: Pcs::new(k1, k2, 4, 8, 0xABC_BEEF),
        pcs_e: R2Pcs::new(k1, k2, 4, 8, 0xCAFE_BABE),
        mats: pack_mimd_matrices(
            &mimd.a_per_instance,
            &mimd.b_per_instance,
            &mimd.c_per_instance,
        ),
    }
}

fn check_satisfied<L: HasPackingConstants>() {
    let log_size = 6; // 64 × 64
    let mimd = build_mimd::<L>(log_size);
    assert!(mimd.is_satisfied(), "instance not satisfied");
    let params = make_params::<L>(&mimd, log_size);
    let proof = prove_mimd(&mimd.witnesses, &params);
    assert!(
        verify_mimd(&params, &proof).is_ok(),
        "verify rejected a valid MIMD proof"
    );
}

fn check_tampered_proof<L: HasPackingConstants>() {
    let log_size = 6;
    let mimd = build_mimd::<L>(log_size);
    let params = make_params::<L>(&mimd, log_size);
    let mut proof = prove_mimd(&mimd.witnesses, &params);
    proof.mult_proof.final_a = proof.mult_proof.final_a + Gr128::<L>::ONE;
    assert!(
        verify_mimd(&params, &proof).is_err(),
        "verify accepted a tampered proof"
    );
}

// The stacked (ê_A, ê_B) commitment is absorbed into the transcript *before* γ
// and bound by the PCS. Tampering its root must be caught — pins the new
// commit-before-γ ordering and that root_estack is transcript/PCS-bound.
fn check_tampered_estack_root<L: HasPackingConstants>() {
    let log_size = 6;
    let mimd = build_mimd::<L>(log_size);
    let params = make_params::<L>(&mimd, log_size);
    let mut proof = prove_mimd(&mimd.witnesses, &params);
    proof.root_estack[0] ^= 1;
    assert!(
        verify_mimd(&params, &proof).is_err(),
        "verify accepted a proof with a tampered stacked-error root"
    );
}

// Corrupting the stacked opening's m̄ must be caught: m̄ feeds both the
// Fiat–Shamir query-index derivation and the recomputed ê_A(s₁)+γ·ê_B(s₁).
fn check_tampered_estack_mbar<L: HasPackingConstants>() {
    let log_size = 6;
    let mimd = build_mimd::<L>(log_size);
    let params = make_params::<L>(&mimd, log_size);
    let mut proof = prove_mimd(&mimd.witnesses, &params);
    proof.open_estack.mbar[0] = proof.open_estack.mbar[0] + Gr128::<L>::ONE;
    assert!(
        verify_mimd(&params, &proof).is_err(),
        "verify accepted a proof with a tampered stacked-error m̄"
    );
}

// ---------------------------------------------------------------------------
// Per-width instantiations.
// ---------------------------------------------------------------------------

macro_rules! width_tests {
    ($modname:ident, $t:ty) => {
        mod $modname {
            use super::*;

            #[test]
            fn mimd_end_to_end_satisfied() {
                check_satisfied::<$t>();
            }
            #[test]
            fn mimd_negative_tampered_proof() {
                check_tampered_proof::<$t>();
            }
            #[test]
            fn mimd_negative_tampered_estack_root() {
                check_tampered_estack_root::<$t>();
            }
            #[test]
            fn mimd_negative_tampered_estack_mbar() {
                check_tampered_estack_mbar::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
