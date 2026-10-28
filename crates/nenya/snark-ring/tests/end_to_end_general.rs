//! End-to-end general-circuit SNARK test (paper §4.3, Halevi–Shoup).
//!
//! Builds a satisfied single R1CS over R_0 = Z/2^L with `x·x = y` constraints
//! (square `n×n`, witness length `n`, `32 | n`), runs `prove_general` /
//! `verify_general`, and checks acceptance + rejection of a tampered proof.
//! Instantiated per base-limb width below (u8/u16/u32/u64).

use galois_ring::{Gr1, Gr128, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, pack_general_matrices, GeneralR1csInstance};
use snark_ring::{prove_general, verify_general, GeneralSnarkParams, R2Pcs};

/// Satisfied general instance: `n×n` matrices, witness `[1, x_0, y_0, x_1, …]`
/// with `y_i = x_i²`. Rows `< k` carry an `x·x = y` constraint; the rest are
/// empty (`0·0 = 0`).
fn build_general<L: Limb>(n: usize) -> GeneralR1csInstance<L> {
    assert!(n % 32 == 0);
    let k = (n - 1) / 2;
    let mut a = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, n);
    for i in 0..k {
        a.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        b.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        c.push(i, 2 + 2 * i, Gr1::<L>::ONE);
    }
    let mut w = vec![Gr1::<L>::ZERO; n];
    w[0] = Gr1::<L>::ONE;
    for i in 0..k {
        let x = Gr1(L::from_u64((i as u64) ^ 0x9E37)
            .wrapping_mul(L::from_u64(2_654_435_761))
            .wrapping_add(L::ONE));
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    GeneralR1csInstance { a, b, c, w }
}

/// **INSECURE — unit-test params only** (`ell = 8`, ~4-bit, small for speed).
/// Real proofs use [`GeneralSnarkParams::secure`] (δ=0.19, λ=100 ⇒ `ell = 1060`).
fn make_params<L: HasPackingConstants>(
    g: &GeneralR1csInstance<L>,
    n: usize,
) -> GeneralSnarkParams<L> {
    let nb = n / 32; // block count = packed length
    let log_blocks = nb.trailing_zeros() as usize;
    let half = log_blocks / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_blocks - half);
    GeneralSnarkParams {
        log_blocks,
        pcs_w: Pcs::new(k1, k2, 4, 8, 0xDEAD_BEEF),
        pcs_abc: Pcs::new(k1, k2, 4, 8, 0xABC_BEEF),
        pcs_e: R2Pcs::new(k1, k2, 4, 8, 0xCAFE_BABE),
        diag: pack_general_matrices(&g.a, &g.b, &g.c),
    }
}

fn check_satisfied<L: HasPackingConstants>() {
    let n = 512; // nb = 16, log_blocks = 4
    let g = build_general::<L>(n);
    assert!(g.is_satisfied(), "instance not satisfied");
    let params = make_params::<L>(&g, n);
    let proof = prove_general(&g.w, &params);
    assert!(
        verify_general(&params, &proof).is_ok(),
        "verify rejected a valid general-circuit proof"
    );
}

fn check_tampered_proof<L: HasPackingConstants>() {
    let n = 512;
    let g = build_general::<L>(n);
    let params = make_params::<L>(&g, n);
    let mut proof = prove_general(&g.w, &params);
    // Corrupt the mult-check final value.
    proof.mult_proof.final_a = proof.mult_proof.final_a + Gr128::<L>::ONE;
    assert!(
        verify_general(&params, &proof).is_err(),
        "verify accepted a tampered proof"
    );
}

// Pins the new commit-before-γ ordering: the stacked (ê_A, ê_B) root is
// absorbed pre-γ and PCS-bound, so tampering it must be caught.
fn check_tampered_estack_root<L: HasPackingConstants>() {
    let n = 512;
    let g = build_general::<L>(n);
    let params = make_params::<L>(&g, n);
    let mut proof = prove_general(&g.w, &params);
    proof.root_estack[0] ^= 1;
    assert!(
        verify_general(&params, &proof).is_err(),
        "verify accepted a proof with a tampered stacked-error root"
    );
}

// Corrupting the stacked opening's m̄ must be caught (it feeds the FS query
// indices and the recomputed ê_A(s₁)+γ·ê_B(s₁)).
fn check_tampered_estack_mbar<L: HasPackingConstants>() {
    let n = 512;
    let g = build_general::<L>(n);
    let params = make_params::<L>(&g, n);
    let mut proof = prove_general(&g.w, &params);
    proof.open_estack.mbar[0] = proof.open_estack.mbar[0] + Gr128::<L>::ONE;
    assert!(
        verify_general(&params, &proof).is_err(),
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
            fn general_end_to_end_satisfied() {
                check_satisfied::<$t>();
            }
            #[test]
            fn general_negative_tampered_proof() {
                check_tampered_proof::<$t>();
            }
            #[test]
            fn general_negative_tampered_estack_root() {
                check_tampered_estack_root::<$t>();
            }
            #[test]
            fn general_negative_tampered_estack_mbar() {
                check_tampered_estack_mbar::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
