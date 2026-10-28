//! Brakedown-style tensor polynomial commitment scheme over
//! `R = GR(2^32, 128)` with an RAA fast-path linear code, committing
//! directly to the 32-u32 packed representation of `reduction::pack_r2`.
//!
//! The protocol is the paper's Protocol 5.1. This crate is a prototype
//! intended to validate the protocol end-to-end and produce concrete
//! commit/open/verify/proof-size numbers — it is not a hardened production
//! implementation.

pub mod merkle;
pub mod pcs;
pub mod raa;

pub use merkle::{Hash, MerkleTree};
pub use pcs::{
    achievable_security, optimal_shape, optimal_shape_for, proof_size_bytes, proof_size_bytes_for,
    recommended_query_size, Commitment, OpeningProof, Pcs, RotatedOpeningBatch,
    StackedSlotCommitment, SECURE_DELTA, SECURE_LAMBDA, SECURE_Q,
};
pub use raa::RaaParams;

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use galois_ring::{Gr1, Gr128, Limb};
    use piop_ring::build_eq_table;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use reduction::constants::HasPackingConstants;
    use reduction::pack_r2;

    fn random_packed_evals<L: Limb>(n: usize, seed: u64) -> Vec<[L; 32]> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut a = [L::ZERO; 32];
                for s in 0..32 {
                    a[s] = L::random(&mut rng);
                }
                a
            })
            .collect()
    }

    fn random_point<L: HasPackingConstants>(num_vars: usize, seed: u64) -> Vec<Gr128<L>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..num_vars)
            .map(|_| Gr128::from_gr1(Gr1(L::random(&mut rng))))
            .collect()
    }

    /// Reference: evaluate a multilinear whose Boolean-cube evals are
    /// `pack_r2(evals_packed[i])`, using the same MSB-first variable order
    /// as `build_eq_table`.
    fn eval_mle_packed<L: HasPackingConstants>(
        evals_packed: &[[L; 32]],
        z: &[Gr128<L>],
    ) -> Gr128<L> {
        let table = build_eq_table(z);
        let t = table.evals();
        assert_eq!(t.len(), evals_packed.len());
        let mut acc = Gr128::<L>::ZERO;
        for i in 0..evals_packed.len() {
            let mut e = [Gr1::<L>::ZERO; 32];
            for s in 0..32 {
                e[s] = Gr1(evals_packed[i][s]);
            }
            acc = acc + t[i] * pack_r2(e);
        }
        acc
    }

    fn check_end_to_end_small<L: HasPackingConstants>() {
        // N = 64 = k1 · k2, k1 = 8, k2 = 8, q = 4, ell = 16.
        let pcs = Pcs::<L>::new(8, 8, 4, 16, 0xDEAD_BEEF);

        let evals = random_packed_evals::<L>(8 * 8, 1);
        let comm = pcs.commit(&evals);
        let z = random_point::<L>(pcs.num_vars(), 2);

        let (y, proof) = pcs.open(&comm, &z);
        let y_ref = eval_mle_packed(&evals, &z);
        assert_eq!(y, y_ref, "open returned wrong evaluation");

        assert!(pcs.verify(comm.root, &z, y, &proof), "verify failed");

        // Tamper detection: a wrong evaluation must reject.
        let bad_y = y + Gr128::<L>::ONE;
        assert!(!pcs.verify(comm.root, &z, bad_y, &proof));
    }

    fn check_end_to_end_medium<L: HasPackingConstants>() {
        // N = 1024 = 32 · 32, q = 4, n = 128, ell = 32.
        let pcs = Pcs::<L>::new(32, 32, 4, 32, 0xCAFE);
        let evals = random_packed_evals::<L>(32 * 32, 5);
        let comm = pcs.commit(&evals);
        let z = random_point::<L>(pcs.num_vars(), 6);
        let (y, proof) = pcs.open(&comm, &z);
        let y_ref = eval_mle_packed(&evals, &z);
        assert_eq!(y, y_ref);
        assert!(pcs.verify(comm.root, &z, y, &proof));
    }

    fn check_tampered_column_rejects<L: HasPackingConstants>() {
        let pcs = Pcs::<L>::new(8, 8, 4, 8, 0x1234);
        let evals = random_packed_evals::<L>(8 * 8, 1);
        let comm = pcs.commit(&evals);
        let z = random_point::<L>(pcs.num_vars(), 2);
        let (y, mut proof) = pcs.open(&comm, &z);
        // Flip one limb of one queried column.
        proof.query_columns[0][0][0] = proof.query_columns[0][0][0].wrapping_add(L::ONE);
        assert!(!pcs.verify(comm.root, &z, y, &proof));
    }

    // -----------------------------------------------------------------------
    // Per-width instantiations. The full stack runs at every base-limb width;
    // u32 is the historical default, u64 is the SNARK's target width.
    // -----------------------------------------------------------------------

    macro_rules! width_tests {
        ($modname:ident, $t:ty) => {
            mod $modname {
                use super::*;

                #[test]
                fn end_to_end_small() {
                    check_end_to_end_small::<$t>();
                }
                #[test]
                fn end_to_end_medium() {
                    check_end_to_end_medium::<$t>();
                }
                #[test]
                fn tampered_column_rejects() {
                    check_tampered_column_rejects::<$t>();
                }
            }
        };
    }

    width_tests!(w8, u8);
    width_tests!(w16, u16);
    width_tests!(w32, u32);
    width_tests!(w64, u64);
}
