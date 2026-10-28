//! Field-side Brakedown tensor PCS over BabyBear (commit base `BB`, evaluate at
//! a `BB4` point). The apples-to-apples baseline for [`pcs_ring::Pcs`]:
//! identical RAA code, blake3 column-Merkle, and Fiat–Shamir; the only
//! deliberate difference is the element type.
//! Used by `snark-field`.

pub mod pcs;
pub mod raa;

pub use pcs::{FieldCommitment, FieldOpening, FieldPcs};
pub use pcs_ring::merkle::Hash;
pub use pcs_ring::{SECURE_DELTA, SECURE_LAMBDA, SECURE_Q};
pub use raa::FieldRaa;

use field::{AbstractExtensionField, BB, BB4};

/// Lift a base element into the extension (`BB -> BB4`).
#[inline]
pub(crate) fn lift(b: BB) -> BB4 {
    <BB4 as AbstractExtensionField<BB>>::from_base(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::AbstractField;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const P: u32 = 0x7800_0001;
    fn rand_bb(rng: &mut StdRng) -> BB {
        BB::from_canonical_u32(rng.gen::<u32>() % P)
    }
    fn rand_bb4(rng: &mut StdRng) -> BB4 {
        let b: [BB; 4] = [rand_bb(rng), rand_bb(rng), rand_bb(rng), rand_bb(rng)];
        <BB4 as AbstractExtensionField<BB>>::from_base_slice(&b)
    }

    /// Reference MLE evaluation: ⟨eq(z), lift(evals)⟩.
    fn eval_mle_ref(evals: &[BB], z: &[BB4]) -> BB4 {
        let eq = pcs::eq_table(z);
        assert_eq!(eq.len(), evals.len());
        let mut acc = BB4::zero();
        for i in 0..evals.len() {
            acc += eq[i] * lift(evals[i]);
        }
        acc
    }

    #[test]
    fn open_verify_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xABCD_1234);
        let pcs = FieldPcs::new(8, 8, 4, 16, 0xDEAD_BEEF);
        let evals: Vec<BB> = (0..64).map(|_| rand_bb(&mut rng)).collect();
        let z: Vec<BB4> = (0..pcs.num_vars()).map(|_| rand_bb4(&mut rng)).collect();

        let comm = pcs.commit(&evals);
        let (y, proof) = pcs.open(&comm, &z);
        assert_eq!(
            y,
            eval_mle_ref(&evals, &z),
            "open returned wrong evaluation"
        );
        assert!(
            pcs.verify(comm.root, &z, y, &proof),
            "verify rejected valid proof"
        );
        assert!(
            !pcs.verify(comm.root, &z, y + BB4::one(), &proof),
            "verify accepted tampered evaluation"
        );
    }
}
