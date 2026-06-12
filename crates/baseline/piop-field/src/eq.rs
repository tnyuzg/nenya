//! `eq(r, h)` MLE table over BB4. See [`piop_ring::eq`] for the full
//! derivation; the algebra is identical, only the element type differs.

use field::{AbstractField, BB4};
use rayon::prelude::*;

use crate::mle::Mle;

/// Build the eval table of `eq(r, ·)` over `{0,1}^n`. `r[0]` ends up at the
/// most-significant index bit (matching [`Mle::fold_in_place`]'s variable
/// order).
pub fn build_eq_table(r: &[BB4]) -> Mle {
    let n = r.len();
    assert!(n > 0, "build_eq_table: need at least one variable");

    // Recursive doubling into a fresh buffer per level so the independent
    // output pairs `(table[2k], table[2k+1])` parallelize by disjoint write.
    // Values and the r[0]=MSB index convention match the serial build.
    let mut table: Vec<BB4> = vec![BB4::one(); 1];
    for &r_i in r {
        let mut next = vec![BB4::zero(); table.len() * 2];
        next.par_chunks_mut(2)
            .zip(table.par_iter())
            .for_each(|(out, &prev)| {
                let hi = prev * r_i;
                out[0] = prev - hi;
                out[1] = hi;
            });
        table = next;
    }

    Mle::new(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_zero_target_is_indicator_of_zero() {
        let r = vec![BB4::zero(); 3];
        let table = build_eq_table(&r);
        assert_eq!(table.evals()[0], BB4::one());
        for i in 1..8 {
            assert_eq!(table.evals()[i], BB4::zero(), "i = {i}");
        }
    }

    #[test]
    fn eq_one_target_is_indicator_of_all_ones() {
        let r = vec![BB4::one(); 3];
        let table = build_eq_table(&r);
        for i in 0..7 {
            assert_eq!(table.evals()[i], BB4::zero(), "i = {i}");
        }
        assert_eq!(table.evals()[7], BB4::one());
    }

    #[test]
    fn eq_table_sums_to_one() {
        // Use `from_canonical_u32` to build an arbitrary BB4 element
        // from base-field constants.
        use field::{AbstractExtensionField, BB};
        let mk = |b0, b1, b2, b3| {
            <BB4 as AbstractExtensionField<BB>>::from_base_slice(&[
                BB::from_canonical_u32(b0),
                BB::from_canonical_u32(b1),
                BB::from_canonical_u32(b2),
                BB::from_canonical_u32(b3),
            ])
        };
        let r = vec![mk(3, 5, 7, 11), mk(13, 17, 19, 23), mk(29, 31, 37, 41)];
        let table = build_eq_table(&r);
        let sum = table.evals().iter().fold(BB4::zero(), |acc, &x| acc + x);
        assert_eq!(sum, BB4::one());
    }
}
