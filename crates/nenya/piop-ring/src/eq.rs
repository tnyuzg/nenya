//! `eq(r, h)` multilinear-extension table builder.
//!
//! For a fixed point `r ∈ R₂^n`, the indicator MLE
//! ```text
//! eq(r, h) = ∏_{i=1..n} ( r_i · h_i + (1 − r_i)(1 − h_i) )
//! ```
//! is a multilinear polynomial in `h` taking value `1` when `h = r` (on the
//! Boolean cube) and `0` at all other Boolean points. Off-cube it interpolates
//! multilinearly.
//!
//! [`build_eq_table`] returns the length-`2^n` evaluation table of `eq(r, ·)`
//! over the Boolean hypercube, indexed in the same convention as
//! [`crate::mle::Mle`]: variable `r[0]` corresponds to the **most-significant**
//! bit of the table index, so `evals[h₁·2^{n−1} + … + h_n·2^0] =
//! eq(r, (h₁, …, h_n))`.
//!
//! ## Cost
//!
//! Standard recursive doubling: at level `i` we extend the table from length
//! `2^i` to `2^{i+1}` by `eq[2k+1] := eq[k] · r_{i+1}` and `eq[2k] := eq[k] −
//! eq[2k+1]`. Total: `Σ_{i=0..n-1} 2^i ≈ 2^n − 1` GR32_128 muls. At `n = 15`,
//! ~32K muls ≈ 30 ms.

use galois_ring::{Gr128, Limb};
use rayon::prelude::*;

use crate::mle::Mle;

/// Build the evaluation table of `eq(r, ·)` over `{0, 1}^n` where `n =
/// r.len()`. The result is an [`Mle`] of length `2^n`.
///
/// Variable order: `r[0]` becomes the **most-significant** bit of the table
/// index — i.e. it corresponds to the *first* variable folded by
/// [`Mle::fold_in_place`].
pub fn build_eq_table<L: Limb>(r: &[Gr128<L>]) -> Mle<L> {
    let n = r.len();
    assert!(n > 0, "build_eq_table: need at least one variable");

    // Recursive doubling: at each level, output pair `(table[2k], table[2k+1])`
    // depends only on the old `table[k]`, so the level parallelizes by disjoint
    // write. We double into a fresh buffer per level (instead of the in-place
    // reverse scan) to expose that independence to rayon; the values — and the
    // r[0]=MSB index convention — are identical to the serial build.
    let mut table: Vec<Gr128<L>> = vec![Gr128::<L>::ONE; 1];
    for &r_i in r {
        let mut next = vec![Gr128::<L>::ZERO; table.len() * 2];
        next.par_chunks_mut(2)
            .zip(table.par_iter())
            .for_each(|(out, &prev)| {
                let hi = prev * r_i; // h_{new} = 1 → multiply by r_i
                out[0] = prev - hi; // h_{new} = 0 → multiply by (1 − r_i)
                out[1] = hi;
            });
        table = next;
    }

    Mle::new(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{GR32_128, GR32_4};

    fn r2_from_u32s(words: &[u32]) -> GR32_128 {
        let mut arr = [GR32_4::ZERO; 32];
        for (i, w) in words.iter().enumerate() {
            arr[i] = GR32_4([*w, 0, 0, 0]);
        }
        GR32_128(arr)
    }

    #[test]
    fn eq_zero_target_is_indicator_of_zero() {
        // r = (0, 0, 0): eq(0, h) = 1 iff h = 0.
        let r = vec![GR32_128::ZERO; 3];
        let table = build_eq_table(&r);
        assert_eq!(table.evals()[0], GR32_128::ONE);
        for i in 1..8 {
            assert_eq!(table.evals()[i], GR32_128::ZERO, "i = {i}");
        }
    }

    #[test]
    fn eq_one_target_is_indicator_of_all_ones() {
        // r = (1, 1, 1): eq(1, h) = 1 iff h = (1,1,1) = index 7.
        let r = vec![GR32_128::ONE; 3];
        let table = build_eq_table(&r);
        for i in 0..7 {
            assert_eq!(table.evals()[i], GR32_128::ZERO, "i = {i}");
        }
        assert_eq!(table.evals()[7], GR32_128::ONE);
    }

    #[test]
    fn eq_table_sums_to_one() {
        // For any r, ∑_h eq(r, h) over the Boolean cube equals 1.
        // (Standard partition-of-unity property of the eq MLE.)
        let r = vec![
            r2_from_u32s(&[3, 5]),
            r2_from_u32s(&[7, 11]),
            r2_from_u32s(&[13, 17]),
        ];
        let table = build_eq_table(&r);
        let sum = table.evals().iter().fold(GR32_128::ZERO, |acc, &x| acc + x);
        assert_eq!(sum, GR32_128::ONE);
    }

    #[test]
    fn eq_table_off_cube_two_var() {
        // For r = (a, b) with a, b arbitrary, the eq table is:
        //   eq(r, (h_1=0, h_2=0)) = (1-a)(1-b)   index = 0·2 + 0 = 0
        //   eq(r, (h_1=0, h_2=1)) = (1-a)·b      index = 0·2 + 1 = 1
        //   eq(r, (h_1=1, h_2=0)) = a·(1-b)      index = 1·2 + 0 = 2
        //   eq(r, (h_1=1, h_2=1)) = a·b          index = 1·2 + 1 = 3
        // (h_1 = MSB = r[0] = a; h_2 = LSB = r[1] = b)
        let a = r2_from_u32s(&[3, 5, 7]);
        let b = r2_from_u32s(&[11, 13, 17]);
        let r = vec![a, b];
        let table = build_eq_table(&r);
        let one = GR32_128::ONE;
        let one_a = one - a;
        let one_b = one - b;

        assert_eq!(table.evals()[0], one_a * one_b, "h = (0,0)");
        assert_eq!(table.evals()[1], one_a * b, "h = (0,1)");
        assert_eq!(table.evals()[2], a * one_b, "h = (1,0)");
        assert_eq!(table.evals()[3], a * b, "h = (1,1)");
    }
}
