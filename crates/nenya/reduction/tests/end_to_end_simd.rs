//! End-to-end correctness for the §3.2.1 SIMD reduction.
//!
//! Builds a satisfied 32-batch R1CS instance, runs `simd_to_packed`, and
//! verifies the resulting `PackedSimdR1csInstance` satisfies its own
//! consistency check. Also a negative test that flipping a single witness
//! entry breaks satisfaction.
//!
//! Layered order: assumes `math_sanity.rs` and `implementation.rs` have
//! already passed. A failure here points at the reduction / instance /
//! matvec layer.
//!
//! Generic over the base-limb width `L` (`u8`/`u16`/`u32`/`u64`) via the
//! `width_tests!` macro at the bottom; the historical u32-only behavior is
//! preserved as the `w32` module.

use galois_ring::{Gr1, Limb};
use reduction::constants::HasPackingConstants;
use reduction::instance::SimdR1csInstance;
use reduction::matrix::SparseMatrix;
use reduction::reduce::simd_to_packed;

// Build a tiny but real R1CS: x * x = y, with witness layout [1, x, y, slack].
// This gives us a single constraint of the form A·w ∘ B·w == C·w with
//   A = (0, 1, 0, 0)
//   B = (0, 1, 0, 0)
//   C = (0, 0, 1, 0)
// for the row "x · x = y".
//
// To exercise multiple constraints we replicate this for each entry of `vals`,
// generating one constraint per (x, y=x²) pair, with a witness of length
// `1 + 2·n` (the leading 1, then n pairs of (x_i, y_i)).
//
// Inputs are passed as `u64` and injected via `L::from_u64`, which reduces
// mod `2^L::BITS`; the squaring `y = x·x` stays consistent under that
// reduction, so the instance is satisfied at every width.
fn build_squaring_batch<L: Limb>(per_instance_xs: &[Vec<u64>]) -> SimdR1csInstance<L> {
    assert_eq!(per_instance_xs.len(), 32, "must supply 32 instances");
    let n = per_instance_xs[0].len(); // number of constraints per instance
    for xs in per_instance_xs {
        assert_eq!(xs.len(), n);
    }
    let m = 1 + 2 * n; // witness length

    let mut a = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, m);
    for i in 0..n {
        let x_col = 1 + 2 * i;
        let y_col = 1 + 2 * i + 1;
        a.push(i, x_col, Gr1::<L>::ONE);
        b.push(i, x_col, Gr1::<L>::ONE);
        c.push(i, y_col, Gr1::<L>::ONE);
    }

    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    for k in 0..32 {
        let mut w = vec![Gr1::<L>::ZERO; m];
        w[0] = Gr1::<L>::ONE;
        for i in 0..n {
            let x = Gr1(L::from_u64(per_instance_xs[k][i]));
            w[1 + 2 * i] = x;
            w[1 + 2 * i + 1] = x * x;
        }
        witnesses[k] = w;
    }

    SimdR1csInstance { a, b, c, witnesses }
}

/// End-to-end happy path: build a satisfied batch, reduce, packed instance is satisfied.
fn check_end_to_end_satisfied<L: HasPackingConstants>() {
    // 32 instances, 4 constraints each, with arbitrary x values.
    let mut per_instance: Vec<Vec<u64>> = Vec::with_capacity(32);
    for k in 0..32u64 {
        let mut xs = Vec::with_capacity(4);
        for i in 0..4u64 {
            // some arbitrary spread of x values
            xs.push((k * 17 + i * 31 + 1).wrapping_mul(0xdead_b00f));
        }
        per_instance.push(xs);
    }
    let batch = build_squaring_batch::<L>(&per_instance);
    assert!(
        batch.is_satisfied(),
        "the batch we constructed should be satisfied"
    );

    let packed = simd_to_packed(&batch);
    assert!(
        packed.is_satisfied(),
        "packed instance must satisfy its consistency check"
    );
}

/// Larger end-to-end with 64 constraints.
fn check_end_to_end_larger<L: HasPackingConstants>() {
    let n = 64;
    let mut per_instance: Vec<Vec<u64>> = Vec::with_capacity(32);
    for k in 0..32u64 {
        let mut xs = Vec::with_capacity(n);
        for i in 0..n as u64 {
            xs.push((k * 7919 + i * 6151).wrapping_add(1));
        }
        per_instance.push(xs);
    }
    let batch = build_squaring_batch::<L>(&per_instance);
    assert!(batch.is_satisfied());
    let packed = simd_to_packed(&batch);
    assert!(packed.is_satisfied());
}

/// Negative test — corrupt one witness entry, batch.is_satisfied()
/// returns false, AND simd_to_packed produces an UNSATISFIED packed
/// instance.
fn check_negative_corrupted_witness<L: HasPackingConstants>() {
    let per_instance: Vec<Vec<u64>> = (0..32u64)
        .map(|k| vec![k.wrapping_add(1), k.wrapping_mul(3).wrapping_add(7)])
        .collect();
    let mut batch = build_squaring_batch::<L>(&per_instance);
    assert!(batch.is_satisfied());

    // Corrupt one entry: change instance 5's y_0 to a wrong value.
    batch.witnesses[5][2] = Gr1(L::from_u64(0xbad_c0de));
    assert!(!batch.is_satisfied(), "corrupted batch should NOT satisfy");

    let packed = simd_to_packed(&batch);
    assert!(
        !packed.is_satisfied(),
        "corrupted packed instance should NOT satisfy"
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
            fn end_to_end_satisfied() {
                check_end_to_end_satisfied::<$t>();
            }
            #[test]
            fn end_to_end_larger() {
                check_end_to_end_larger::<$t>();
            }
            #[test]
            fn negative_corrupted_witness() {
                check_negative_corrupted_witness::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
