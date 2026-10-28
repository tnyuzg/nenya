//! End-to-end correctness for the §3.2.3 general-circuit reduction.
//!
//! Builds a single satisfied R1CS over `R_0` with witness length divisible
//! by 32 and square `(n × n)` matrices, runs `general_to_packed`, and
//! verifies that the resulting `PackedGeneralR1csInstance` satisfies its
//! consistency check (which combines the rotation relation and the
//! linear+mult identity).
//!
//! Generic over the base-limb width `L` (`u8`/`u16`/`u32`/`u64`) via the
//! `width_tests!` macro at the bottom; the historical u32-only behavior is
//! preserved as the `w32` module.

use galois_ring::{Gr1, Limb};
use reduction::constants::HasPackingConstants;
use reduction::instance::GeneralR1csInstance;
use reduction::matrix::SparseMatrix;
use reduction::reduce::general_to_packed;

/// Build a general R1CS of size `n × n` with `n` divisible by 32. The
/// first `n_constraints` rows enforce `(a_i · w[2i+1]) · (b_i · w[0]) ==
/// w[2i+2]` (so column 0 holds the constant 1). All later rows are
/// trivially `0 == 0`. Witness layout `[1, x_0, y_0, x_1, y_1, …]` with
/// padding to length `n`.
///
/// Inputs and derived scalars are injected via `L::from_u64`, which reduces
/// mod `2^L::BITS`; `y = a·b·x` is computed *in the ring* so the instance is
/// satisfied at every width.
fn build_general<L: Limb>(n: usize, n_constraints: usize, xs: &[u64]) -> GeneralR1csInstance<L> {
    assert_eq!(n % 32, 0, "n must be divisible by 32");
    assert!(1 + 2 * n_constraints <= n, "witness must fit in length n");
    assert_eq!(xs.len(), n_constraints);

    let scalar_a = |i: u64| -> Gr1<L> { Gr1(L::from_u64(i.wrapping_mul(0x9E37_79B9) | 1)) };
    let scalar_b = |i: u64| -> Gr1<L> { Gr1(L::from_u64(i.wrapping_mul(0xBB67_AE85) | 1)) };

    let mut a = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, n);
    for i in 0..n_constraints {
        let x_col = 1 + 2 * i;
        let y_col = 1 + 2 * i + 1;
        a.push(i, x_col, scalar_a(i as u64));
        b.push(i, 0, scalar_b(i as u64));
        c.push(i, y_col, Gr1::<L>::ONE);
    }

    let mut w = vec![Gr1::<L>::ZERO; n];
    w[0] = Gr1::<L>::ONE;
    for i in 0..n_constraints {
        let x = Gr1(L::from_u64(xs[i]));
        let y = scalar_a(i as u64) * scalar_b(i as u64) * x;
        w[1 + 2 * i] = x;
        w[1 + 2 * i + 1] = y;
    }
    GeneralR1csInstance { a, b, c, w }
}

fn check_general_end_to_end_n32<L: HasPackingConstants>() {
    // n = 32: a single block (n_blocks = 1), exercises the small case.
    let n = 32;
    let n_constraints = 4;
    let xs: Vec<u64> = (0..n_constraints as u64)
        .map(|i| (i * 31 + 1).wrapping_mul(0xdead_b00f))
        .collect();
    let g = build_general::<L>(n, n_constraints, &xs);
    assert!(g.is_satisfied(), "general R1CS should be satisfied");

    let packed = general_to_packed(&g);
    assert!(
        packed.is_satisfied(),
        "packed general instance must satisfy its consistency check"
    );
}

fn check_general_end_to_end_n64<L: HasPackingConstants>() {
    // n = 64: 2 blocks per axis, exercises multi-block cyclic-diagonal layers.
    let n = 64;
    let n_constraints = 20;
    let xs: Vec<u64> = (0..n_constraints as u64)
        .map(|i| (i * 7919 + 1).wrapping_mul(0xcafe_babe))
        .collect();
    let g = build_general::<L>(n, n_constraints, &xs);
    assert!(g.is_satisfied());
    let packed = general_to_packed(&g);
    assert!(packed.is_satisfied());
}

fn check_general_negative_corrupted_witness<L: HasPackingConstants>() {
    let n = 32;
    let n_constraints = 2;
    let xs = vec![123u64, 456u64];
    let mut g = build_general::<L>(n, n_constraints, &xs);
    assert!(g.is_satisfied());

    // Corrupt: flip the y_0 slot to a wrong value.
    g.w[2] = Gr1(L::from_u64(0xbad_c0de));
    assert!(!g.is_satisfied(), "corrupted general should NOT satisfy");

    let packed = general_to_packed(&g);
    assert!(
        !packed.is_satisfied(),
        "corrupted packed general should NOT satisfy"
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
            fn general_end_to_end_n32() {
                check_general_end_to_end_n32::<$t>();
            }
            #[test]
            fn general_end_to_end_n64() {
                check_general_end_to_end_n64::<$t>();
            }
            #[test]
            fn general_negative_corrupted_witness() {
                check_general_negative_corrupted_witness::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
