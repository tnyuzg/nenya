//! End-to-end correctness for the §3.2.2 MIMD reduction.
//!
//! Builds 32 R1CS instances with *distinct* `(A_i, B_i, C_i)` matrices
//! (so the MIMD path is genuinely exercised, not a degenerate SIMD), runs
//! `mimd_to_packed`, and checks the resulting packed instance satisfies
//! its consistency check. Plus a negative test that corrupting one
//! witness coordinate breaks satisfaction.
//!
//! Generic over the base-limb width `L` (`u8`/`u16`/`u32`/`u64`) via the
//! `width_tests!` macro at the bottom; the historical u32-only behavior is
//! preserved as the `w32` module.

use galois_ring::{Gr1, Limb};
use reduction::constants::HasPackingConstants;
use reduction::instance::MimdR1csInstance;
use reduction::matrix::SparseMatrix;
use reduction::reduce::mimd_to_packed;

/// Each instance `k` enforces `n` constraints of the form
///   `(a_{k,i} · x_i) · (b_{k,i} · 1) == y_i`
/// with witness layout `[1, x_0, y_0, x_1, y_1, …, x_{n-1}, y_{n-1}]`,
/// width `m = 1 + 2n`. The per-instance row scalars `(a_{k,i}, b_{k,i})`
/// are derived from `(k, i)` — so `A_k`, `B_k` differ across instances
/// (column-0 entries in `B_k`; column `1+2i` entries in `A_k`).
///
/// Inputs and derived scalars are injected via `L::from_u64`, which reduces
/// mod `2^L::BITS`; `y = a·b·x` is computed *in the ring* so the instance is
/// satisfied at every width.
fn build_squaring_mimd<L: Limb>(per_instance_xs: &[Vec<u64>]) -> MimdR1csInstance<L> {
    assert_eq!(per_instance_xs.len(), 32, "must supply 32 instances");
    let n = per_instance_xs[0].len();
    for xs in per_instance_xs {
        assert_eq!(xs.len(), n);
    }
    let m = 1 + 2 * n;

    let scalar_a = |k: u64, i: u64| -> Gr1<L> {
        Gr1(L::from_u64(
            (k.wrapping_mul(0x9E37_79B9) ^ i.wrapping_mul(0xDEAD_BEEF)) | 1,
        ))
    };
    let scalar_b = |k: u64, i: u64| -> Gr1<L> {
        Gr1(L::from_u64(
            (k.wrapping_mul(0xBB67_AE85) ^ i.wrapping_mul(0xCAFE_BABE)) | 1,
        ))
    };

    let mut a_per_instance: [SparseMatrix<Gr1<L>>; 32] =
        std::array::from_fn(|_| SparseMatrix::<Gr1<L>>::new(n, m));
    let mut b_per_instance: [SparseMatrix<Gr1<L>>; 32] =
        std::array::from_fn(|_| SparseMatrix::<Gr1<L>>::new(n, m));
    let mut c_per_instance: [SparseMatrix<Gr1<L>>; 32] =
        std::array::from_fn(|_| SparseMatrix::<Gr1<L>>::new(n, m));
    for k in 0..32 {
        for i in 0..n {
            let x_col = 1 + 2 * i;
            let y_col = 1 + 2 * i + 1;
            a_per_instance[k].push(i, x_col, scalar_a(k as u64, i as u64));
            b_per_instance[k].push(i, 0, scalar_b(k as u64, i as u64));
            c_per_instance[k].push(i, y_col, Gr1::<L>::ONE);
        }
    }

    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    for k in 0..32 {
        let mut w = vec![Gr1::<L>::ZERO; m];
        w[0] = Gr1::<L>::ONE;
        for i in 0..n {
            let x = Gr1(L::from_u64(per_instance_xs[k][i]));
            // y = (a_{k,i} · x) · (b_{k,i} · 1) = a_{k,i} · b_{k,i} · x.
            let y = scalar_a(k as u64, i as u64) * scalar_b(k as u64, i as u64) * x;
            w[1 + 2 * i] = x;
            w[1 + 2 * i + 1] = y;
        }
        witnesses[k] = w;
    }

    MimdR1csInstance {
        a_per_instance,
        b_per_instance,
        c_per_instance,
        witnesses,
    }
}

fn check_mimd_end_to_end_satisfied<L: HasPackingConstants>() {
    let n = 4;
    let mut per_instance: Vec<Vec<u64>> = Vec::with_capacity(32);
    for k in 0..32u64 {
        let mut xs = Vec::with_capacity(n);
        for i in 0..n as u64 {
            xs.push((k * 17 + i * 31 + 1).wrapping_mul(0xdead_b00f));
        }
        per_instance.push(xs);
    }
    let mimd = build_squaring_mimd::<L>(&per_instance);
    assert!(mimd.is_satisfied(), "the MIMD instance should be satisfied");

    let packed = mimd_to_packed(&mimd);
    assert!(
        packed.is_satisfied(),
        "packed MIMD instance must satisfy its consistency check"
    );
}

fn check_mimd_end_to_end_larger<L: HasPackingConstants>() {
    let n = 32;
    let mut per_instance: Vec<Vec<u64>> = Vec::with_capacity(32);
    for k in 0..32u64 {
        let mut xs = Vec::with_capacity(n);
        for i in 0..n as u64 {
            xs.push((k * 7919 + i * 6151).wrapping_add(1));
        }
        per_instance.push(xs);
    }
    let mimd = build_squaring_mimd::<L>(&per_instance);
    assert!(mimd.is_satisfied());
    let packed = mimd_to_packed(&mimd);
    assert!(packed.is_satisfied());
}

fn check_mimd_negative_corrupted_witness<L: HasPackingConstants>() {
    let per_instance: Vec<Vec<u64>> = (0..32u64)
        .map(|k| vec![k.wrapping_add(1), k.wrapping_mul(3).wrapping_add(7)])
        .collect();
    let mut mimd = build_squaring_mimd::<L>(&per_instance);
    assert!(mimd.is_satisfied());

    // Corrupt one entry: change instance 5's y_0 to a wrong value.
    mimd.witnesses[5][2] = Gr1(L::from_u64(0xbad_c0de));
    assert!(!mimd.is_satisfied(), "corrupted MIMD should NOT satisfy");

    let packed = mimd_to_packed(&mimd);
    assert!(
        !packed.is_satisfied(),
        "corrupted packed MIMD should NOT satisfy"
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
                check_mimd_end_to_end_satisfied::<$t>();
            }
            #[test]
            fn mimd_end_to_end_larger() {
                check_mimd_end_to_end_larger::<$t>();
            }
            #[test]
            fn mimd_negative_corrupted_witness() {
                check_mimd_negative_corrupted_witness::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
