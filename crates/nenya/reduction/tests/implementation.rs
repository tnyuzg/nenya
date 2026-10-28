//! Implementation correctness — check that the runtime-computed packing
//! constants and the `embedding.rs` code realize the math from the paper
//! §2.3, **for every base-limb width** (`u8`, `u16`, `u32`, `u64`).
//!
//! Layered order: assumes `math_sanity.rs` has already passed (so the formulas
//! in the paper are correct). A failure here points at the constant
//! computation or the embedding implementation, not the spec.
//!
//! This is also the u8 algebraic-core milestone: the quotient identity
//! `φ(a)·φ(b) = φ(a∘b) + q1·v_α + q2·v_β` and the round-trip `ψ∘φ = id` are
//! exercised over `Z/2^8` here, and `PackingConstants::compute` validates the
//! freshly-computed u8 constants via its internal `debug_assert!`s on first use.

use galois_ring::{Gr1, Gr128, Gr4, Limb};
use reduction::constants::{constants, HasPackingConstants};
use reduction::embedding::{mul_with_quotients, pack_r1, pack_r2, pack_r2_matrix, unpack_r2};

// ---------------------------------------------------------------------------
// Generic checks, run once per width below.
// ---------------------------------------------------------------------------

fn alpha<L: Limb>() -> Gr4<L> {
    Gr4([L::ZERO, L::ONE, L::ZERO, L::ZERO])
}

/// INV_15, L_ALPHA, V_ALPHA, Q_ALPHA, V_BETA, L_BETA, Q_BETA identities.
fn check_constants<L: HasPackingConstants>() {
    let c = constants::<L>();

    // 15 · 15⁻¹ == 1 in Z/2^n.
    assert_eq!(Gr1(L::from_u64(15)) * c.inv_15, Gr1::<L>::ONE, "inv_15");

    // L_ALPHA = [1 - α, α].
    let a = alpha::<L>();
    assert_eq!(c.l_alpha[0], Gr4::<L>::ONE - a, "l_alpha[0]");
    assert_eq!(c.l_alpha[1], a, "l_alpha[1]");

    // V_ALPHA = α² - α.
    assert_eq!(c.v_alpha, a * a - a, "v_alpha");

    // Q_ALPHA: ℓ_j·ℓ_k == q_{j,k}·v_α + δ·ℓ_j.
    for j in 0..2 {
        for k in 0..2 {
            let lhs = c.l_alpha[j] * c.l_alpha[k];
            let rhs =
                c.q_alpha[j][k] * c.v_alpha + if j == k { c.l_alpha[j] } else { Gr4::<L>::ZERO };
            assert_eq!(lhs, rhs, "Q_ALPHA identity ({j},{k})");
        }
    }

    // V_BETA = β^16 - β.
    let mut arr = [Gr4::<L>::ZERO; 32];
    arr[1] = Gr4::<L>::ONE;
    let beta = Gr128(arr);
    assert_eq!(c.v_beta, beta.pow(16) - beta, "v_beta");

    // Σ ℓ_i(β) == 1.
    let sum = c
        .l_beta
        .iter()
        .copied()
        .fold(Gr128::<L>::ZERO, |x, y| x + y);
    assert_eq!(sum, Gr128::<L>::ONE, "l_beta partition of unity");

    // Q_BETA: ℓ_i·ℓ_j == q_{i,j}·v_β + δ·ℓ_i.
    for i in 0..16 {
        for j in 0..16 {
            let lhs = c.l_beta[i] * c.l_beta[j];
            let rhs = c.q_beta[i][j] * c.v_beta
                + if i == j {
                    c.l_beta[i]
                } else {
                    Gr128::<L>::ZERO
                };
            assert_eq!(lhs, rhs, "Q_BETA identity ({i},{j})");
        }
    }
}

/// φ_1 / φ_2 are linear and faithful; ψ∘φ = id.
fn check_packing<L: HasPackingConstants>() {
    let c = constants::<L>();

    // pack_r1 one-hot → L_ALPHA columns.
    assert_eq!(pack_r1([Gr1::<L>::ONE, Gr1::<L>::ZERO]), c.l_alpha[0]);
    assert_eq!(pack_r1([Gr1::<L>::ZERO, Gr1::<L>::ONE]), c.l_alpha[1]);

    // pack_r2 one-hot → ℓ_j(α)·ℓ_i(β).
    for i in 0..16 {
        for j in 0..2 {
            let mut m = [[Gr1::<L>::ZERO; 2]; 16];
            m[i][j] = Gr1::<L>::ONE;
            let got = pack_r2_matrix(m);
            let want = Gr128::from_gr4(c.l_alpha[j]) * c.l_beta[i];
            assert_eq!(got, want, "pack_r2 one-hot ({i},{j})");
        }
    }

    // ψ ∘ φ = id on Im(φ), a few patterns (values truncate to width via from_u64).
    let patterns: [[u64; 32]; 4] = [
        [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
        [
            0xdead_beef,
            0xcafe_babe,
            0x1234_5678,
            0x8765_4321,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [0xffff_ffff; 32],
    ];
    for pat in &patterns {
        let mut input = [Gr1::<L>::ZERO; 32];
        for k in 0..32 {
            input[k] = Gr1(L::from_u64(pat[k]));
        }
        let unpacked = unpack_r2(pack_r2(input));
        for i in 0..16 {
            for j in 0..2 {
                assert_eq!(
                    unpacked[i][j],
                    input[2 * i + j],
                    "round-trip at {}",
                    2 * i + j
                );
            }
        }
    }
}

/// The quotient identity `φ(a)·φ(b) = φ(a∘b) + q1·v_α + q2·v_β`.
fn check_mul_with_quotients<L: HasPackingConstants>() {
    let c = constants::<L>();
    let v_alpha_r2 = Gr128::from_gr4(c.v_alpha);

    let rand_matrix = |seed: u64| -> [[Gr1<L>; 2]; 16] {
        let mut x = seed | 1;
        let mut next = || -> u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut m = [[Gr1::<L>::ZERO; 2]; 16];
        for i in 0..16 {
            for j in 0..2 {
                m[i][j] = Gr1(L::from_u64(next()));
            }
        }
        m
    };

    for seed in 0..32u64 {
        let a = rand_matrix(seed * 2 + 1);
        let b = rand_matrix(seed * 2 + 2);
        let (c_phi, q1, q2) = mul_with_quotients(a, b);

        let lhs = pack_r2_matrix(a) * pack_r2_matrix(b);
        let rhs = c_phi + q1 * v_alpha_r2 + q2 * c.v_beta;
        assert_eq!(lhs, rhs, "quotient identity at seed={seed}");

        // c_phi == φ_2(a ∘ b).
        let mut hadamard = [[Gr1::<L>::ZERO; 2]; 16];
        for i in 0..16 {
            for j in 0..2 {
                hadamard[i][j] = a[i][j] * b[i][j];
            }
        }
        assert_eq!(c_phi, pack_r2_matrix(hadamard), "c_phi ≠ φ_2(a∘b)");
    }
}

// ---------------------------------------------------------------------------
// Per-width instantiations.
// ---------------------------------------------------------------------------

macro_rules! width_tests {
    ($modname:ident, $t:ty) => {
        mod $modname {
            use super::*;

            #[test]
            fn constants_identities() {
                check_constants::<$t>();
            }
            #[test]
            fn packing_faithful() {
                check_packing::<$t>();
            }
            #[test]
            fn quotient_identity() {
                check_mul_with_quotients::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
