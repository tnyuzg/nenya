//! Math sanity — independently re-derive the Lagrange / quotient / vanishing
//! identities from the paper's Appendix B using only `GR32_*` and `T1`. Does NOT
//! depend on `crates/reduction/src/constants.rs`; the point is to catch
//! transcription errors in the markdown itself, not in the implementation.
//!
//! Layered order: run this first. `implementation.rs` and `end_to_end.rs`
//! assume these have passed.
//!
//! The β-side polynomials are represented as `Vec<GR32_4>` in the
//! "polynomial in t with R_1 coefficients" sense. We never lift to R_2 here:
//! every algebraic property we want to test can be checked at the polynomial
//! level over R_1.
//!
//! The α-side polynomials are degree ≤ 2 with R_0 coefficients, also as
//! `Vec<GR32_1>`.

use galois_ring::{GR32_1, GR32_4, T1};

// ---------------------------------------------------------------------------
// Polynomial helpers (R_1 coefficients)
// ---------------------------------------------------------------------------
type Poly = Vec<GR32_4>;

fn pn(mut p: Poly) -> Poly {
    while p.last() == Some(&GR32_4::ZERO) {
        p.pop();
    }
    p
}
fn padd(a: &[GR32_4], b: &[GR32_4]) -> Poly {
    let n = a.len().max(b.len());
    let mut r = vec![GR32_4::ZERO; n];
    for i in 0..a.len() {
        r[i] = r[i] + a[i];
    }
    for i in 0..b.len() {
        r[i] = r[i] + b[i];
    }
    pn(r)
}
fn psub(a: &[GR32_4], b: &[GR32_4]) -> Poly {
    let n = a.len().max(b.len());
    let mut r = vec![GR32_4::ZERO; n];
    for i in 0..a.len() {
        r[i] = r[i] + a[i];
    }
    for i in 0..b.len() {
        r[i] = r[i] - b[i];
    }
    pn(r)
}
fn pmul(a: &[GR32_4], b: &[GR32_4]) -> Poly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut r = vec![GR32_4::ZERO; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            r[i + j] = r[i + j] + a[i] * b[j];
        }
    }
    pn(r)
}
fn pscale(a: &[GR32_4], s: GR32_4) -> Poly {
    pn(a.iter().map(|&x| x * s).collect())
}
/// Polynomial division by a monic divisor; returns (quotient, remainder).
fn pdiv(a: &[GR32_4], b: &[GR32_4]) -> (Poly, Poly) {
    assert!(!b.is_empty());
    assert_eq!(
        *b.last().unwrap(),
        GR32_4::ONE,
        "pdiv requires monic divisor"
    );
    if a.len() < b.len() {
        return (vec![], a.to_vec());
    }
    let mut rem = a.to_vec();
    let mut q = vec![GR32_4::ZERO; a.len() - b.len() + 1];
    while rem.len() >= b.len() {
        let factor = *rem.last().unwrap();
        let q_idx = rem.len() - b.len();
        q[q_idx] = q[q_idx] + factor;
        for i in 0..b.len() {
            let idx = q_idx + i;
            rem[idx] = rem[idx] - factor * b[i];
        }
        rem = pn(rem);
    }
    (pn(q), rem)
}
fn peval(p: &[GR32_4], t: GR32_4) -> GR32_4 {
    let mut acc = GR32_4::ZERO;
    for &c in p.iter().rev() {
        acc = acc * t + c;
    }
    acc
}

/// `1 / 15 mod 2^32`, lifted to R_1.
fn inv_15() -> GR32_4 {
    let a: u32 = 15;
    let mut x: u32 = 1;
    for _ in 0..6 {
        x = x.wrapping_mul(2u32.wrapping_sub(a.wrapping_mul(x)));
    }
    GR32_4::from_gr1(GR32_1(x))
}

/// β-basis Lagrange polynomials per the paper eq. 84-85:
///   ℓ_0(t) = 1 - t^15
///   ℓ_i(t) = (1/15) · ∏_{j ≠ i} (t - e_j)   for i ≠ 0
fn ell_beta(i: usize) -> Poly {
    if i == 0 {
        let mut p = vec![GR32_4::ZERO; 16];
        p[0] = GR32_4::ONE;
        p[15] = -GR32_4::ONE;
        pn(p)
    } else {
        let mut p: Poly = vec![GR32_4::ONE];
        for j in 0..16 {
            if j == i {
                continue;
            }
            // (t - e_j)
            let lin = vec![-T1[j], GR32_4::ONE];
            p = pmul(&p, &lin);
        }
        pscale(&p, inv_15())
    }
}

/// α-basis Lagrange polynomials per the paper eq. 74:
///   ℓ_0(t) = 1 - t,  ℓ_1(t) = t  (as polynomials in R_1[t])
fn ell_alpha(i: usize) -> Poly {
    match i {
        0 => vec![GR32_4::ONE, -GR32_4::ONE],
        1 => vec![GR32_4::ZERO, GR32_4::ONE],
        _ => panic!("α basis index out of range"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// α partition of unity (polynomial identity): (1-t) + t = 1.
#[test]
fn alpha_partition_of_unity() {
    let sum = padd(&ell_alpha(0), &ell_alpha(1));
    assert_eq!(sum, vec![GR32_4::ONE]);
}

/// α delta property: ℓ_j(e_k) = δ_{j,k} for e_0=0, e_1=1.
#[test]
fn alpha_delta_property() {
    let pts = [GR32_4::ZERO, GR32_4::ONE];
    for j in 0..2 {
        let p = ell_alpha(j);
        for (k, &t) in pts.iter().enumerate() {
            let v = peval(&p, t);
            let expected = if j == k { GR32_4::ONE } else { GR32_4::ZERO };
            assert_eq!(v, expected, "ℓ_{j}({:?}) ≠ δ_{{j,k={k}}}", t);
        }
    }
}

/// β partition of unity (polynomial identity): Σ ℓ_i(t) = 1 in R_1[t].
///
/// This is the strongest non-trivial check on the closed-form formulas in
/// the paper eq. 84-85. It would fail (silently giving non-constant 1) if any
/// of the ℓ_i had a transcription error.
#[test]
fn beta_partition_of_unity() {
    let mut sum: Poly = vec![];
    for i in 0..16 {
        sum = padd(&sum, &ell_beta(i));
    }
    assert_eq!(sum, vec![GR32_4::ONE], "Σ ℓ_i(t) is not the constant 1");
}

/// β delta property: ℓ_i(e_k) = δ_{i,k}.
#[test]
fn beta_delta_property() {
    for i in 0..16 {
        let p = ell_beta(i);
        for k in 0..16 {
            let v = peval(&p, T1[k]);
            let expected = if i == k { GR32_4::ONE } else { GR32_4::ZERO };
            assert_eq!(v, expected, "ℓ_{i}(T1[{k}]) ≠ δ_{{i,k}}");
        }
    }
}

/// v_α = t² - t  vanishes on T_0 = {0, 1}.
#[test]
fn v_alpha_vanishes_on_t0() {
    // v_α(t) = t² - t
    let v_alpha: Poly = vec![GR32_4::ZERO, -GR32_4::ONE, GR32_4::ONE];
    assert_eq!(peval(&v_alpha, GR32_4::ZERO), GR32_4::ZERO);
    assert_eq!(peval(&v_alpha, GR32_4::ONE), GR32_4::ZERO);
}

/// v_β = t^16 - t  vanishes on T_1 (the Teichmüller property).
#[test]
fn v_beta_vanishes_on_t1() {
    // v_β(t) = t^16 - t
    let mut v_beta: Poly = vec![GR32_4::ZERO; 17];
    v_beta[1] = -GR32_4::ONE;
    v_beta[16] = GR32_4::ONE;
    let v_beta = pn(v_beta);
    for i in 0..16 {
        assert_eq!(peval(&v_beta, T1[i]), GR32_4::ZERO, "v_β(T1[{i}]) ≠ 0");
    }
}

/// α-side quotient identity (using v_α = α² - α, the *minimal* vanishing
/// polynomial). For every (j, k):
///
///   ℓ_j(t)·ℓ_k(t) - δ_{j,k}·ℓ_j(t)  =  q_{j,k}(t) · (t² - t)
///
/// with the polynomial division having ZERO remainder. Furthermore, the
/// quotient comes out to constants ±1 (matching the closed-form Q_ALPHA).
#[test]
fn alpha_quotient_identity() {
    let v_alpha: Poly = vec![GR32_4::ZERO, -GR32_4::ONE, GR32_4::ONE]; // t² - t
    let expected_q = [[1i64, -1], [-1, 1]];
    for j in 0..2 {
        for k in 0..2 {
            let prod = pmul(&ell_alpha(j), &ell_alpha(k));
            let lhs = if j == k {
                psub(&prod, &ell_alpha(j))
            } else {
                prod
            };
            let (q, rem) = pdiv(&lhs, &v_alpha);
            assert!(rem.is_empty(), "α quotient remainder ≠ 0 at (j={j},k={k})");
            // quotient should be a single constant
            assert_eq!(q.len(), 1, "α quotient is not a constant");
            let exp = if expected_q[j][k] == 1 {
                GR32_4::ONE
            } else {
                -GR32_4::ONE
            };
            assert_eq!(q[0], exp, "α q_{{{j},{k}}} ≠ {}", expected_q[j][k]);
        }
    }
}

/// β-side **explicit** quotient formula (paper Appendix B.2).
///
/// For i ≠ j, the paper gives a closed form for q_{i,j}(X) as an R_1-linear
/// combination of ℓ_i(X) and ℓ_j(X). The formula contains inverses
/// (1/(e_i-e_j), 1/e_j) which we don't have in the algebra crate, so we test
/// the **denominator-cleared** version of the identity:
///
///   case i,j ≠ 0:    15(e_i - e_j) · ℓ_i·ℓ_j  ==  (ℓ_i - ℓ_j) · v
///   case i=0, j ≠ 0: 15·e_j        · ℓ_0·ℓ_j  ==  (-ℓ_0 - 15·ℓ_j) · v
///   case i≠0, j=0:   15·e_i        · ℓ_i·ℓ_0  ==  (-15·ℓ_i - ℓ_0) · v
///
/// All three cases reduce to the explicit formula in the paper after
/// dividing by the unit constants. (The i = j case has no explicit formula in
/// the paper — we already test divisibility for it in `beta_quotient_identity`.)
#[test]
fn beta_quotient_explicit_formula_inverse_free() {
    // v(X) = X^16 - X
    let mut v: Poly = vec![GR32_4::ZERO; 17];
    v[1] = -GR32_4::ONE;
    v[16] = GR32_4::ONE;
    let v = pn(v);

    let fifteen = GR32_4::from_gr1(GR32_1(15));

    for i in 0..16 {
        for j in 0..16 {
            if i == j {
                continue;
            }
            let li = ell_beta(i);
            let lj = ell_beta(j);
            let prod = pmul(&li, &lj);

            // Compute LHS scalar coefficient and RHS = (combination)·v.
            let (scalar, combo): (GR32_4, Poly) = if i != 0 && j != 0 {
                // 15(e_i - e_j) · ℓ_i·ℓ_j == (ℓ_i - ℓ_j) · v
                (fifteen * (T1[i] - T1[j]), psub(&li, &lj))
            } else if i == 0 {
                // 15·e_j · ℓ_0·ℓ_j == (-ℓ_0 - 15·ℓ_j) · v
                let neg_l0 = pscale(&li, -GR32_4::ONE);
                let neg_15_lj = pscale(&lj, -fifteen);
                let combo = padd(&neg_l0, &neg_15_lj);
                (fifteen * T1[j], combo)
            } else {
                // i ≠ 0, j = 0:  15·e_i · ℓ_i·ℓ_0 == (-15·ℓ_i - ℓ_0) · v
                let neg_15_li = pscale(&li, -fifteen);
                let neg_l0 = pscale(&lj, -GR32_4::ONE);
                let combo = padd(&neg_15_li, &neg_l0);
                (fifteen * T1[i], combo)
            };

            let lhs = pscale(&prod, scalar);
            let rhs = pmul(&combo, &v);
            assert_eq!(
                lhs, rhs,
                "the Appendix B.2 explicit q_{{i,j}} formula failed at (i={i}, j={j})"
            );
        }
    }
}

/// β-side quotient identity (using v_β = β^16 - β). For every (i, j),
/// the polynomial `ℓ_i(t)·ℓ_j(t) - δ_{i,j}·ℓ_i(t)` is divisible by `t^16 - t`
/// with zero remainder.
#[test]
fn beta_quotient_identity() {
    let mut v_beta: Poly = vec![GR32_4::ZERO; 17];
    v_beta[1] = -GR32_4::ONE;
    v_beta[16] = GR32_4::ONE;
    let v_beta = pn(v_beta);
    for i in 0..16 {
        for j in 0..16 {
            let prod = pmul(&ell_beta(i), &ell_beta(j));
            let lhs = if i == j {
                psub(&prod, &ell_beta(i))
            } else {
                prod
            };
            let (_, rem) = pdiv(&lhs, &v_beta);
            assert!(rem.is_empty(), "β quotient remainder ≠ 0 at (i={i},j={j})");
        }
    }
}
