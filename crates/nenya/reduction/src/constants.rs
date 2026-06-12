//! Lagrange basis, quotient-decomposition, and vanishing constants for the
//! packing embedding φ_2 — computed at runtime, generic over the base width.
//!
//! Previously this was a generated file of hardcoded `u32` literals. It is now
//! a generic computation: [`PackingConstants::compute`] builds the α/β Lagrange
//! bases, vanishing polynomials, and quotient tables for any [`Limb`] width,
//! verifying all the algebraic identities of the paper §2.3 on first use
//! (`debug_assert!`). Results are cached per type via [`constants`], so each
//! width pays the (sub-millisecond) build cost at most once.
//!
//!   - `v_alpha = α² - α`   (minimal vanishing polynomial of T_0 = {0, 1})
//!   - `v_beta  = β^16 - β`  (minimal vanishing polynomial of T_1 ⊂ R_1)
//!   - All identities `ℓ_j·ℓ_k = q_{j,k}·v + δ_{j,k}·ℓ_j` hold by construction.

use std::sync::OnceLock;

use galois_ring::{compute_t1_generic, Gr1, Gr128, Gr4, Limb};

/// All packing constants for a fixed base-limb width `L`.
pub struct PackingConstants<L: Limb> {
    /// `15⁻¹ mod 2^n`, lifted into `R_0`. Every nonzero `ℓ_i(β)` coeff carries it.
    pub inv_15: Gr1<L>,
    /// α-Lagrange basis on `T_0 = {0,1}`: `[1-α, α]`.
    pub l_alpha: [Gr4<L>; 2],
    /// `v_α = α² - α`.
    pub v_alpha: Gr4<L>,
    /// α-side quotient table `[[+1,-1],[-1,+1]]`.
    pub q_alpha: [[Gr4<L>; 2]; 2],
    /// β-Lagrange basis on `T_1`, materialized in `R_2`.
    pub l_beta: [Gr128<L>; 16],
    /// `v_β = β^16 - β`, materialized in `R_2`.
    pub v_beta: Gr128<L>,
    /// β-side quotient table `q_{i,j}(β)`.
    pub q_beta: [[Gr128<L>; 16]; 16],
    /// Preprocessed `Q_ALPHA_L_BETA[i][j][k] = φ(Q_ALPHA[j][k]) · L_BETA[i]`,
    /// collapsing the witness-independent factor of `q_1`'s closed form.
    pub q_alpha_l_beta: [[[Gr128<L>; 2]; 2]; 16],
}

// ---------------------------------------------------------------------------
// Polynomial arithmetic over R_2 (Vec<Gr128<L>>, coeff[i] = coeff of t^i).
// ---------------------------------------------------------------------------

fn poly_normalize<L: Limb>(p: &mut Vec<Gr128<L>>) {
    while p.last() == Some(&Gr128::<L>::ZERO) {
        p.pop();
    }
}

fn poly_add<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> Vec<Gr128<L>> {
    let n = a.len().max(b.len());
    let mut r = vec![Gr128::<L>::ZERO; n];
    for i in 0..a.len() {
        r[i] = r[i] + a[i];
    }
    for i in 0..b.len() {
        r[i] = r[i] + b[i];
    }
    poly_normalize(&mut r);
    r
}

fn poly_sub<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> Vec<Gr128<L>> {
    let n = a.len().max(b.len());
    let mut r = vec![Gr128::<L>::ZERO; n];
    for i in 0..a.len() {
        r[i] = r[i] + a[i];
    }
    for i in 0..b.len() {
        r[i] = r[i] - b[i];
    }
    poly_normalize(&mut r);
    r
}

fn poly_mul<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> Vec<Gr128<L>> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut r = vec![Gr128::<L>::ZERO; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            r[i + j] = r[i + j] + a[i] * b[j];
        }
    }
    poly_normalize(&mut r);
    r
}

fn poly_scale<L: Limb>(a: &[Gr128<L>], s: Gr128<L>) -> Vec<Gr128<L>> {
    let mut r: Vec<Gr128<L>> = a.iter().map(|&x| x * s).collect();
    poly_normalize(&mut r);
    r
}

/// Schoolbook poly division. Requires `b` monic (leading coeff `1`) — true for
/// `v_β` and the divisors used here.
fn poly_div<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
    assert!(!b.is_empty(), "divide by zero polynomial");
    let lc_b = *b.last().unwrap();
    assert!(
        lc_b == Gr128::<L>::ONE,
        "divisor must be monic for this routine"
    );
    if a.len() < b.len() {
        return (vec![], a.to_vec());
    }
    let mut rem = a.to_vec();
    let q_len = a.len() - b.len() + 1;
    let mut q = vec![Gr128::<L>::ZERO; q_len];
    while rem.len() >= b.len() {
        let dr = rem.len() - 1;
        let db = b.len() - 1;
        let factor = *rem.last().unwrap(); // lc_b == 1
        let q_idx = dr - db;
        q[q_idx] = q[q_idx] + factor;
        for i in 0..b.len() {
            let idx = q_idx + i;
            rem[idx] = rem[idx] - factor * b[i];
        }
        poly_normalize(&mut rem);
    }
    poly_normalize(&mut q);
    (q, rem)
}

fn p_const<L: Limb>(c: Gr128<L>) -> Vec<Gr128<L>> {
    if c == Gr128::<L>::ZERO {
        vec![]
    } else {
        vec![c]
    }
}

fn p_t_minus<L: Limb>(e: Gr128<L>) -> Vec<Gr128<L>> {
    vec![-e, Gr128::<L>::ONE]
}

/// `α = [0,1,0,0]`, the natural lift of the primitive element of `F_16`.
fn alpha<L: Limb>() -> Gr4<L> {
    Gr4([L::ZERO, L::ONE, L::ZERO, L::ZERO])
}

/// `β = Y`, the canonical generator of `R_2` (second `Gr4` slot).
fn beta<L: Limb>() -> Gr128<L> {
    let mut arr = [Gr4::<L>::ZERO; 32];
    arr[1] = Gr4::<L>::ONE;
    Gr128(arr)
}

/// Evaluate `p(t) ∈ R_2[t]` at `t = β` (Horner), materializing a polynomial in
/// `t` into a single `R_2` element.
fn eval_at_beta<L: Limb>(p: &[Gr128<L>]) -> Gr128<L> {
    let b = beta::<L>();
    let mut acc = Gr128::<L>::ZERO;
    for &c in p.iter().rev() {
        acc = acc * b + c;
    }
    acc
}

/// β-Lagrange basis as polynomials in `t`, from the closed form of the paper
/// §2.3.2: `ℓ_0(t) = 1 - t^15`, `ℓ_i(t) = (1/15) ∏_{j≠i} (t - e_j)` for `i ≠ 0`.
fn build_l_beta<L: Limb>(e_pts: &[Gr128<L>], inv15: Gr128<L>) -> Vec<Vec<Gr128<L>>> {
    assert_eq!(e_pts.len(), 16);
    let mut out = Vec::with_capacity(16);

    // ℓ_0(t) = 1 - t^15
    {
        let mut p = vec![Gr128::<L>::ZERO; 16];
        p[0] = Gr128::<L>::ONE;
        p[15] = -Gr128::<L>::ONE;
        poly_normalize(&mut p);
        out.push(p);
    }
    // ℓ_i(t) = (1/15) · ∏_{j ≠ i} (t - e_j)  for i ≠ 0
    for i in 1..16 {
        let mut p = p_const(Gr128::<L>::ONE);
        for j in 0..16 {
            if j == i {
                continue;
            }
            p = poly_mul(&p, &p_t_minus(e_pts[j]));
        }
        out.push(poly_scale(&p, inv15));
    }
    out
}

/// `v_β(t) = t^16 - t`.
fn build_v_beta<L: Limb>() -> Vec<Gr128<L>> {
    let mut p = vec![Gr128::<L>::ZERO; 17];
    p[1] = -Gr128::<L>::ONE;
    p[16] = Gr128::<L>::ONE;
    poly_normalize(&mut p);
    p
}

impl<L: Limb> PackingConstants<L> {
    /// Build all constants and verify the Lagrange/quotient identities.
    pub fn compute() -> Self {
        // --- 15⁻¹ ---
        let inv_15 = Gr1(L::from_u64(15).inv_2adic());
        let inv15_r2 = Gr128::from_gr1(inv_15);

        // --- α basis ---
        let a = alpha::<L>();
        let l_alpha = [Gr4::<L>::ONE - a, a];
        let v_alpha = a * a - a;
        // Q_ALPHA = [[+1,-1],[-1,+1]] (derived in the paper §2.3.4).
        let q_alpha = [
            [Gr4::<L>::ONE, -Gr4::<L>::ONE],
            [-Gr4::<L>::ONE, Gr4::<L>::ONE],
        ];

        debug_assert_eq!(
            l_alpha[0] + l_alpha[1],
            Gr4::<L>::ONE,
            "α-basis partition of unity failed"
        );
        for j in 0..2 {
            for k in 0..2 {
                let lhs = l_alpha[j] * l_alpha[k];
                let rhs =
                    q_alpha[j][k] * v_alpha + if j == k { l_alpha[j] } else { Gr4::<L>::ZERO };
                debug_assert_eq!(lhs, rhs, "α-side quotient identity failed ({j},{k})");
            }
        }

        // --- β basis ---
        let t1 = compute_t1_generic::<L>();
        let e_pts: Vec<Gr128<L>> = t1.iter().map(|&e| Gr128::from_gr4(e)).collect();

        let l_beta_polys = build_l_beta(&e_pts, inv15_r2);
        let v_beta_poly = build_v_beta::<L>();

        let mut l_beta = [Gr128::<L>::ZERO; 16];
        for (i, p) in l_beta_polys.iter().enumerate() {
            l_beta[i] = eval_at_beta(p);
        }
        let v_beta = eval_at_beta(&v_beta_poly);

        // Σ ℓ_i(t) == 1 (as a polynomial), and == 1 after eval at β.
        let mut poly_sum: Vec<Gr128<L>> = vec![];
        for p in &l_beta_polys {
            poly_sum = poly_add(&poly_sum, p);
        }
        debug_assert!(
            poly_sum.len() == 1 && poly_sum[0] == Gr128::<L>::ONE,
            "β polynomial partition of unity failed"
        );
        let sum_l = l_beta.iter().copied().fold(Gr128::<L>::ZERO, |x, y| x + y);
        debug_assert_eq!(sum_l, Gr128::<L>::ONE, "β-basis partition of unity failed");

        // v_β = β^16 - β (independent check).
        let b = beta::<L>();
        debug_assert_eq!(v_beta, b.pow(16) - b, "v_β identity failed");

        // Q_BETA[i][j] = (ℓ_i·ℓ_j - δ_{i,j}·ℓ_i) / v_β, evaluated at β.
        let mut q_beta = [[Gr128::<L>::ZERO; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let prod = poly_mul(&l_beta_polys[i], &l_beta_polys[j]);
                let to_divide = if i == j {
                    poly_sub(&prod, &l_beta_polys[i])
                } else {
                    prod
                };
                let (q, rem) = poly_div(&to_divide, &v_beta_poly);
                debug_assert!(rem.is_empty(), "β quotient remainder nonzero ({i},{j})");
                q_beta[i][j] = eval_at_beta(&q);
            }
        }
        for i in 0..16 {
            for j in 0..16 {
                let lhs = l_beta[i] * l_beta[j];
                let rhs = q_beta[i][j] * v_beta + if i == j { l_beta[i] } else { Gr128::<L>::ZERO };
                debug_assert_eq!(lhs, rhs, "β-side quotient identity failed ({i},{j})");
            }
        }

        // Q_ALPHA_L_BETA[i][j][k] = φ(Q_ALPHA[j][k]) · L_BETA[i].
        let mut q_alpha_l_beta = [[[Gr128::<L>::ZERO; 2]; 2]; 16];
        for i in 0..16 {
            for j in 0..2 {
                for k in 0..2 {
                    q_alpha_l_beta[i][j][k] = Gr128::from_gr4(q_alpha[j][k]) * l_beta[i];
                }
            }
        }

        PackingConstants {
            inv_15,
            l_alpha,
            v_alpha,
            q_alpha,
            l_beta,
            v_beta,
            q_beta,
            q_alpha_l_beta,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-type cache. A `static OnceLock<PackingConstants<L>>` cannot live at item
// scope (statics can't be generic), so each width gets its own monomorphic
// static inside its `HasPackingConstants` impl.
// ---------------------------------------------------------------------------

/// Limb widths for which packing constants are available (all of them).
pub trait HasPackingConstants: Limb + Sized {
    fn packing_constants() -> &'static PackingConstants<Self>;
}

macro_rules! cache_for {
    ($t:ty) => {
        impl HasPackingConstants for $t {
            fn packing_constants() -> &'static PackingConstants<$t> {
                static C: OnceLock<PackingConstants<$t>> = OnceLock::new();
                C.get_or_init(PackingConstants::<$t>::compute)
            }
        }
    };
}
cache_for!(u8);
cache_for!(u16);
cache_for!(u32);
cache_for!(u64);

/// Lazily-computed, cached packing constants for base width `L`.
#[inline]
pub fn constants<L: HasPackingConstants>() -> &'static PackingConstants<L> {
    L::packing_constants()
}
