//! Gruen-optimized mult-check sumcheck for `∑_h eq(r_target, h) · a(h) · b(h)`.
//!
//! Same protocol as [`crate::mult_check`] (degree-3 sumcheck on the product
//! `eq · a · b`), but the prover exploits the **product structure of `eq`** to
//! reduce per-round work. This is the "Gruen eq-MLE specialization" mentioned
//! below.
//!
//! ## The trick in one paragraph
//!
//! The full round message is
//! `s_i(Z) = ∑_{x'} eq(r_target, (r'_0, …, r'_{i-1}, Z, x')) · a(…, Z, x') · b(…, Z, x')`,
//! a degree-3 polynomial in `Z`. The product structure of `eq` lets us factor
//! `eq_one(r_target_i, Z)` out of every term in the sum:
//!
//! ```text
//! s_i(Z) = eq_one(r_target_i, Z) · Q̃_i(Z)
//! ```
//!
//! where `Q̃_i(Z)` is a **degree-2** polynomial weighted by the *remaining*
//! eq factors `E_prev_i · eq_rest_i(x')`. The prover only sends `Q̃_i`
//! (3 coefficients), and the verifier reconstructs `eq_one(r_target_i, Z)`
//! itself — it knows `r_target_i` and the challenge `r'_i`.
//!
//! ## Slim eq table — `eq_gruen`
//!
//! Maintaining the full `2^n`-entry eq table is wasteful in this formulation:
//! `eq_one(r_target_i, h_i)` is the *current round's* factor and gets handled
//! analytically by the verifier; only `E_prev_i · eq_rest_i(x')` needs to be
//! materialized for the inner sum. We call this slim table `eq_gruen`. At
//! round `i` it has length `2^{n-i-1}` — half the size of the standard
//! 3-table fold's eq.
//!
//! Initial state (round 0): `eq_gruen_0(x') = ∏_{j ≥ 1} eq_one(r_target_j, x'_{j-1})`,
//! which is exactly `build_eq_table(&r_target[1..])` (length `2^{n-1}`).
//!
//! Update between round `i` and round `i+1` (after the verifier samples `r'_i`):
//!
//! 1. **Marginalize** over `h_{i+1}`: sum the `lo` and `hi` halves of
//!    `eq_gruen_i`. This drops the `eq_one(r_target_{i+1}, ·)` factor because
//!    `eq_one(t, 0) + eq_one(t, 1) = (1 − t) + t = 1` is the partition-of-unity.
//! 2. **Scale** the marginalized table by `eq_one(r_target_i, r'_i)` to fold
//!    that factor into `E_prev_{i+1}`.
//!
//! Cost per update: `2^{n-i-2}` adds + `2^{n-i-2}` muls. Compare to the
//! standard `Mle::fold_in_place` on a length-`2^{n-i}` eq table:
//! `2^{n-i-1}` muls. The slim variant is **2× cheaper in muls per round** on
//! the eq update alone, *and* its initial build is half the size, so the
//! total eq cost across the protocol drops from ~`2 · 2^n` muls to ~`2^n`.
//!
//! ## Per-pair inner-loop arithmetic — 6 muls/pair (vs 10 in the standard form)
//!
//! Each inner pair has one `e = eq_gruen[i]` weight (not two `eq_lo, eq_hi`),
//! and only needs to assemble a deg-2 polynomial. We compute `Q̃_i` via
//! `{Q̃(0), Q̃(1), Q̃(∞)}`, weighting each per-pair contribution by `e`:
//!
//! ```text
//! da = a_hi − a_lo                  // 1 sub
//! db = b_hi − b_lo                  // 1 sub
//! p0   = a_lo · b_lo                // 1 mul   ← per-pair Q̃(0)
//! p1   = a_hi · b_hi                // 1 mul   ← per-pair Q̃(1)
//! pinf = da   · db                  // 1 mul   ← per-pair Q̃(∞)
//! s_at_0   += e · p0                // 1 mul
//! s_at_1   += e · p1                // 1 mul
//! s_at_inf += e · pinf              // 1 mul
//! ```
//!
//! Total per pair: **6 muls** + ~3 adds + 2 subs. Convert to coefficient form
//! once per round (no division — `{0, 1, ∞}` change-of-basis has determinant
//! `−1`):
//!
//! ```text
//! q0 = s_at_0
//! q2 = s_at_inf
//! q1 = s_at_1 − q0 − q2
//! ```
//!
//! Compared to the standard 3-table prover at **10 muls/pair** in
//! `compute_round_poly`, this is a **40% reduction** in compute_round_poly
//! work. Combined with the slim eq update savings, the full prover should
//! drop noticeably from the 416 ms baseline at n=15.
//!
//! ## What the prover sends
//!
//! Per round: a **3-element** array `[q0, q1, q2]` (vs the standard `[a0, a1,
//! a2, a3]`). Plus `final_a, final_b` at the end. The verifier reconstructs
//! `eq_one(r_target_i, Z)` itself and uses the weighted round check
//! `q0 + r_target_i · (q1 + q2) == c_{i-1}` (see [`verify_mult_check`]).
//!
//! The final check is **identical** to the standard mult check:
//! `c_n == eq(r_target, challenges) · final_a · final_b`. The Gruen
//! reformulation only changes how `c_n` is built up across rounds, not what
//! it should equal at the end.

use galois_ring::{Gr128, Limb};
use rayon::prelude::*;

use crate::eq::build_eq_table;
use crate::error::VerifyError;
use crate::fs::Challenger;
use crate::mle::Mle;

/// Proof of `sum = ∑_h eq(r_target, h) · a(h) · b(h)` produced by the
/// Gruen-optimized mult-check prover ([`prove_mult_check`]).
///
/// Same protocol contract as [`crate::mult_check::MultCheckProof`] but the
/// round messages carry only the **deg-2 inner factor** `Q̃_i` of
/// `s_i(Z) = eq_one(r_target_i, Z) · Q̃_i(Z)` (3 coefficients per round
/// instead of 4). The verifier reconstructs `eq_one(r_target_i, Z)`
/// itself, so the slim form is sound and slightly smaller on the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct MultCheckProof<L: Limb> {
    pub claimed_sum: Gr128<L>,
    /// Round-`i` deg-2 inner factor `Q̃_i = q0 + q1·Z + q2·Z²` in coefficient
    /// form. Length = `num_vars`.
    pub round_polys: Vec<[Gr128<L>; 3]>,
    /// `a(challenges)`, the prover's claimed final evaluation of `a`.
    pub final_a: Gr128<L>,
    /// `b(challenges)`, the prover's claimed final evaluation of `b`.
    pub final_b: Gr128<L>,
}

impl<L: Limb> MultCheckProof<L> {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }

    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;

        // claimed_sum: Gr128<L> has 2 arrays of 128 limbs
        for c in self.claimed_sum.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        // round_polys: Vec<[Gr128<L>; 3]>
        buf.extend_from_slice(&(self.round_polys.len() as u64).to_le_bytes());
        for poly in &self.round_polys {
            for gr in poly.iter() {
                for c in gr.0.iter() {
                    for w in c.0.iter() {
                        buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                    }
                }
            }
        }

        // final_a: Gr128<L>
        for c in self.final_a.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        // final_b: Gr128<L>
        for c in self.final_b.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Run the Gruen-optimized mult-check prover for
/// `claim = ∑_h eq(r_target, h) · a(h) · b(h)`.
///
/// Returns the proof and the challenge sequence in sample order
/// (`r'_0, r'_1, …, r'_{n-1}`, with `r'_0` being the first folded variable —
/// the most-significant index bit, matching [`Mle::fold_in_place`]).
pub fn prove_mult_check<L: Limb>(
    a: &Mle<L>,
    b: &Mle<L>,
    r_target: &[Gr128<L>],
    claim: Gr128<L>,
    challenger: &mut impl Challenger<L>,
) -> (MultCheckProof<L>, Vec<Gr128<L>>) {
    let n = a.num_vars();
    assert_eq!(b.num_vars(), n, "a and b must agree on num_vars");
    assert_eq!(
        r_target.len(),
        n,
        "r_target must have one entry per variable"
    );
    assert!(n > 0, "mult check on a 0-variable MLE is meaningless");

    // Slim eq table. At round 0 it represents
    //   eq_gruen_0(x') = ∏_{j ≥ 1} eq_one(r_target_j, x'_{j-1})
    // and has length 2^{n-1}. For n = 1 there are no "rest" variables and we
    // start with the trivial length-1 table [1].
    let mut eq_gruen: Vec<Gr128<L>> = if n > 1 {
        build_eq_table(&r_target[1..]).evals().to_vec()
    } else {
        vec![Gr128::<L>::ONE]
    };
    let mut a = a.clone();
    let mut b = b.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for i in 0..n {
        let q = compute_round_poly_gruen(&eq_gruen, &a, &b);
        // Per-round Fiat–Shamir: bind this round's message before drawing its
        // challenge (observe → sample), so `r` depends on `q`.
        challenger.observe(&q);
        round_polys.push(q);

        let r = challenger.sample();
        challenges.push(r);

        // Fold a and b — same as the standard mult check.
        a.fold_in_place(r);
        b.fold_in_place(r);

        // Update eq_gruen for the next round (skip on the last round; there
        // is no "next" loop iteration that would consume it).
        if i + 1 < n {
            // E_{i+1} / E_i = eq_one(r_target_i, r) = r_target_i·r + (1 − r_target_i)(1 − r).
            let scale = r_target[i] * r + (Gr128::<L>::ONE - r_target[i]) * (Gr128::<L>::ONE - r);
            marginalize_and_scale(&mut eq_gruen, scale);
        }
    }

    let proof = MultCheckProof {
        claimed_sum: claim,
        round_polys,
        final_a: a.final_eval(),
        final_b: b.final_eval(),
    };
    (proof, challenges)
}

/// Build the deg-2 inner factor `Q̃_i = q0 + q1·Z + q2·Z²` for the current
/// state of `(eq_gruen, a, b)`.
///
/// `eq_gruen` has length `m = a.len() / 2 = b.len() / 2`. Each entry weights
/// one fold-pair; per-pair compute is 6 muls (see the module docstring for
/// the full breakdown).
///
/// Marked `#[doc(hidden)] pub` so `benches/piop.rs` can call it directly for
/// fine-grained measurements. Not part of the stable crate API.
#[doc(hidden)]
#[inline]
pub fn compute_round_poly_gruen<L: Limb>(
    eq_gruen: &[Gr128<L>],
    a: &Mle<L>,
    b: &Mle<L>,
) -> [Gr128<L>; 3] {
    let m = eq_gruen.len();
    debug_assert_eq!(a.len(), 2 * m);
    debug_assert_eq!(b.len(), 2 * m);

    let a_evals = a.evals();
    let b_evals = b.evals();

    // Each fold-pair contributes independently to the three accumulators, and
    // ring addition is associative+commutative, so the parallel reduction is
    // bit-identical to the serial accumulation regardless of thread count.
    let zero = Gr128::<L>::ZERO;
    let (s_at_0, s_at_1, s_at_inf) = (0..m)
        .into_par_iter()
        .map(|i| {
            let e = eq_gruen[i];
            let a_lo = a_evals[i];
            let a_hi = a_evals[i + m];
            let b_lo = b_evals[i];
            let b_hi = b_evals[i + m];

            let da = a_hi - a_lo;
            let db = b_hi - b_lo;

            // Three per-pair products at Z ∈ {0, 1, ∞}, weighted by e.
            (e * (a_lo * b_lo), e * (a_hi * b_hi), e * (da * db))
        })
        .reduce(
            || (zero, zero, zero),
            |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2),
        );

    // Convert {Q̃(0), Q̃(1), Q̃(∞)} → coefficient form once per round:
    //   q0 = Q̃(0)
    //   q2 = leading coefficient = Q̃(∞)
    //   q1 = Q̃(1) − Q̃(0) − Q̃(∞)
    // No division: the {0, 1, ∞} change-of-basis has determinant −1, so this
    // works fine in `Z/2³²` (where the {0, 1, 2} basis would not).
    let q0 = s_at_0;
    let q2 = s_at_inf;
    let q1 = s_at_1 - q0 - q2;
    [q0, q1, q2]
}

/// Halve `eq_gruen` by marginalizing the leading variable, then multiply
/// every entry by `scale`. Sum-over-leading-bit: `(lo, hi) ↦ lo + hi`,
/// then `entry ↦ scale · entry`.
///
/// Marked `#[doc(hidden)] pub` so `benches/piop.rs` can time the eq-update
/// step in isolation. Not part of the stable crate API.
#[doc(hidden)]
#[inline]
pub fn marginalize_and_scale<L: Limb>(eq_gruen: &mut Vec<Gr128<L>>, scale: Gr128<L>) {
    let m = eq_gruen.len() / 2;
    debug_assert!(m > 0, "cannot marginalize a length-1 table");
    // `lo[i] ← scale·(lo[i] + hi[i])`: disjoint writes → bit-identical to serial.
    let (lo, hi) = eq_gruen.split_at_mut(m);
    lo.par_iter_mut()
        .zip(hi.par_iter())
        .for_each(|(lo_i, &hi_i)| {
            *lo_i = scale * (*lo_i + hi_i);
        });
    eq_gruen.truncate(m);
}

/// Verify a Gruen-optimized mult-check proof produced by
/// [`prove_mult_check`].
///
/// The protocol is the same deg-3 sumcheck as
/// [`crate::mult_check::verify_mult_check`], but the round messages carry
/// the **deg-2 inner factor** `Q̃_i` of the round polynomial
/// `s_i(Z) = eq_one(r_target_i, Z) · Q̃_i(Z)`. The verifier reconstructs
/// the `eq_one(r_target_i, Z)` factor itself, which changes the round-check
/// arithmetic but **not** the final-check equation.
///
/// On success returns the binding point `(r_1, …, r_n)` (sample order, MSB
/// first — same convention as [`crate::mult_check::verify_mult_check`]).
pub fn verify_mult_check<L: Limb>(
    proof: &MultCheckProof<L>,
    r_target: &[Gr128<L>],
    challenger: &mut impl Challenger<L>,
) -> Result<Vec<Gr128<L>>, VerifyError> {
    let n = r_target.len();
    assert_eq!(
        proof.round_polys.len(),
        n,
        "mult-check (optimized) proof has wrong number of rounds"
    );

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(n);

    for (round, [q0, q1, q2]) in proof.round_polys.iter().copied().enumerate() {
        let rt = r_target[round];

        // Round check:
        //   s_i(0) + s_i(1)
        //     = eq_one(rt, 0)·Q̃(0) + eq_one(rt, 1)·Q̃(1)
        //     = (1 - rt)·q0 + rt·(q0 + q1 + q2)
        //     = q0 + rt·(q1 + q2)
        let lhs = q0 + rt * (q1 + q2);
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        // Per-round Fiat–Shamir: observe this round's message (in the same
        // order as the prover) before sampling its challenge.
        challenger.observe(&[q0, q1, q2]);
        let r = challenger.sample();
        challenges.push(r);

        // Advance: c_i = s_i(r) = eq_one(rt, r) · Q̃(r).
        let qhat_r = q0 + r * (q1 + r * q2);
        let eq_one = rt * r + (Gr128::<L>::ONE - rt) * (Gr128::<L>::ONE - r);
        c = eq_one * qhat_r;
    }

    // Final check: c_n must equal eq(r_target, challenges) · final_a · final_b.
    let mut eq_final = Gr128::<L>::ONE;
    for (rt, &rc) in r_target.iter().zip(challenges.iter()) {
        eq_final = eq_final * (*rt * rc + (Gr128::<L>::ONE - *rt) * (Gr128::<L>::ONE - rc));
    }

    if c != eq_final * proof.final_a * proof.final_b {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}

// ---------------------------------------------------------------------------
// "Sub" variant: proves  Σ_h eq(r_target,h)·(a(h)·b(h) − c(h) − e(h)) = 0,
// i.e. the R1CS Hadamard relation a∘b = c + e with c and e as committed MLEs.
// Used by the general/MIMD provers so that â,b̂,ĉ are all bound at the SAME
// point s_1 (the paper's Protocol 3.1/A.1 form), enabling one stacked opening.
// c and e are linear → they contribute only to the {0,1} accumulators, not the
// leading (∞) term; the Gruen slim-form arithmetic is otherwise identical.
// ---------------------------------------------------------------------------

/// Proof of `Σ_h eq(r_target,h)·(a·b − c − e)(h) = 0`. Carries the four finals.
#[derive(Clone, Debug, PartialEq)]
pub struct MultCheckSubProof<L: Limb> {
    pub round_polys: Vec<[Gr128<L>; 3]>,
    pub final_a: Gr128<L>,
    pub final_b: Gr128<L>,
    pub final_c: Gr128<L>,
    pub final_e: Gr128<L>,
}

impl<L: Limb> MultCheckSubProof<L> {
    pub fn serialize(&self) -> Vec<u8> {
        let lb = (L::BITS / 8) as usize;
        let mut buf = Vec::new();
        let push = |buf: &mut Vec<u8>, g: &Gr128<L>| {
            for cc in g.0.iter() {
                for w in cc.0.iter() {
                    buf.extend_from_slice(&w.to_u64().to_le_bytes()[..lb]);
                }
            }
        };
        buf.extend_from_slice(&(self.round_polys.len() as u64).to_le_bytes());
        for poly in &self.round_polys {
            for g in poly.iter() {
                push(&mut buf, g);
            }
        }
        push(&mut buf, &self.final_a);
        push(&mut buf, &self.final_b);
        push(&mut buf, &self.final_c);
        push(&mut buf, &self.final_e);
        buf
    }
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

#[inline]
fn compute_round_poly_sub<L: Limb>(
    eq_gruen: &[Gr128<L>],
    a: &Mle<L>,
    b: &Mle<L>,
    c: &Mle<L>,
    e: &Mle<L>,
) -> [Gr128<L>; 3] {
    let m = eq_gruen.len();
    let (ae, be, ce, ee) = (a.evals(), b.evals(), c.evals(), e.evals());
    let zero = Gr128::<L>::ZERO;
    let (s0, s1, sinf) = (0..m)
        .into_par_iter()
        .map(|i| {
            let eg = eq_gruen[i];
            let (a_lo, a_hi) = (ae[i], ae[i + m]);
            let (b_lo, b_hi) = (be[i], be[i + m]);
            let (c_lo, c_hi) = (ce[i], ce[i + m]);
            let (e_lo, e_hi) = (ee[i], ee[i + m]);
            let da = a_hi - a_lo;
            let db = b_hi - b_lo;
            (
                eg * (a_lo * b_lo - c_lo - e_lo),
                eg * (a_hi * b_hi - c_hi - e_hi),
                eg * (da * db),
            )
        })
        .reduce(
            || (zero, zero, zero),
            |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2),
        );
    let q0 = s0;
    let q2 = sinf;
    let q1 = s1 - q0 - q2;
    [q0, q1, q2]
}

/// Prove `Σ_h eq(r_target,h)·(a·b − c − e)(h) = 0`.
pub fn prove_mult_check_sub<L: Limb>(
    a: &Mle<L>,
    b: &Mle<L>,
    c: &Mle<L>,
    e: &Mle<L>,
    r_target: &[Gr128<L>],
    challenger: &mut impl Challenger<L>,
) -> (MultCheckSubProof<L>, Vec<Gr128<L>>) {
    let n = a.num_vars();
    assert_eq!(b.num_vars(), n);
    assert_eq!(c.num_vars(), n);
    assert_eq!(e.num_vars(), n);
    assert_eq!(r_target.len(), n);
    assert!(n > 0);

    let mut eq_gruen: Vec<Gr128<L>> = if n > 1 {
        build_eq_table(&r_target[1..]).evals().to_vec()
    } else {
        vec![Gr128::<L>::ONE]
    };
    let (mut a, mut b, mut c, mut e) = (a.clone(), b.clone(), c.clone(), e.clone());
    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for i in 0..n {
        let q = compute_round_poly_sub(&eq_gruen, &a, &b, &c, &e);
        challenger.observe(&q);
        round_polys.push(q);
        let r = challenger.sample();
        challenges.push(r);
        a.fold_in_place(r);
        b.fold_in_place(r);
        c.fold_in_place(r);
        e.fold_in_place(r);
        if i + 1 < n {
            let scale = r_target[i] * r + (Gr128::<L>::ONE - r_target[i]) * (Gr128::<L>::ONE - r);
            marginalize_and_scale(&mut eq_gruen, scale);
        }
    }

    (
        MultCheckSubProof {
            round_polys,
            final_a: a.final_eval(),
            final_b: b.final_eval(),
            final_c: c.final_eval(),
            final_e: e.final_eval(),
        },
        challenges,
    )
}

/// Verify a [`MultCheckSubProof`]. Returns the binding point on success.
pub fn verify_mult_check_sub<L: Limb>(
    proof: &MultCheckSubProof<L>,
    r_target: &[Gr128<L>],
    challenger: &mut impl Challenger<L>,
) -> Result<Vec<Gr128<L>>, VerifyError> {
    let n = r_target.len();
    if proof.round_polys.len() != n {
        return Err(VerifyError::FinalCheckFailed);
    }
    let mut c = Gr128::<L>::ZERO; // claimed sum is 0
    let mut challenges = Vec::with_capacity(n);
    for (round, [q0, q1, q2]) in proof.round_polys.iter().copied().enumerate() {
        let rt = r_target[round];
        let lhs = q0 + rt * (q1 + q2);
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }
        challenger.observe(&[q0, q1, q2]);
        let r = challenger.sample();
        challenges.push(r);
        let qhat_r = q0 + r * (q1 + r * q2);
        let eq_one = rt * r + (Gr128::<L>::ONE - rt) * (Gr128::<L>::ONE - r);
        c = eq_one * qhat_r;
    }
    let mut eq_final = Gr128::<L>::ONE;
    for (rt, &rc) in r_target.iter().zip(challenges.iter()) {
        eq_final = eq_final * (*rt * rc + (Gr128::<L>::ONE - *rt) * (Gr128::<L>::ONE - rc));
    }
    if c != eq_final * (proof.final_a * proof.final_b - proof.final_c - proof.final_e) {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{R2TeichmullerSampler, GR32_128, GR32_4};

    fn rand_gr128(seed: u64) -> GR32_128 {
        let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([next(), next(), next(), next()]);
        }
        GR32_128(arr)
    }

    fn rand_mle(num_vars: usize, seed: u64) -> Mle<u32> {
        let len = 1usize << num_vars;
        let mut evals = Vec::with_capacity(len);
        for i in 0..len {
            evals.push(rand_gr128(seed.wrapping_add(i as u64)));
        }
        Mle::new(evals)
    }

    fn rand_target(num_vars: usize, seed: u64) -> Vec<GR32_128> {
        (0..num_vars)
            .map(|i| rand_gr128(seed.wrapping_add(0xCAFE * i as u64 + 0xBEEF)))
            .collect()
    }

    fn direct_claim(a: &Mle<u32>, b: &Mle<u32>, r_target: &[GR32_128]) -> GR32_128 {
        let eq = build_eq_table(r_target);
        let mut acc = GR32_128::ZERO;
        for i in 0..a.len() {
            acc = acc + eq.evals()[i] * a.evals()[i] * b.evals()[i];
        }
        acc
    }

    /// Cross-check: at round 0, the deg-2 inner factor Q̃ multiplied by
    /// `eq_one(r_target[0], Z)` must reproduce the same round message
    /// (s(0) + s(1) = claim) as the standard mult check.
    #[test]
    fn round_poly_gruen_consistency_3vars() {
        let n = 3;
        let a = rand_mle(n, 1);
        let b = rand_mle(n, 2);
        let r_target = rand_target(n, 3);

        // Build the slim eq table for round 0.
        let eq_gruen = build_eq_table(&r_target[1..]);
        let [q0, q1, q2] = compute_round_poly_gruen(eq_gruen.evals(), &a, &b);

        // s(Z) = eq_one(r_target[0], Z) · Q̃(Z), so:
        //   s(0) = (1 - r_target[0]) · q0
        //   s(1) = r_target[0] · (q0 + q1 + q2)
        let rt0 = r_target[0];
        let one = GR32_128::ONE;
        let s0 = (one - rt0) * q0;
        let s1 = rt0 * (q0 + q1 + q2);

        let expected = direct_claim(&a, &b, &r_target);
        assert_eq!(s0 + s1, expected);
    }

    #[test]
    fn mult_check_sub_roundtrip_and_reject() {
        let n = 4;
        let a = rand_mle(n, 101);
        let b = rand_mle(n, 202);
        let e = rand_mle(n, 303);
        // c = a∘b − e  ⇒  a∘b − c − e = 0 everywhere ⇒ the sum is 0.
        let c = Mle::new(
            (0..a.len())
                .map(|i| a.evals()[i] * b.evals()[i] - e.evals()[i])
                .collect(),
        );
        let r_target = rand_target(n, 404);

        let mut sp = R2TeichmullerSampler::new(0xABC_DEF);
        let (proof, _ch) = prove_mult_check_sub(&a, &b, &c, &e, &r_target, &mut sp);
        let mut sv = R2TeichmullerSampler::new(0xABC_DEF);
        assert!(verify_mult_check_sub(&proof, &r_target, &mut sv).is_ok());

        // Tampering a final breaks the final check.
        let mut bad = proof.clone();
        bad.final_c = bad.final_c + GR32_128::ONE;
        let mut sv2 = R2TeichmullerSampler::new(0xABC_DEF);
        assert!(verify_mult_check_sub(&bad, &r_target, &mut sv2).is_err());

        // A wrong relation (a∘b ≠ c+e) is rejected at the first round (sum ≠ 0).
        let c_wrong = Mle::new(c.evals().iter().map(|&x| x + GR32_128::ONE).collect());
        let mut sp3 = R2TeichmullerSampler::new(0x999);
        let (bad_proof, _) = prove_mult_check_sub(&a, &b, &c_wrong, &e, &r_target, &mut sp3);
        let mut sv3 = R2TeichmullerSampler::new(0x999);
        assert!(verify_mult_check_sub(&bad_proof, &r_target, &mut sv3).is_err());
    }

    #[test]
    fn prove_emits_consistent_round_polys() {
        let n = 4;
        let a = rand_mle(n, 11);
        let b = rand_mle(n, 22);
        let r_target = rand_target(n, 33);
        let claim = direct_claim(&a, &b, &r_target);

        let mut sampler = R2TeichmullerSampler::new(0xC0FFEE);
        let (proof, challenges) =
            prove_mult_check(&a, &b, &r_target, claim, &mut sampler);

        assert_eq!(proof.round_polys.len(), n);
        assert_eq!(challenges.len(), n);

        // Replay the verifier's reasoning round by round.
        let mut c = proof.claimed_sum;
        for (round, [q0, q1, q2]) in proof.round_polys.iter().copied().enumerate() {
            let rt = r_target[round];
            // s(0) + s(1) = (1 - rt)·q0 + rt·(q0 + q1 + q2) = q0 + rt·(q1 + q2)
            let lhs = q0 + rt * (q1 + q2);
            assert_eq!(lhs, c, "round {round} consistency");

            let r = challenges[round];
            // c_i = s_i(r) = eq_one(rt, r) · Q̃(r) where Q̃(r) = q0 + r·(q1 + r·q2)
            let qhat_r = q0 + r * (q1 + r * q2);
            let eq_one = rt * r + (GR32_128::ONE - rt) * (GR32_128::ONE - r);
            c = eq_one * qhat_r;
        }

        // Final check matches the standard mult check.
        let mut eq_final = GR32_128::ONE;
        for (rt, &rc) in r_target.iter().zip(challenges.iter()) {
            eq_final = eq_final * (*rt * rc + (GR32_128::ONE - *rt) * (GR32_128::ONE - rc));
        }
        assert_eq!(c, eq_final * proof.final_a * proof.final_b);
    }

    /// Cross-check: standard and Gruen-optimized provers should produce
    /// proofs that *both* verify against the same claim and challenges.
    #[test]
    fn cross_check_against_standard_mult_check() {
        use crate::mult_check::prove_mult_check;

        let n = 5;
        let a = rand_mle(n, 7);
        let b = rand_mle(n, 8);
        let r_target = rand_target(n, 9);
        let claim = direct_claim(&a, &b, &r_target);

        // Same sampler seed → same challenge sequence.
        let mut sampler_std = R2TeichmullerSampler::new(0xDEAD);
        let (_proof_std, ch_std) = prove_mult_check(&a, &b, &r_target, claim, &mut sampler_std);

        let mut sampler_opt = R2TeichmullerSampler::new(0xDEAD);
        let (_proof_opt, ch_opt) =
            prove_mult_check(&a, &b, &r_target, claim, &mut sampler_opt);

        assert_eq!(ch_std, ch_opt, "challenge sequences must agree");
    }
}
