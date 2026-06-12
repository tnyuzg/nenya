//! Soundness sanity tests: tamper with one round message and check the
//! verifier rejects with `RoundCheckFailed`.
//!
//! The random-generation helpers are generic over the base-limb width
//! `L: Limb`, so each tamper scenario is exercised at u8/u16/u32/u64 via the
//! `width_tests!` macro at the bottom of this file (canonical idiom from
//! `crates/reduction/tests/implementation.rs`).

use galois_ring::{Gr1, Gr128, Gr4, Limb, R2Sampler, GR32_128};
use piop_ring::{
    build_eq_table, prove_linear_check_simd_sumcheck_only, prove_mult_check,
    verify_linear_check_simd, verify_mult_check, Mle, R2TeichmullerSampler, VerifyError,
};

// ---------------------------------------------------------------------------
// Generic random-generation helpers (LCG-seeded, width-agnostic).
//
// The LCG runs over `u64` and the low `L`-many bits of each draw are taken via
// `L::from_u64` — so the *same* seed produces width-consistent (but not
// bit-identical across widths) inputs. This keeps the helpers zero-`as`-cast.
// ---------------------------------------------------------------------------

fn rand_gr128<L: Limb>(state: &mut u64) -> Gr128<L> {
    let mut next = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        L::from_u64(*state >> 32)
    };
    let mut arr = [Gr4::<L>::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = Gr4([next(), next(), next(), next()]);
    }
    Gr128(arr)
}

fn rand_mle<L: Limb>(num_vars: usize, state: &mut u64) -> Mle<L> {
    let len = 1usize << num_vars;
    let mut evals = Vec::with_capacity(len);
    for _ in 0..len {
        evals.push(rand_gr128::<L>(state));
    }
    Mle::new(evals)
}

fn rand_target<L: Limb>(num_vars: usize, state: &mut u64) -> Vec<Gr128<L>> {
    (0..num_vars).map(|_| rand_gr128::<L>(state)).collect()
}

fn rand_sparse<L: Limb>(
    log_rows: usize,
    log_cols: usize,
    nnz_per_row: usize,
    state: &mut u64,
) -> Vec<(usize, usize, Gr1<L>)> {
    let mut next = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 32
    };
    let ncols = 1usize << log_cols;
    let nrows = 1usize << log_rows;
    let mut entries = Vec::with_capacity(nrows * nnz_per_row);
    for r in 0..nrows {
        for _ in 0..nnz_per_row {
            let c = (next() as usize) % ncols;
            let v = Gr1(L::from_u64(next()));
            entries.push((r, c, v));
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// Generic tamper-check cores.
// ---------------------------------------------------------------------------

/// Tamper with round-0 of the Gruen-optimized mult-check and expect rejection.
fn check_mult_check_optimized_tampered<L: Limb>(seed: u64, sampler_seed: u64) {
    let n = 6;
    let mut state = seed;
    let a = rand_mle::<L>(n, &mut state);
    let b = rand_mle::<L>(n, &mut state);
    let r_target = rand_target::<L>(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut claim = Gr128::<L>::ZERO;
    for i in 0..a.len() {
        claim = claim + eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    let mut sampler = R2Sampler::<L>::new(sampler_seed);
    let (mut proof, _) = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);

    // Flip q0 of round 0: the weighted round check
    //   q0 + r_target[0] · (q1 + q2) == c_0
    // now fails on the q0 side.
    proof.round_polys[0][0] = proof.round_polys[0][0] + Gr128::<L>::ONE;

    let mut verifier_sampler = R2Sampler::<L>::new(sampler_seed);
    let err = verify_mult_check(&proof, &r_target, &mut verifier_sampler)
        .expect_err("tampered mult-check (optimized) proof must be rejected");
    assert_eq!(err, VerifyError::RoundCheckFailed { round: 0 });
}

/// Tamper with round-0 of the SIMD linear-check and expect rejection.
///
/// The public `prove_linear_check_simd` is pinned to the `u32` matrix API, so
/// the generic path rebuilds its two preprocessing steps by hand: the R₂ eq
/// table over `r_target`, then the R₀-aware matvec `A_row[c] = Σ_y eq[y]·A[y,c]`
/// via `mul_by_gr1` — exactly what `build_a_row` does at `u32` — before
/// handing off to the width-generic `prove_linear_check_simd_sumcheck_only`.
fn check_linear_check_tampered<L: Limb>(seed: u64, sampler_seed: u64) {
    let n = 6;
    let mut state = seed;
    let entries = rand_sparse::<L>(n, n, 3, &mut state);
    let w = rand_mle::<L>(n, &mut state);
    let r_target = rand_target::<L>(n, &mut state);

    // Step 1+2: eq table over r_target, then the R₀-aware matvec into A_row.
    let eq = build_eq_table(&r_target);
    let num_cols = 1usize << n;
    let mut a_row_evals = vec![Gr128::<L>::ZERO; num_cols];
    for &(row, col, val) in &entries {
        a_row_evals[col] = a_row_evals[col] + eq.evals()[row].mul_by_gr1(val);
    }
    let a_row = Mle::new(a_row_evals);

    let mut claim = Gr128::<L>::ZERO;
    for i in 0..w.len() {
        claim = claim + a_row.evals()[i] * w.evals()[i];
    }

    let mut sampler = R2Sampler::<L>::new(sampler_seed);
    let (mut proof, _) =
        prove_linear_check_simd_sumcheck_only(a_row, w.clone(), claim, &mut sampler);

    // Flip the round-0 quadratic coefficient.
    proof.round_polys[0][2] = proof.round_polys[0][2] + Gr128::<L>::ONE;

    let mut verifier_sampler = R2Sampler::<L>::new(sampler_seed);
    let err = verify_linear_check_simd(&proof, &mut verifier_sampler)
        .expect_err("tampered linear-check proof must be rejected");
    assert_eq!(err, VerifyError::RoundCheckFailed { round: 0 });
}

// ---------------------------------------------------------------------------
// u32-only tamper check: the naive (non-Gruen) mult-check prover/verifier pair
// is pinned to the concrete `GR32_128` API, so this scenario stays at `u32`.
// The width-generic mult-check coverage lives in `check_mult_check_optimized_tampered`.
// ---------------------------------------------------------------------------

#[test]
fn tampered_mult_check_first_round_rejected() {
    let n = 6;
    let mut state = 0xC0FFEE_u64;
    let a = rand_mle::<u32>(n, &mut state);
    let b = rand_mle::<u32>(n, &mut state);
    let r_target = rand_target::<u32>(n, &mut state);

    let eq = build_eq_table(&r_target);
    let mut claim = GR32_128::ZERO;
    for i in 0..a.len() {
        claim = claim + eq.evals()[i] * a.evals()[i] * b.evals()[i];
    }

    let mut sampler = R2TeichmullerSampler::new(7);
    let (mut proof, _) = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);

    // Flip the round-0 constant term so s_0(0) + s_0(1) ≠ claimed_sum.
    proof.round_polys[0][0] = proof.round_polys[0][0] + GR32_128::ONE;

    let mut verifier_sampler = R2TeichmullerSampler::new(7);
    let err = verify_mult_check(&proof, &r_target, &mut verifier_sampler)
        .expect_err("tampered mult-check proof must be rejected");
    assert_eq!(err, VerifyError::RoundCheckFailed { round: 0 });
}

// ---------------------------------------------------------------------------
// Per-width instantiations of the two generic tamper checks.
// ---------------------------------------------------------------------------

macro_rules! width_tests {
    ($modname:ident, $t:ty) => {
        mod $modname {
            use super::*;

            #[test]
            fn tampered_mult_check_optimized_first_round_rejected() {
                // Distinct seeds per width keep the scenarios independent.
                check_mult_check_optimized_tampered::<$t>(0xBAD5EED, 17);
            }
            #[test]
            fn tampered_linear_check_first_round_rejected() {
                check_linear_check_tampered::<$t>(0xBADBEEF, 13);
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
