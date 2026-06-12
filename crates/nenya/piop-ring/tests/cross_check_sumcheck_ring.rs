//! Cross-checks between `sumcheck-ring`'s reference provers and
//! `piop-ring`'s specialized provers. Both pairs must emit bit-identical
//! round messages and bind to the same final evaluations for identical
//! inputs + identically seeded samplers.
//!
//! - **Deg-3.** `sumcheck_ring::prove(f1=eq, f2=a, f3=b)` vs.
//!   `piop_ring::prove_mult_check(a, b, r_target)`. The reference folds the
//!   full `eq(r_target, ·)` table, so its round message is the complete deg-3
//!   `s_i`; the Gruen-optimized mult_check sends only the deg-2 inner factor
//!   `Q̃_i`, with `s_i(Z) = eq_one(r_target_i, Z)·Q̃_i(Z)`. We reconstruct `s_i`
//!   from `Q̃_i` and check it matches the reference round-by-round.
//! - **Deg-2.** `sumcheck_ring::prove_deg2(A_row, w)` vs.
//!   `piop_ring::prove_linear_check_simd(matrix, w, r_target)`. sumcheck-ring
//!   uses direct coefficient form (4 muls/pair); linear_check uses
//!   `{0, 1, ∞}` form and recovers `a1` from the running claim
//!   (2 muls/pair). They still produce the same wire-level
//!   `[a0, a1, a2]`.

use galois_ring::{R2TeichmullerSampler, GR32_1, GR32_128, GR32_4};
use piop_ring::{build_eq_table, prove_linear_check_simd, prove_mult_check, Mle, SparseMatrix};
use sumcheck_ring::{
    prove as sumcheck_ring_prove, prove_deg2 as sumcheck_ring_prove_deg2, Mle as SrMle,
};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn rand_gr128(state: &mut u64) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([lcg(state), lcg(state), lcg(state), lcg(state)]);
    }
    GR32_128(arr)
}

fn rand_evals(num_vars: usize, state: &mut u64) -> Vec<GR32_128> {
    (0..(1usize << num_vars))
        .map(|_| rand_gr128(state))
        .collect()
}

fn rand_target(num_vars: usize, state: &mut u64) -> Vec<GR32_128> {
    (0..num_vars).map(|_| rand_gr128(state)).collect()
}

fn triple_claim(f1: &[GR32_128], f2: &[GR32_128], f3: &[GR32_128]) -> GR32_128 {
    let mut acc = GR32_128::ZERO;
    for i in 0..f1.len() {
        acc = acc + f1[i] * f2[i] * f3[i];
    }
    acc
}

// ---------------------------------------------------------------------
// Deg-3: sumcheck-ring reference vs piop-ring mult_check.
// ---------------------------------------------------------------------

fn run_deg3_at(num_vars: usize, seed: u64) {
    let mut s = seed | 1;

    let a_evals = rand_evals(num_vars, &mut s);
    let b_evals = rand_evals(num_vars, &mut s);
    let r_target = rand_target(num_vars, &mut s);

    // mult_check uses eq(r_target, ·) as its implicit third factor.
    let eq = build_eq_table(&r_target);
    let c = triple_claim(eq.evals(), &a_evals, &b_evals);

    // sumcheck-ring: ∑ f1·f2·f3 with f1 = eq, f2 = a, f3 = b.
    let f1_sr = SrMle::new(eq.evals().to_vec());
    let f2_sr = SrMle::new(a_evals.clone());
    let f3_sr = SrMle::new(b_evals.clone());
    let mut sampler_sr = R2TeichmullerSampler::new(seed);
    let (sr_proof, sr_challenges) = sumcheck_ring_prove(&f1_sr, &f2_sr, &f3_sr, c, &mut sampler_sr);

    // piop-ring mult_check: same claim, Gruen-optimized prover (its round
    // message is the deg-2 inner factor Q̃_i, not the full deg-3 s_i).
    let a = Mle::new(a_evals);
    let b = Mle::new(b_evals);
    let mut sampler_mc = R2TeichmullerSampler::new(seed);
    let (mc_proof, mc_challenges) = prove_mult_check(&a, &b, &r_target, c, &mut sampler_mc);

    assert_eq!(sr_challenges, mc_challenges, "challenge sequences diverged");
    assert_eq!(
        sr_proof.round_polys.len(),
        mc_proof.round_polys.len(),
        "round count mismatch"
    );
    // Reconstruct the reference's deg-3 round message from the optimized
    // deg-2 inner factor via the Gruen identity s_i = eq_one(r_i,·)·Q̃_i, where
    // eq_one(r_i, Z) = (1 − r_i) + (2·r_i − 1)·Z = e0 + e1·Z.
    for (i, (sr_poly, q)) in sr_proof
        .round_polys
        .iter()
        .zip(mc_proof.round_polys.iter())
        .enumerate()
    {
        let e0 = GR32_128::ONE - r_target[i];
        let e1 = r_target[i] - e0; // = 2·r_target_i − 1
        let reconstructed = [
            e0 * q[0],
            e0 * q[1] + e1 * q[0],
            e0 * q[2] + e1 * q[1],
            e1 * q[2],
        ];
        assert_eq!(
            sr_poly, &reconstructed,
            "round {i}: reference s_i ≠ eq_one(r_i,·)·Q̃_i \
             (sumcheck-ring {sr_poly:?} vs reconstructed {reconstructed:?})"
        );
    }

    // sumcheck-ring binds final_f1 = eq(r), final_f2 = a(r), final_f3 = b(r).
    // mult_check does not send final_eq (the verifier recomputes it in O(n)),
    // so we check against a(r) and b(r).
    assert_eq!(sr_proof.final_f2, mc_proof.final_a, "final a mismatch");
    assert_eq!(sr_proof.final_f3, mc_proof.final_b, "final b mismatch");
}

#[test]
fn deg3_cross_check_4_vars() {
    run_deg3_at(4, 0xC0DE_C0DE);
}

#[test]
fn deg3_cross_check_8_vars() {
    run_deg3_at(8, 0xBEEF_FACE);
}

#[test]
fn deg3_cross_check_10_vars() {
    run_deg3_at(10, 0x1234_5678);
}

// ---------------------------------------------------------------------
// Deg-2: sumcheck-ring reference vs piop-ring linear_check.
// ---------------------------------------------------------------------

fn rand_sparse(
    log_rows: usize,
    log_cols: usize,
    nnz_per_row: usize,
    state: &mut u64,
) -> SparseMatrix {
    let nrows = 1usize << log_rows;
    let ncols = 1usize << log_cols;
    let mut m = SparseMatrix::new(log_rows, log_cols);
    for r in 0..nrows {
        for _ in 0..nnz_per_row {
            let c = (lcg(state) as usize) % ncols;
            let v = GR32_1(lcg(state));
            m.push(r, c, v);
        }
    }
    m
}

/// Reproduce `linear_check::build_a_row` externally (it's `pub(crate)`).
fn build_a_row(matrix: &SparseMatrix, eq_row: &[GR32_128]) -> Vec<GR32_128> {
    let mut a_row = vec![GR32_128::ZERO; matrix.num_cols()];
    for &(row, col, val) in &matrix.entries {
        a_row[col] = a_row[col] + eq_row[row].mul_by_gr1(val);
    }
    a_row
}

fn run_deg2_at(log_rows: usize, log_cols: usize, nnz_per_row: usize, seed: u64) {
    let mut s = seed | 1;

    let matrix = rand_sparse(log_rows, log_cols, nnz_per_row, &mut s);
    let r_target = rand_target(log_rows, &mut s);
    let w_evals = rand_evals(log_cols, &mut s);

    let eq = build_eq_table(&r_target);
    let a_row_evals = build_a_row(&matrix, eq.evals());
    let mut c = GR32_128::ZERO;
    for i in 0..a_row_evals.len() {
        c = c + a_row_evals[i] * w_evals[i];
    }

    // sumcheck-ring deg-2 reference on (A_row, w).
    let f1_sr = SrMle::new(a_row_evals.clone());
    let f2_sr = SrMle::new(w_evals.clone());
    let mut sampler_sr = R2TeichmullerSampler::new(seed);
    let (sr_proof, sr_challenges) = sumcheck_ring_prove_deg2(&f1_sr, &f2_sr, c, &mut sampler_sr);

    // piop-ring linear_check specialized prover.
    let w = Mle::new(w_evals);
    let mut sampler_lc = R2TeichmullerSampler::new(seed);
    let (lc_proof, lc_challenges) =
        prove_linear_check_simd(&matrix, &w, &r_target, c, &mut sampler_lc);

    assert_eq!(sr_challenges, lc_challenges, "challenge sequences diverged");
    assert_eq!(
        sr_proof.round_polys.len(),
        lc_proof.round_polys.len(),
        "round count mismatch"
    );
    for (i, (sr_poly, lc_poly)) in sr_proof
        .round_polys
        .iter()
        .zip(lc_proof.round_polys.iter())
        .enumerate()
    {
        assert_eq!(
            sr_poly, lc_poly,
            "round {i} message mismatch: sumcheck-ring {sr_poly:?} vs linear_check {lc_poly:?}"
        );
    }
    assert_eq!(
        sr_proof.final_f1, lc_proof.final_a_row,
        "final A_row mismatch"
    );
    assert_eq!(sr_proof.final_f2, lc_proof.final_w, "final w mismatch");
}

#[test]
fn deg2_cross_check_4x4() {
    run_deg2_at(2, 4, 3, 0xC0DE_C0DE);
}

#[test]
fn deg2_cross_check_8x8() {
    run_deg2_at(3, 8, 4, 0xBEEF_FACE);
}

#[test]
fn deg2_cross_check_10x10() {
    run_deg2_at(4, 10, 5, 0x1234_5678);
}
