//! Field SNARK prover. Mirror of `snark_ring::simd`, minus the φ-packing
//! machinery (no reduction, no `q1/q2`, no `v_α/v_β`): one witness commitment,
//! the same two sumchecks, one PCS open.

use field::{AbstractExtensionField, AbstractField, BB, BB4};
use pcs_field::FieldPcs;
use piop_field::{
    build_eq_table, prove_linear_check_sumcheck_only, prove_mult_check, LinearCheckProof, Mle,
    MultCheckProof,
};
use rayon::prelude::*;

use crate::instance::{matvec_bb, FieldR1csInstance, SparseMatrixBB};
use crate::proof::FieldSnarkProof;
use crate::transcript::Transcript;

#[inline]
fn lift(b: BB) -> BB4 {
    <BB4 as AbstractExtensionField<BB>>::from_base(b)
}

/// Public parameters: dimensions + the witness PCS (sized for length `m`).
pub struct FieldSnarkParams {
    pub log_n: usize,
    pub log_m: usize,
    pub pcs_w: FieldPcs,
}

impl FieldSnarkParams {
    pub fn n(&self) -> usize {
        1usize << self.log_n
    }
    pub fn m(&self) -> usize {
        1usize << self.log_m
    }

    /// **Secure default** params at the 100-bit operating point — the field
    /// mirror of `SimdSnarkParams::secure`. Same `(δ, λ, q)` as the ring (the
    /// field code shares the ring's distance by the distance-preserving lift),
    /// so the comparison stays matched. Real proofs use this; raw construction
    /// with a small explicit `ell` is for tests/benchmarks only.
    pub fn secure(log_n: usize, log_m: usize, seed: u64) -> Self {
        Self {
            log_n,
            log_m,
            pcs_w: FieldPcs::setup(
                log_m,
                pcs_field::SECURE_Q,
                pcs_field::SECURE_DELTA,
                pcs_field::SECURE_LAMBDA,
                seed,
            ),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FieldProveTiming {
    pub commit_w_ms: f64,
    /// commit split: RAA-encode vs Merkle (leaf-hash + tree).
    pub commit_w_encode_ms: f64,
    pub commit_w_merkle_ms: f64,
    pub matvec_abc_ms: f64,
    pub eq_r_ms: f64,
    pub mult_sumcheck_ms: f64,
    pub linear_arow_ms: f64,
    pub linear_sumcheck_ms: f64,
    pub open_w_ms: f64,
    pub total_ms: f64,
}

pub fn prove_field(inst: &FieldR1csInstance, params: &FieldSnarkParams) -> FieldSnarkProof {
    prove_field_with_timing(inst, params).0
}

pub fn prove_field_with_timing(
    inst: &FieldR1csInstance,
    params: &FieldSnarkParams,
) -> (FieldSnarkProof, FieldProveTiming) {
    assert_eq!(inst.a.rows, params.n());
    assert_eq!(inst.a.cols, params.m());
    assert_eq!(params.pcs_w.k1 * params.pcs_w.k2, params.m());

    let log_n = params.log_n;
    let log_m = params.log_m;
    let m = params.m();

    let mut timing = FieldProveTiming::default();
    let total_t = std::time::Instant::now();
    let mut t;

    // ----- 1. Commit the witness w (BB). One commitment — no q1/q2. -----
    let (comm_w, w_enc, w_merk) = params.pcs_w.commit_with_timing(&inst.w);
    timing.commit_w_encode_ms = w_enc;
    timing.commit_w_merkle_ms = w_merk;
    timing.commit_w_ms = w_enc + w_merk;

    // ----- 2. Transcript, mult-check target r ∈ BB4^{log n}. -----
    let mut transcript = Transcript::new(b"field");
    transcript.absorb_root(&comm_w.root);
    let r_target = transcript.challenge_target(b"mult_check_target", log_n);

    // ----- 3. a = A·w, b = B·w, c = C·w over BB; lift to BB4 MLEs. -----
    t = std::time::Instant::now();
    let aw = matvec_bb(&inst.a, &inst.w);
    let bw = matvec_bb(&inst.b, &inst.w);
    let cw = matvec_bb(&inst.c, &inst.w);
    // Lift BB → BB4 element-wise (independent → parallel map, order-preserving).
    let a_mle = Mle::new(aw.par_iter().map(|&x| lift(x)).collect());
    let b_mle = Mle::new(bw.par_iter().map(|&x| lift(x)).collect());
    timing.matvec_abc_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 4. claim_mult = c(r) = ⟨eq(r), c⟩ (exact over a field). -----
    t = std::time::Instant::now();
    let eq_r = build_eq_table(&r_target);
    // ⟨eq(r), lift(c)⟩ — associative+commutative field add → parallel reduce.
    let cw_at_r = eq_r
        .evals()
        .par_iter()
        .zip(cw.par_iter())
        .map(|(&e, &ci)| e * lift(ci))
        .reduce(BB4::zero, |u, v| u + v);
    timing.eq_r_ms = t.elapsed().as_secs_f64() * 1e3;
    let claim_mult = cw_at_r;
    transcript.absorb_bb4_slice(b"claim_mult", &[claim_mult]);

    // ----- 5. Mult-check sumcheck (deg-3). Per-round Fiat–Shamir. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"mult_check_sumcheck");
    let (mult_proof, s_1) = prove_mult_check(&a_mle, &b_mle, &r_target, claim_mult, &mut transcript);
    timing.mult_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(s_1.len(), log_n);
    absorb_mult_finals(&mut transcript, &mult_proof);

    // ----- 6. γ; M_γ_row = A_row(s1) + γ·B_row(s1) + γ²·C_row(r). -----
    let gamma = transcript.challenge_bb4(b"linear_check_gamma");
    let gamma2 = gamma * gamma;
    t = std::time::Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    let m_gamma_row = build_combined_row(
        &inst.a,
        &inst.b,
        &inst.c,
        eq_s1.evals(),
        eq_r.evals(),
        gamma,
        gamma2,
        m,
    );
    timing.linear_arow_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 7. claim_lin = a(s1) + γ·b(s1) + γ²·c(r). -----
    let claim_lin = mult_proof.final_a + gamma * mult_proof.final_b + gamma2 * cw_at_r;
    transcript.absorb_bb4_slice(b"claim_lin", &[claim_lin]);

    // ----- 8. Linear-check sumcheck (deg-2) over the combined row and w.
    //          Per-round Fiat–Shamir. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"linear_check_sumcheck");
    let w_mle = Mle::new(inst.w.iter().map(|&x| lift(x)).collect());
    let (lin_proof, s_2) =
        prove_linear_check_sumcheck_only(m_gamma_row, w_mle, claim_lin, &mut transcript);
    timing.linear_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(s_2.len(), log_m);
    absorb_lin_finals(&mut transcript, &lin_proof);

    // ----- 9. PCS open w at s_2. -----
    t = std::time::Instant::now();
    let (_y_w, open_w) = params.pcs_w.open(&comm_w, &s_2);
    timing.open_w_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(_y_w, lin_proof.final_w);

    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;

    let proof = FieldSnarkProof {
        root_w: comm_w.root,
        mult_proof,
        lin_proof,
        open_w,
    };
    (proof, timing)
}

/// `M_γ_row[h] = A_row(s1)[h] + γ·B_row(s1)[h] + γ²·C_row(r)[h]`, length `m`.
pub(crate) fn build_combined_row(
    a: &SparseMatrixBB,
    b: &SparseMatrixBB,
    c: &SparseMatrixBB,
    eq_s1: &[BB4],
    eq_r: &[BB4],
    gamma: BB4,
    gamma2: BB4,
    m: usize,
) -> Mle {
    // A, B, C row accumulations are independent → run the three sparse matvecs
    // concurrently (each matvec stays serial; per-row CSR is a deferred follow-up).
    let (mut out, (bb, cc)) = rayon::join(
        || accumulate_row(a, eq_s1, m),
        || {
            rayon::join(
                || accumulate_row(b, eq_s1, m),
                || accumulate_row(c, eq_r, m),
            )
        },
    );
    // Combine: disjoint output index `h` → parallel (bit-identical to serial).
    out.par_iter_mut().enumerate().for_each(|(h, o)| {
        *o = *o + gamma * bb[h] + gamma2 * cc[h];
    });
    Mle::new(out)
}

/// `out[col] = Σ_{(row,col,v) ∈ mat} eq[row]·lift(v)`. One sparse matvec used
/// by [`build_combined_row`].
fn accumulate_row(mat: &SparseMatrixBB, eq: &[BB4], m: usize) -> Vec<BB4> {
    let mut v = vec![BB4::zero(); m];
    for &(row, col, val) in &mat.triples {
        v[col] += eq[row] * lift(val);
    }
    v
}

// Bind the sumcheck finals. Round messages are bound per-round via
// `Challenger::observe` inside the sumcheck; the claim is bound by the
// `claim_mult` / `claim_lin` absorb before each sumcheck.
pub(crate) fn absorb_mult_finals(t: &mut Transcript, proof: &MultCheckProof) {
    t.absorb_bb4_slice(b"mult/finals", &[proof.final_a, proof.final_b]);
}

pub(crate) fn absorb_lin_finals(t: &mut Transcript, proof: &LinearCheckProof) {
    t.absorb_bb4_slice(b"lin/finals", &[proof.final_a_row, proof.final_w]);
}
