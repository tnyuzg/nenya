//! End-to-end zkSNARK over `R_2 = GR(2^n, 128)` for **SIMD** R1CS, generic over
//! the base-limb width `L` (`u8`/`u16`/`u32`/`u64`).
//!
//! Pipeline (paper §4.1, SIMD path):
//!
//! 1. `SimdR1csInstance<L>` ── [`reduction::simd_to_packed`] ──▶
//!    `PackedSimdR1csInstance<L>` over R_2 with kernel quotients `q1, q2`.
//! 2. Commit:
//!    - [`pcs_ring::Pcs`] commits `w_phi` (length `m`, in `Im(φ)`).
//!    - [`crate::r2_pcs::R2Pcs`] commits the combined error `ê`
//!      (length `n`, generic R_2).
//! 3. Mult check at random target `r ∈ T_2^{log n}` — the **Gruen-optimized**
//!    deg-3 kernel ([`piop_ring::prove_mult_check`], 6 muls/pair):
//!    ```text
//!    ∑_h eq(r,h) · (A·w_phi)(h) · (B·w_phi)(h)
//!      = c_phi(r) + q1(r)·v_α + q2(r)·v_β
//!    ```
//!    Reduces to `final_a = a_phi(s_1)`, `final_b = b_phi(s_1)`.
//! 4. Linear check batched with `γ` — the **`{0,1,∞}`-form** deg-2 kernel
//!    ([`piop_ring::prove_linear_check_simd_sumcheck_only`], 2 muls/pair):
//!    ```text
//!    final_a + γ·final_b + γ²·c_phi(r)
//!      = ∑_h (A_row(s_1,h) + γ·B_row(s_1,h) + γ²·C_row(r,h)) · w_phi(h)
//!    ```
//!    Reduces to `final_w = w_phi(s_2)` and a verifier-reconstructible
//!    `final_a_row`.
//! 5. PCS opens at `s_2` (for `w_phi`) and `r` (for `ê`).
//!
//! ## Sumcheck kernels (the optimized PIOP path)
//!
//! Both sumchecks use the optimized `piop-ring` provers:
//!
//! - **Mult check** — Gruen's eq-MLE specialization
//!   ([`piop_ring::prove_mult_check`]). The round message of the
//!   degree-3 sum `∑_h eq(r,h)·a(h)·b(h)` factors as
//!   `s_i(Z) = eq_one(r_i, Z) · Q̃_i(Z)` with `Q̃_i` degree 2; the prover sends
//!   only `Q̃_i` (3 coeffs) and maintains a slim half-size `eq` table. That is
//!   6 muls/pair (vs 10 for the naive 3-table form) plus a 2×-cheaper eq update.
//!   `eq(r,·)` is *not* a committed factor here — it is reconstructed by the
//!   verifier — so the prover passes only `a_φ, b_φ`.
//!
//! - **Linear check** — `{0, 1, ∞}` evaluation-basis form
//!   ([`piop_ring::prove_linear_check_simd_sumcheck_only`]). The degree-2 round
//!   message of `∑_h M_γ_row(h)·w_φ(h)` is built from `s(0)` and `s(∞)`, with
//!   `s(1)` recovered from the running claim (`a1 = c_prev − 2·s(0) − s(∞)`),
//!   so it costs 2 muls/pair instead of the reference 4. The matvec that builds
//!   `M_γ_row` uses `mul_by_gr1` (R_0 factor lifted to R_2).
//!
//! ## Fiat–Shamir
//!
//! **Per-round FS.** Both sumchecks drive the [`Transcript`] (a
//! [`piop_ring::Challenger`]) directly: each round `observe`s its message and
//! then `sample`s that round's challenge, so every challenge is bound to all
//! prior round messages. The orchestrator absorbs the inter-check glue
//! (commitment roots, claims, final evaluations, the `γ` batching point).
//!
//! ## Well-formedness (both sides enforced)
//!
//! - **`w_phi ∈ Im(φ)`** — by construction: committed in 32-limb packed form,
//!   so the pre-image is a packed value.
//! - **`ê ∈ ker(ψ)`** — the free-subspace check
//!   ([`crate::r2_pcs::columns_in_kernel`]) on the opened error columns. This is
//!   what ties the mult-check back to the R1CS relation: a dishonest
//!   `ê = â∘b̂ − ĉ` with `a∘b ≠ c` has `ψ(ê) = a∘b − c ≠ 0` and is rejected.
//!   Pure verifier work — no prover or proof-size cost.

use galois_ring::{Gr1, Gr128, Limb};
use piop_ring::{
    build_eq_table, prove_linear_check_simd_sumcheck_only, prove_mult_check,
    verify_linear_check_simd, verify_mult_check, LinearCheckSimdProof, Mle, MultCheckProof,
};
use rayon::prelude::*;
use reduction::constants::HasPackingConstants;
use reduction::matrix::{matvec_gr128_csr, CsrMatrix, SparseMatrix};
use reduction::{simd_to_packed, SimdR1csInstance};

use pcs_ring::{Hash, OpeningProof, Pcs, SECURE_DELTA, SECURE_LAMBDA, SECURE_Q};

use crate::error::VerifyError;
use crate::r2_pcs::{columns_in_kernel, R2OpeningProof, R2Pcs};
use crate::transcript::Transcript;

// ---------------------------------------------------------------------------
// Public params / proof / timing
// ---------------------------------------------------------------------------

/// Public parameters: matrix dimensions plus the two PCS instances. Both
/// PCSs must have `k1·k2` matching the polynomial sizes — `pcs_w` for
/// length `m = 2^log_m`, `pcs_q` for length `n = 2^log_n`.
pub struct SimdSnarkParams<L: Limb = u32> {
    pub log_n: usize,
    pub log_m: usize,
    pub pcs_w: Pcs<L>,
    pub pcs_q: R2Pcs<L>,
}

impl<L: Limb> SimdSnarkParams<L> {
    pub fn n(&self) -> usize {
        1usize << self.log_n
    }
    pub fn m(&self) -> usize {
        1usize << self.log_m
    }
}

impl<L: HasPackingConstants> SimdSnarkParams<L> {
    /// **Secure default** params at the 100-bit operating point: query budget and
    /// proof-size-optimal matrix shape from `*::setup` (δ = [`SECURE_DELTA`],
    /// λ = [`SECURE_LAMBDA`], q = [`SECURE_Q`]). Use this for any real proof.
    /// Constructing the struct directly with a small explicit `ell` (via
    /// `Pcs::new`/`R2Pcs::new`) is insecure — for unit tests / benchmarks only.
    pub fn secure(log_n: usize, log_m: usize, seed_w: u64, seed_q: u64) -> Self {
        Self {
            log_n,
            log_m,
            pcs_w: Pcs::setup(log_m, SECURE_Q, SECURE_DELTA, SECURE_LAMBDA, seed_w),
            pcs_q: R2Pcs::setup(log_n, SECURE_Q, SECURE_DELTA, SECURE_LAMBDA, seed_q),
        }
    }
}

/// Full SNARK proof for a satisfied SIMD R1CS instance.
#[derive(PartialEq)]
pub struct SimdSnarkProof<L: Limb = u32> {
    /// Commitments.
    pub root_w: Hash,
    /// Single commitment to the combined kernel error `ê = q1·v_α + q2·v_β`
    /// (replaces the separate `q1,q2` commitments — one PCS, one query).
    pub root_e: Hash,

    /// Mult-check proof (Gruen-optimized deg-3) for
    /// `claimed_sum = ∑_h eq(r,h)·a_phi(h)·b_phi(h)`.
    /// Round messages are the deg-2 inner factor `Q̃_i`; `final_a = a_phi(s_1)`,
    /// `final_b = b_phi(s_1)`. The `eq(r,·)` factor is reconstructed by the
    /// verifier, so it is not carried in the proof.
    pub mult_proof: MultCheckProof<L>,

    /// Linear-check proof (deg-2, `{0,1,∞}` form) for
    /// `claim_lin = ∑_h M_γ_row(h) · w_phi(h)`.
    /// `final_a_row = M_γ_row(s_2)` (verifier-recomputable),
    /// `final_w = w_phi(s_2)`.
    pub lin_proof: LinearCheckSimdProof<L>,

    /// PCS opening of `w_phi` at `s_2`.
    pub open_w: OpeningProof<L>,
    /// PCS opening of `ê` at `r` (the mult-check target).
    pub open_e: R2OpeningProof<L>,
}

impl<L: Limb> SimdSnarkProof<L> {
    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Commitments
        buf.extend_from_slice(&self.root_w);
        buf.extend_from_slice(&self.root_e);

        // Proofs
        buf.extend_from_slice(&self.mult_proof.serialize());
        buf.extend_from_slice(&self.lin_proof.serialize());

        // PCS openings
        buf.extend_from_slice(&self.open_w.serialize());
        buf.extend_from_slice(&self.open_e.serialize());

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Per-stage prover timing in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct SimdProveTiming {
    pub reduce_ms: f64,
    pub commit_w_encode_ms: f64,
    pub commit_w_merkle_ms: f64,
    pub commit_e_encode_ms: f64,
    pub commit_e_merkle_ms: f64,
    pub matvec_abc_ms: f64,
    pub eq_r_ms: f64,
    pub mult_sumcheck_ms: f64,
    pub linear_arow_ms: f64,
    pub linear_sumcheck_ms: f64,
    pub open_w_ms: f64,
    pub open_e_ms: f64,
    pub total_ms: f64,
}

/// Per-stage verifier timing in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct SimdVerifyTiming {
    pub pcs_e_ms: f64,
    /// Free-subspace (kernel) check `ψ(ê)=0` on the opened error columns.
    pub kernel_ms: f64,
    pub mult_sumcheck_ms: f64,
    pub linear_sumcheck_ms: f64,
    pub linear_arow_recompute_ms: f64,
    pub pcs_w_ms: f64,
    pub total_ms: f64,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Run the SIMD SNARK prover. Returns the proof.
pub fn prove_simd<L: HasPackingConstants>(
    simd: &SimdR1csInstance<L>,
    params: &SimdSnarkParams<L>,
) -> SimdSnarkProof<L> {
    prove_simd_with_timing(simd, params).0
}

/// Same as [`prove_simd`] but also returns per-stage timing.
pub fn prove_simd_with_timing<L: HasPackingConstants>(
    simd: &SimdR1csInstance<L>,
    params: &SimdSnarkParams<L>,
) -> (SimdSnarkProof<L>, SimdProveTiming) {
    assert_eq!(simd.a.rows, params.n());
    assert_eq!(simd.a.cols, params.m());
    assert_eq!(simd.b.rows, params.n());
    assert_eq!(simd.b.cols, params.m());
    assert_eq!(simd.c.rows, params.n());
    assert_eq!(simd.c.cols, params.m());
    assert_eq!(params.pcs_w.k1 * params.pcs_w.k2, params.m());
    assert_eq!(params.pcs_q.k1 * params.pcs_q.k2, params.n());

    let log_n = params.log_n;
    let log_m = params.log_m;
    let m = params.m();
    let n = params.n();

    let mut timing = SimdProveTiming::default();
    let total_t = std::time::Instant::now();
    let mut t = std::time::Instant::now();

    // ----- 1. Reduce SIMD R1CS to packed R1CS over R_2. -----
    let packed = simd_to_packed(simd);
    timing.reduce_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(packed.w_phi.len(), m);
    debug_assert_eq!(packed.e.len(), n);

    // ----- 2. Commit w_phi (Im(φ) packed) and the single kernel error ê. -----
    //  `ê = q1·v_α + q2·v_β` replaces the two `q1,q2` commitments (one PCS, one
    //  query). Commit timing is split into RAA-encode vs Merkle for each.
    let mut w_phi_packed: Vec<[L; 32]> = Vec::with_capacity(m);
    for j in 0..m {
        let mut col = [L::ZERO; 32];
        for k in 0..32 {
            col[k] = simd.witnesses[k][j].0;
        }
        w_phi_packed.push(col);
    }
    let (comm_w, w_enc, w_merk) = params.pcs_w.commit_with_timing(&w_phi_packed);
    timing.commit_w_encode_ms = w_enc;
    timing.commit_w_merkle_ms = w_merk;

    let (comm_e, e_enc, e_merk) = params.pcs_q.commit_with_timing(&packed.e);
    timing.commit_e_encode_ms = e_enc;
    timing.commit_e_merkle_ms = e_merk;

    // ----- 3. Init transcript, absorb commitments, sample mult-check target r. -----
    let mut transcript = Transcript::new(b"simd");
    transcript.absorb_root(&comm_w.root);
    transcript.absorb_root(&comm_e.root);

    let r_target: Vec<Gr128<L>> = transcript.challenge_target(b"mult_check_target", log_n);

    // ----- 4. Compute a_phi, b_phi, c_phi as length-n R_2 vectors. Row-group
    //          the (shared, R_0) matrices once, then matvec parallel over rows. -----
    t = std::time::Instant::now();
    let a_csr = CsrMatrix::from_coo(&packed.a);
    let b_csr = CsrMatrix::from_coo(&packed.b);
    let c_csr = CsrMatrix::from_coo(&packed.c);
    let aw: Vec<Gr128<L>> = matvec_gr128_csr(&a_csr, &packed.w_phi);
    let bw: Vec<Gr128<L>> = matvec_gr128_csr(&b_csr, &packed.w_phi);
    let cw: Vec<Gr128<L>> = matvec_gr128_csr(&c_csr, &packed.w_phi);
    timing.matvec_abc_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 5. claim_mult = c_phi(r) + ê(r),  ê(r) = q1(r)·v_α + q2(r)·v_β. -----
    t = std::time::Instant::now();
    let eq_r = build_eq_table(&r_target);
    let cw_at_r = inner_product(eq_r.evals(), &cw);
    let e_at_r = inner_product(eq_r.evals(), &packed.e);
    timing.eq_r_ms = t.elapsed().as_secs_f64() * 1e3;
    let claim_mult = cw_at_r + e_at_r;

    transcript.absorb_gr128_slice(b"claim_mult", &[claim_mult]);

    // ----- 6. Mult-check sumcheck (Gruen-optimized deg-3; eq is internal).
    //          Per-round Fiat–Shamir: the sumcheck drives the transcript
    //          (observe round message → sample challenge) each round. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"mult_check_sumcheck");
    let a_mle = Mle::new(aw);
    let b_mle = Mle::new(bw);
    let (mult_proof, s_1) =
        prove_mult_check(&a_mle, &b_mle, &r_target, claim_mult, &mut transcript);
    timing.mult_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(s_1.len(), log_n);

    absorb_mult_finals(&mut transcript, &mult_proof);

    // ----- 7. Sample γ. -----
    let gamma: Gr128<L> = transcript.challenge_gr128(b"linear_check_gamma");
    let gamma2 = gamma * gamma;

    // ----- 8. M_γ_row[h] = A_row(s_1)[h] + γ·B_row(s_1)[h] + γ²·C_row(r)[h]. -----
    t = std::time::Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    let m_gamma_row = build_combined_a_row(
        &simd.a,
        &simd.b,
        &simd.c,
        eq_s1.evals(),
        eq_r.evals(),
        gamma,
        gamma2,
        m,
    );
    timing.linear_arow_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 9. claim_lin = a_phi(s_1) + γ·b_phi(s_1) + γ²·c_phi(r). -----
    let final_a = mult_proof.final_a;
    let final_b = mult_proof.final_b;
    let claim_lin = final_a + gamma * final_b + gamma2 * cw_at_r;

    transcript.absorb_gr128_slice(b"claim_lin", &[claim_lin]);

    // ----- 10. Linear-check sumcheck ({0,1,∞} deg-2). Per-round FS. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"linear_check_sumcheck");
    let g1 = Mle::new(m_gamma_row);
    // Move w_phi (not clone): the matvecs ran above and the PCS open uses the
    // commitment, so packed.w_phi is unused after here — the 512 MB clone @2^25
    // (mis-attributed to the sumcheck timer) was pure waste.
    let g2 = Mle::new(packed.w_phi);
    let (lin_proof, s_2) =
        prove_linear_check_simd_sumcheck_only(g1, g2, claim_lin, &mut transcript);
    timing.linear_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(s_2.len(), log_m);

    absorb_lin_finals(&mut transcript, &lin_proof);

    // ----- 11. PCS opens. -----
    t = std::time::Instant::now();
    let (_y_w, open_w) = params.pcs_w.open(&comm_w, &s_2);
    timing.open_w_ms = t.elapsed().as_secs_f64() * 1e3;

    t = std::time::Instant::now();
    let (_y_e, open_e) = params.pcs_q.open(&comm_e, &r_target);
    timing.open_e_ms = t.elapsed().as_secs_f64() * 1e3;

    debug_assert_eq!(_y_w, lin_proof.final_w);
    debug_assert_eq!(_y_e, e_at_r);

    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;

    let proof = SimdSnarkProof {
        root_w: comm_w.root,
        root_e: comm_e.root,
        mult_proof,
        lin_proof,
        open_w,
        open_e,
    };
    (proof, timing)
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify a SIMD SNARK proof against the public R1CS matrices.
pub fn verify_simd<L: HasPackingConstants>(
    a: &SparseMatrix<Gr1<L>>,
    b: &SparseMatrix<Gr1<L>>,
    c: &SparseMatrix<Gr1<L>>,
    params: &SimdSnarkParams<L>,
    proof: &SimdSnarkProof<L>,
) -> Result<(), VerifyError> {
    verify_simd_with_timing(a, b, c, params, proof).map(|_| ())
}

/// Same as [`verify_simd`] but returns per-stage timing on success.
pub fn verify_simd_with_timing<L: HasPackingConstants>(
    a: &SparseMatrix<Gr1<L>>,
    b: &SparseMatrix<Gr1<L>>,
    c: &SparseMatrix<Gr1<L>>,
    params: &SimdSnarkParams<L>,
    proof: &SimdSnarkProof<L>,
) -> Result<SimdVerifyTiming, VerifyError> {
    let log_n = params.log_n;
    let log_m = params.log_m;
    assert_eq!(a.rows, params.n());
    assert_eq!(a.cols, params.m());
    assert_eq!(b.rows, params.n());
    assert_eq!(b.cols, params.m());
    assert_eq!(c.rows, params.n());
    assert_eq!(c.cols, params.m());

    let mut timing = SimdVerifyTiming::default();
    let total_t = std::time::Instant::now();
    let mut t;

    // ----- 1. Replay the transcript through the commitment phase. -----
    let mut transcript = Transcript::new(b"simd");
    transcript.absorb_root(&proof.root_w);
    transcript.absorb_root(&proof.root_e);

    let r_target: Vec<Gr128<L>> = transcript.challenge_target(b"mult_check_target", log_n);

    // ----- 2. PCS open of the single kernel error ê at r. -----
    let log_k1_q = params.pcs_q.k1.trailing_zeros() as usize;
    let e1_q = build_eq_table(&r_target[log_k1_q..]).evals().to_vec();
    let e_at_r = mbar_dot_e1(&proof.open_e.mbar, &e1_q);

    t = std::time::Instant::now();
    if !params
        .pcs_q
        .verify(proof.root_e, &r_target, e_at_r, &proof.open_e)
    {
        return Err(VerifyError::PcsE);
    }
    timing.pcs_e_ms = t.elapsed().as_secs_f64() * 1e3;

    // Free-subspace check: ê ∈ ker(ψ). Without it a prover with a∘b ≠ c could
    // pass by committing the non-kernel error ê = â∘b̂ − ĉ; forcing ê ∈ ker(ψ)
    // makes ψ(ê) = a∘b − c, so a∘b = c is enforced. Pure verifier work on the
    // already-opened columns (paper §5.2). The Im(φ) side (w_phi packed form)
    // is enforced by construction at the pcs_w commitment.
    t = std::time::Instant::now();
    if !columns_in_kernel(&proof.open_e) {
        return Err(VerifyError::KernelCheck);
    }
    timing.kernel_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 3. claim_mult; reconstruct c_phi(r) = claim_mult − ê(r). -----
    let claim_mult = proof.mult_proof.claimed_sum;
    let cw_at_r = claim_mult - e_at_r;

    transcript.absorb_gr128_slice(b"claim_mult", &[claim_mult]);

    // ----- 4. Mult-check (Gruen-optimized): round checks + final
    //          c_n == eq(r, s_1)·final_a·final_b are internal. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"mult_check_sumcheck");
    let s_1 = verify_mult_check(&proof.mult_proof, &r_target, &mut transcript)
        .map_err(|_| VerifyError::MultSumcheck)?;
    timing.mult_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;

    absorb_mult_finals(&mut transcript, &proof.mult_proof);

    // ----- 5. Sample γ and reconstruct claim_lin. -----
    let gamma: Gr128<L> = transcript.challenge_gr128(b"linear_check_gamma");
    let gamma2 = gamma * gamma;
    let final_a = proof.mult_proof.final_a;
    let final_b = proof.mult_proof.final_b;
    let claim_lin = final_a + gamma * final_b + gamma2 * cw_at_r;

    if claim_lin != proof.lin_proof.claimed_sum {
        return Err(VerifyError::LinClaimMismatch);
    }
    transcript.absorb_gr128_slice(b"claim_lin", &[claim_lin]);

    // ----- 6. Linear-check ({0,1,∞}): round checks + final
    //          c_n == final_a_row·final_w are internal. -----
    t = std::time::Instant::now();
    transcript.absorb_label(b"linear_check_sumcheck");
    let s_2 = verify_linear_check_simd(&proof.lin_proof, &mut transcript)
        .map_err(|_| VerifyError::LinSumcheck)?;
    timing.linear_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 7. Cross-check final_a_row against verifier-computed M_γ_row(s_2). -----
    // eq-table builds belong to the linear/piop bucket, not the matrix-eval
    // component — keep piop_mxeval = row rebuild + final dot, comparable to the
    // MIMD/general verifiers (which build eq outside their recompute timer).
    let teq = std::time::Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    let eq_r = build_eq_table(&r_target);
    let eq_s2 = build_eq_table(&s_2);
    timing.linear_sumcheck_ms += teq.elapsed().as_secs_f64() * 1e3;
    t = std::time::Instant::now();
    // Reuse the prover's parallel row build + a parallel inner product with
    // eq_s2.
    let m_gamma_row = build_combined_a_row(
        a,
        b,
        c,
        eq_s1.evals(),
        eq_r.evals(),
        gamma,
        gamma2,
        params.m(),
    );
    let m_gamma_row_at_s2 = inner_product(eq_s2.evals(), &m_gamma_row);
    if m_gamma_row_at_s2 != proof.lin_proof.final_a_row {
        return Err(VerifyError::LinARowMismatch);
    }
    timing.linear_arow_recompute_ms = t.elapsed().as_secs_f64() * 1e3;

    absorb_lin_finals(&mut transcript, &proof.lin_proof);

    // ----- 8. PCS open of w_phi at s_2 must match lin_proof.final_w. -----
    t = std::time::Instant::now();
    if !params
        .pcs_w
        .verify(proof.root_w, &s_2, proof.lin_proof.final_w, &proof.open_w)
    {
        return Err(VerifyError::PcsW);
    }
    timing.pcs_w_ms = t.elapsed().as_secs_f64() * 1e3;

    let _ = log_m;
    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    Ok(timing)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn inner_product<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> Gr128<L> {
    debug_assert_eq!(a.len(), b.len());
    // Associative+commutative ring add → parallel reduction is bit-identical.
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| x * y)
        .reduce(|| Gr128::<L>::ZERO, |u, v| u + v)
}

/// Build `M_γ_row[h] = A_row(s1)[h] + γ·B_row(s1)[h] + γ²·C_row(r)[h]`,
/// length `m`, using `mul_by_gr1` for each sparse-matrix triple.
fn build_combined_a_row<L: Limb>(
    a: &SparseMatrix<Gr1<L>>,
    b: &SparseMatrix<Gr1<L>>,
    c: &SparseMatrix<Gr1<L>>,
    eq_s1: &[Gr128<L>],
    eq_r: &[Gr128<L>],
    gamma: Gr128<L>,
    gamma2: Gr128<L>,
    m: usize,
) -> Vec<Gr128<L>> {
    // The A, B, C row accumulations are independent → run the three sparse
    // matvecs concurrently. (Each matvec is itself serial here; per-row CSR
    // parallelism within a matvec is a deferred follow-up.)
    let (mut out, (bb, cc)) = rayon::join(
        || accumulate_a_row(a, eq_s1, m),
        || {
            rayon::join(
                || accumulate_a_row(b, eq_s1, m),
                || accumulate_a_row(c, eq_r, m),
            )
        },
    );
    // Combine: disjoint output index `h` → parallel (bit-identical to serial).
    out.par_iter_mut().enumerate().for_each(|(h, o)| {
        *o = *o + gamma * bb[h] + gamma2 * cc[h];
    });
    out
}

/// `out[col] = Σ_{(row,col,v) ∈ mat} eq[row]·v` (R_0 factor lifted to R_2 via
/// `mul_by_gr1`). One sparse matvec used by [`build_combined_a_row`].
fn accumulate_a_row<L: Limb>(
    mat: &SparseMatrix<Gr1<L>>,
    eq: &[Gr128<L>],
    m: usize,
) -> Vec<Gr128<L>> {
    let mut v = vec![Gr128::<L>::ZERO; m];
    for &(row, col, val) in &mat.triples {
        v[col] = v[col] + eq[row].mul_by_gr1(val);
    }
    v
}

fn mbar_dot_e1<L: Limb>(mbar: &[Gr128<L>], e1: &[Gr128<L>]) -> Gr128<L> {
    debug_assert_eq!(mbar.len(), e1.len());
    let mut y = Gr128::<L>::ZERO;
    for j in 0..mbar.len() {
        y = y + mbar[j] * e1[j];
    }
    y
}

#[cfg(test)]
mod kernel_soundness_tests {
    use super::*;
    use reduction::matrix::SparseMatrix;

    /// Build a *grossly unsatisfied* SIMD `x·x = y` instance (every `y` wrong),
    /// so the malicious error is non-kernel in many positions.
    fn unsatisfied_simd() -> SimdR1csInstance<u32> {
        let log = 4usize;
        let n = 1usize << log;
        let m = n;
        let k = (m - 1) / 2;
        let mut a = SparseMatrix::<Gr1<u32>>::new(n, m);
        let mut b = SparseMatrix::<Gr1<u32>>::new(n, m);
        let mut c = SparseMatrix::<Gr1<u32>>::new(n, m);
        for i in 0..k {
            a.push(i, 1 + 2 * i, Gr1::<u32>::ONE);
            b.push(i, 1 + 2 * i, Gr1::<u32>::ONE);
            c.push(i, 2 + 2 * i, Gr1::<u32>::ONE);
        }
        let mut witnesses: [Vec<Gr1<u32>>; 32] = std::array::from_fn(|_| vec![]);
        for kk in 0..32 {
            let mut w = vec![Gr1::<u32>::ZERO; m];
            w[0] = Gr1::<u32>::ONE;
            for i in 0..k {
                let x = Gr1((((kk * 0x9E37) ^ (i * 0x85EB)) as u32).wrapping_add(1));
                w[1 + 2 * i] = x;
                w[2 + 2 * i] = x * x + Gr1(1); // y := x² + 1  ⇒  x·x ≠ y
            }
            witnesses[kk] = w;
        }
        SimdR1csInstance { a, b, c, witnesses }
    }

    /// Malicious SIMD prover: commits the **non-kernel** error `ê = â∘b̂ − ĉ`
    /// (instead of the honest `q1·v_α + q2·v_β`), so `claim_mult = ĉ(r) + ê(r) =
    /// (â∘b̂)(r)` and the mult-check passes even though `a∘b ≠ c`. Everything
    /// else mirrors `prove_simd_with_timing`. The only check that can catch this
    /// is the free-subspace check (`columns_in_kernel`).
    fn malicious_prove(
        simd: &SimdR1csInstance<u32>,
        params: &SimdSnarkParams<u32>,
    ) -> SimdSnarkProof<u32> {
        let log_n = params.log_n;
        let m = params.m();
        let packed = simd_to_packed(simd);

        // Honest w_phi commitment (packed form ⇒ Im φ by construction).
        let mut w_phi_packed: Vec<[u32; 32]> = Vec::with_capacity(m);
        for j in 0..m {
            let mut col = [0u32; 32];
            for kk in 0..32 {
                col[kk] = simd.witnesses[kk][j].0;
            }
            w_phi_packed.push(col);
        }
        let comm_w = params.pcs_w.commit_with_timing(&w_phi_packed).0;

        let a_csr = CsrMatrix::from_coo(&packed.a);
        let b_csr = CsrMatrix::from_coo(&packed.b);
        let c_csr = CsrMatrix::from_coo(&packed.c);
        let aw = matvec_gr128_csr(&a_csr, &packed.w_phi);
        let bw = matvec_gr128_csr(&b_csr, &packed.w_phi);
        let cw = matvec_gr128_csr(&c_csr, &packed.w_phi);

        // The cheat: ê = â∘b̂ − ĉ (∉ ker ψ wherever a∘b ≠ c).
        let e_mal: Vec<Gr128<u32>> = (0..aw.len()).map(|i| aw[i] * bw[i] - cw[i]).collect();
        let comm_e = params.pcs_q.commit_with_timing(&e_mal).0;

        let mut tr = Transcript::new(b"simd");
        tr.absorb_root(&comm_w.root);
        tr.absorb_root(&comm_e.root);
        let r_target: Vec<Gr128<u32>> = tr.challenge_target(b"mult_check_target", log_n);

        let eq_r = build_eq_table(&r_target);
        let cw_at_r = inner_product(eq_r.evals(), &cw);
        let e_at_r = inner_product(eq_r.evals(), &e_mal);
        let claim_mult = cw_at_r + e_at_r; // = (â∘b̂)(r)
        tr.absorb_gr128_slice(b"claim_mult", &[claim_mult]);

        tr.absorb_label(b"mult_check_sumcheck");
        let a_mle = Mle::new(aw);
        let b_mle = Mle::new(bw);
        let (mult_proof, s_1) = prove_mult_check(&a_mle, &b_mle, &r_target, claim_mult, &mut tr);
        absorb_mult_finals(&mut tr, &mult_proof);

        let gamma: Gr128<u32> = tr.challenge_gr128(b"linear_check_gamma");
        let gamma2 = gamma * gamma;
        let eq_s1 = build_eq_table(&s_1);
        let m_gamma_row = build_combined_a_row(
            &simd.a,
            &simd.b,
            &simd.c,
            eq_s1.evals(),
            eq_r.evals(),
            gamma,
            gamma2,
            m,
        );
        let claim_lin = mult_proof.final_a + gamma * mult_proof.final_b + gamma2 * cw_at_r;
        tr.absorb_gr128_slice(b"claim_lin", &[claim_lin]);
        tr.absorb_label(b"linear_check_sumcheck");
        let g1 = Mle::new(m_gamma_row);
        let g2 = Mle::new(packed.w_phi.clone());
        let (lin_proof, s_2) = prove_linear_check_simd_sumcheck_only(g1, g2, claim_lin, &mut tr);
        absorb_lin_finals(&mut tr, &lin_proof);

        let (_yw, open_w) = params.pcs_w.open(&comm_w, &s_2);
        let (_ye, open_e) = params.pcs_q.open(&comm_e, &r_target);

        SimdSnarkProof {
            root_w: comm_w.root,
            root_e: comm_e.root,
            mult_proof,
            lin_proof,
            open_w,
            open_e,
        }
    }

    #[test]
    fn malicious_nonkernel_error_rejected_by_free_subspace_check() {
        let simd = unsatisfied_simd();
        assert!(!simd.is_satisfied(), "test instance must be unsatisfied");

        let (k1, k2) = (1usize << 2, 1usize << 2); // 16 = 4×4
        let params = SimdSnarkParams {
            log_n: 4,
            log_m: 4,
            pcs_w: Pcs::new(k1, k2, 4, 16, 0xDEAD_BEEF),
            pcs_q: R2Pcs::new(k1, k2, 4, 16, 0xCAFE_BABE),
        };

        let proof = malicious_prove(&simd, &params);
        // The proof satisfies the mult-check, the linear-check and both PCS
        // openings — only the free-subspace check stands between it and
        // acceptance. Before this check, verify would (unsoundly) accept.
        let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
        assert_eq!(
            r,
            Err(VerifyError::KernelCheck),
            "forged non-kernel error must be rejected by the free-subspace check, got {r:?}"
        );
    }
}

// Bind the sumcheck final evaluations into the transcript. These MUST match
// between prover and verifier exactly (Fiat–Shamir). The round messages and
// the claim are already bound — round polys per-round via `Challenger::observe`
// inside the sumcheck, the claim via the `claim_mult` / `claim_lin` absorbs
// before each sumcheck.
fn absorb_mult_finals<L: Limb>(t: &mut Transcript, proof: &MultCheckProof<L>) {
    t.absorb_gr128_slice(b"mult/finals", &[proof.final_a, proof.final_b]);
}

fn absorb_lin_finals<L: Limb>(t: &mut Transcript, proof: &LinearCheckSimdProof<L>) {
    t.absorb_gr128_slice(b"lin/finals", &[proof.final_a_row, proof.final_w]);
}
