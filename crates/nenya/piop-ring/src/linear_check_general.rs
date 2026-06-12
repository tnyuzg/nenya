//! Degree-2 sumcheck for `∑_h ∑_ℓ Row^{(ℓ)}[h] · ρ^{(ℓ)}[h]` — the linear
//! check for the general-circuit reduction (paper §3.2.3, Halevi–Shoup
//! packing).
//!
//! # Protocol
//!
//! The general reduction takes a single R1CS instance with witness length
//! `N = 2^LOG_N` u32 and reduces it, via the Halevi–Shoup cyclic-diagonal
//! decomposition, to a `LAYERS = κ = 32`-way claim over `R_2`:
//!
//! ```text
//! Claim:  c  ?=  Σ_{h ∈ {0,1}^{LOG_N - 5}}  Σ_{ℓ ∈ [LAYERS]}
//!                    Row^{(ℓ)}(h) · ρ^{(ℓ)}(h)
//! ```
//!
//! - `Row^{(ℓ)}` is the row-projection of the `ℓ`-th cyclic-diagonal slice
//!   of the (γ-batched) matrix `M = A + γ·B + γ²·C`, evaluated against the
//!   row challenge `r_target`:
//!
//!   ```text
//!   Row^{(ℓ)}[h] = Σ_y eq(r_target, y) · M^{(ℓ)}[y, h]
//!   ```
//!
//!   Built by the caller; this crate sees only the resulting 32 row MLEs.
//!
//! - `ρ^{(ℓ)}` is the `ℓ`-th cyclic rotation of the packed witness `w_φ`.
//!   `ρ^{(0)} = w_φ`. The 32 rotation views are derived from a single
//!   `w_φ` commitment via the "virtual PCS" trick on the verifier side.
//!
//! Each round of the sumcheck folds one variable from the shared `h`
//! domain and produces a degree-2 univariate (the sum stays linear in the
//! sumcheck variable, multiplied across `Row · ρ` for two factors → deg 2).
//! Round messages are in coefficient form `[a₀, a₁, a₂]` so the verifier
//! can advance via Horner.
//!
//! Soundness: deg-2 ⇒ `2(LOG_N − 5)/2¹²⁸` over the Teichmüller sampler.
//!
//! # Where the cost lives
//!
//! Per round `i`, with `m = 2^{LOG_N − 5 − 1 − i}` fold-pair indices:
//!
//! | Stage                                     | R_2 muls/round |
//! |-------------------------------------------|---------------:|
//! | `compute_round_poly`: 2 muls/pair/layer   | 64 · m         |
//! | `fold` of 32 rows + 32 ρ's: 1 mul/elt     | 64 · m         |
//!
//! Total across all rounds ≈ 128 · (2^{LOG_N − 5} − 1) ≈ `4 · 2^LOG_N`
//! R_2 muls — i.e. 32× more than SIMD/MIMD, which fold a single MLE pair
//! at each round.
//!
//! # Two parallelization optimizations (this file)
//!
//! The 32 layers are independent at every fold-pair-index — they only
//! collide at the per-round reduction into the shared `s0`, `s_inf`
//! accumulators. We exploit this at two levels:
//!
//! ## A. ILP via per-layer accumulators (single-thread win)
//!
//! The naive inner loop reads/writes a single `s0` per iteration:
//!
//! ```text
//! for i in 0..m:
//!   for ell in 0..LAYERS:
//!       s0   += row^{(ℓ)}_lo * ρ^{(ℓ)}_lo      // RAW chain on s0
//!       sinf += Δrow^{(ℓ)}   * Δρ^{(ℓ)}        // RAW chain on sinf
//! ```
//!
//! The read-after-write dependency on `s0`, `sinf` serializes iterations
//! even though the muls themselves are independent. Splitting into 32
//! per-layer accumulators breaks the chain — the OOO scheduler keeps
//! many `GR32_128 *` operations in flight at once. Reduction across
//! layers is `2 · (LAYERS − 1)` adds per round, cost-free against the
//! `64 · m` muls.
//!
//! Expected speedup on a single OOO core: 2–4× (depends on how many
//! mul/add units the core has and whether memory bandwidth saturates).
//!
//! ## B. Layer-level threading via `rayon`
//!
//! Within each round, the 32 (layer, fold-pair-index) units are fully
//! independent. We rayon-parallelize over **layers** (granularity = 32),
//! not over indices, because layer parallelism is constant across rounds
//! while index parallelism collapses in late rounds (round `LOG_N − 6`
//! has only one fold-pair-index).
//!
//! ```text
//! per round:
//!   (s0, sinf) = (0..LAYERS).par_iter().map(|ell| {
//!       per-layer compute_round_poly  // returns (s0_ℓ, sinf_ℓ)
//!   }).reduce(...);
//!
//!   sample r;
//!   (0..LAYERS).par_iter().for_each(|ell| {
//!       rows[ell].fold_in_place(r);
//!       rhos[ell].fold_in_place(r);
//!   });
//! ```
//!
//! Synchronization at round boundaries is unavoidable (next round needs
//! the sampled challenge `r` and the folded MLEs). Within a round both
//! the sumcheck-product step and the fold step are parallel.
//!
//! Expected wall-time saving on `T` cores: roughly `T×` until `T = 32`,
//! plateauing thereafter. On 8 P-cores: ~8×.
//!
//! ## A and B compose
//!
//! Each parallel task runs the per-layer reducer for *its* layer — that
//! per-task body is itself a tight, dependency-free hot loop (no shared
//! accumulator to bottleneck on), so each thread independently extracts
//! ILP inside its lane. Total ceiling is `cores × per-core-ILP`.

use galois_ring::{Gr128, Limb};
use rayon::prelude::*;

use crate::error::VerifyError;
use crate::fs::Challenger;
use crate::mle::Mle;

/// Number of cyclic-diagonal layers in the general-circuit reduction. Tied
/// to `κ = 32` from the `φ_2: R_0^32 → R_2` embedding.
pub const LAYERS: usize = 32;

/// Proof of `sum = ∑_h ∑_{ℓ ∈ [LAYERS]} Row^{(ℓ)}[h] · ρ^{(ℓ)}[h]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCheckGeneralProof<L: Limb = u32> {
    pub claimed_sum: Gr128<L>,
    /// Round-`i` univariate `s_i(Z) = a0 + a1·Z + a2·Z²`, in coefficient form.
    /// Length = `num_vars`.
    pub round_polys: Vec<[Gr128<L>; 3]>,
    /// `Row^{(ℓ)}(challenges)` for each layer.
    pub final_row: [Gr128<L>; LAYERS],
    /// `ρ^{(ℓ)}(challenges)` for each layer.
    pub final_rho: [Gr128<L>; LAYERS],
}

impl<L: Limb> LinearCheckGeneralProof<L> {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }

    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;

        // claimed_sum: Gr128<L>
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

        // final_row: [Gr128<L>; LAYERS]
        for gr in self.final_row.iter() {
            for c in gr.0.iter() {
                for w in c.0.iter() {
                    buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                }
            }
        }

        // final_rho: [Gr128<L>; LAYERS]
        for gr in self.final_rho.iter() {
            for c in gr.0.iter() {
                for w in c.0.iter() {
                    buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                }
            }
        }

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Run the deg-2 general-circuit linear-check sumcheck prover.
///
/// Inputs:
/// - `rows[ℓ]`: pre-batched length-`2^n` MLE for the row-projection of layer
///   `ℓ`. The caller is responsible for any A/B/C γ-batching that produces
///   this (verifier-recomputable preprocessing — not part of PIOP-prover
///   time).
/// - `rhos[ℓ]`: length-`2^n` MLE of the packed rotation witness for layer
///   `ℓ`. (`ρ^{(0)}` is `w_φ` itself.)
/// - `claim`: the claimed value of the double sum.
///
/// Every `rows[ℓ]` and `rhos[ℓ]` must agree on `num_vars`.
///
/// Returns the proof and the challenge sequence (`r_1, …, r_n`, `r_1` first
/// folded — same convention as [`crate::Mle::fold_in_place`]).
///
/// # Parallelization
///
/// Both the round-message computation and the per-round fold are
/// parallelized over the 32 layers via `rayon`. See file-level docs for
/// the two-level (ILP + threading) strategy.
pub fn prove_linear_check_general<L: Limb>(
    mut rows: [Mle<L>; LAYERS],
    mut rhos: [Mle<L>; LAYERS],
    claim: Gr128<L>,
    challenger: &mut impl Challenger<L>,
) -> (LinearCheckGeneralProof<L>, Vec<Gr128<L>>) {
    let n = rows[0].num_vars();
    for ell in 0..LAYERS {
        assert_eq!(rows[ell].num_vars(), n, "rows[{ell}] num_vars mismatch");
        assert_eq!(rhos[ell].num_vars(), n, "rhos[{ell}] num_vars mismatch");
    }
    assert!(n > 0, "linear check on 0-variable MLEs is meaningless");

    // Working tables folded in place per round — taken BY VALUE (moved in), so no
    // 32 GB clone of the 64 MLEs (the caller no longer needs them).

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let mut c = claim;

    for _ in 0..n {
        // (1) Layer-parallel sumcheck-product step. Each layer accumulates
        //     its own (s0_ℓ, sinf_ℓ) over the fold-pair-index range; rayon
        //     reduces across layers at the end.
        let poly = compute_round_poly_parallel(&rows, &rhos, c);
        // Per-round Fiat–Shamir: bind the round message before its challenge.
        challenger.observe(&poly);
        round_polys.push(poly);

        // (2) Sample the next challenge (serial across rounds).
        let r = challenger.sample();
        challenges.push(r);

        // (3) Advance the running claim via Horner: c ← a0 + r·(a1 + r·a2).
        let [a0, a1, a2] = poly;
        c = a0 + r * (a1 + r * a2);

        // (4) Layer-parallel fold step. Each layer's rows[ℓ], rhos[ℓ]
        //     fold independently in place.
        fold_layers_parallel(&mut rows, &mut rhos, r);
    }

    let mut final_row = [Gr128::<L>::ZERO; LAYERS];
    let mut final_rho = [Gr128::<L>::ZERO; LAYERS];
    for ell in 0..LAYERS {
        final_row[ell] = rows[ell].final_eval();
        final_rho[ell] = rhos[ell].final_eval();
    }

    let proof = LinearCheckGeneralProof {
        claimed_sum: claim,
        round_polys,
        final_row,
        final_rho,
    };
    (proof, challenges)
}

/// Layer-parallel round-message builder.
///
/// Maps `0..LAYERS` to per-layer `(s0_ℓ, sinf_ℓ)` partial sums via
/// `par_iter`, then reduces by ring-add into the round-wide `(s0, sinf)`.
/// Per-layer body is a tight, dependency-free hot loop in `{0, 1, ∞}`
/// form — the only state inside is the local `s0_ℓ`/`sinf_ℓ`, which the
/// CPU can pipeline freely (the ILP win from per-layer accumulators).
///
/// Recovers `a₁` once per round from the running claim
/// `c_prev = s(0) + s(1)`, never computing `s(1)` explicitly:
///
/// ```text
/// a₁ = s(1) − s(0) − s(∞)
///    = (c_prev − s(0)) − s(0) − s(∞)
///    = c_prev − 2·s(0) − s(∞)
/// ```
#[inline]
fn compute_round_poly_parallel<L: Limb>(
    rows: &[Mle<L>],
    rhos: &[Mle<L>],
    c_prev: Gr128<L>,
) -> [Gr128<L>; 3] {
    let m = rows[0].len() / 2;

    // Parallelize over the HYPERCUBE (the big axis, up to 2^(LOG_N-5)), summing
    // the 32 layers' contributions at each point. The previous version
    // parallelized only over the 32 layers, capping utilization at 32 cores on
    // a many-core box while serializing the heavy inner sweep. Ring addition is
    // exactly associative + commutative, so reassociating the sum (over the
    // hypercube and across rayon's reduce tree) is **bit-identical** regardless
    // of thread count.
    // Each layer's fold parallelizes over the hypercube (the big axis), summed
    // serially across the 32 layers. Layers-outer / points-inner keeps each
    // layer's access **contiguous** (cache-friendly streaming) while using all
    // cores on the heavy `m`-axis — vs a points-outer loop that would thrash 32
    // separate layer buffers per point. Bit-identical (exact assoc+comm add).
    let mut s0 = Gr128::<L>::ZERO;
    let mut sinf = Gr128::<L>::ZERO;
    for ell in 0..LAYERS {
        let r = rows[ell].evals();
        let p = rhos[ell].evals();
        let (s0_ell, sinf_ell) = (0..m)
            .into_par_iter()
            .map(|i| {
                let r_lo = r[i];
                let r_hi = r[i + m];
                let p_lo = p[i];
                let p_hi = p[i + m];
                (r_lo * p_lo, (r_hi - r_lo) * (p_hi - p_lo))
            })
            .reduce(
                || (Gr128::<L>::ZERO, Gr128::<L>::ZERO),
                |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
            );
        s0 = s0 + s0_ell;
        sinf = sinf + sinf_ell;
    }

    // Coefficient form via the c_prev shortcut: 1 sub-chain, no muls.
    let two_s0 = s0 + s0;
    let a1 = c_prev - two_s0 - sinf;
    [s0, a1, sinf]
}

/// Fold all 32 (rows[ℓ], ρ[ℓ]) pairs in place by `r`.
///
/// `Mle::fold_in_place` is itself parallel over the hypercube (the big axis),
/// so we iterate the 32 MLEs **serially** and let each fold saturate the cores
/// — rather than nesting a coarse 32-way `par_iter_mut` over an already-parallel
/// inner fold (which capped utilization at 32 threads).
#[inline]
fn fold_layers_parallel<L: Limb>(rows: &mut [Mle<L>], rhos: &mut [Mle<L>], r: Gr128<L>) {
    for row in rows.iter_mut() {
        row.fold_in_place(r);
    }
    for rho in rhos.iter_mut() {
        rho.fold_in_place(r);
    }
}

/// Verify a general-circuit linear-check proof.
///
/// On success returns the binding point `(r_1, …, r_n)` (sample order, MSB
/// first — same convention as the prover's challenge sequence).
///
/// The verifier accepts `proof.final_row` and `proof.final_rho` at face
/// value; the caller is expected to plug in:
/// - independent verification of `final_row[ℓ]` against the public matrices
///   (caller has the matrices, γ, eq tables);
/// - PCS opens for `final_rho[ℓ]` (e.g. via the virtual-PCS trick that
///   derives all 32 from a single `w_φ` commitment).
///
/// Verifier work is O(LOG_N · LAYERS) ring operations — far below the
/// prover's O(N · LAYERS / cores) — so it remains serial.
pub fn verify_linear_check_general<L: Limb>(
    proof: &LinearCheckGeneralProof<L>,
    challenger: &mut impl Challenger<L>,
) -> Result<Vec<Gr128<L>>, VerifyError> {
    let n = proof.round_polys.len();
    assert!(n > 0, "linear-check proof has zero rounds");

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(n);

    for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        // Round-consistency check: s_i(0) + s_i(1) = 2·a0 + a1 + a2 = c_i.
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        // Per-round Fiat–Shamir: observe before sampling (prover's order).
        challenger.observe(&[a0, a1, a2]);
        let r = challenger.sample();
        challenges.push(r);

        // Horner update: c_{i+1} = s_i(r) = a0 + r·(a1 + r·a2).
        c = a0 + r * (a1 + r * a2);
    }

    // Final check: c_n == Σ_ℓ final_row[ℓ] · final_rho[ℓ].
    let mut final_sum = Gr128::<L>::ZERO;
    for ell in 0..LAYERS {
        final_sum = final_sum + proof.final_row[ell] * proof.final_rho[ell];
    }
    if c != final_sum {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}
