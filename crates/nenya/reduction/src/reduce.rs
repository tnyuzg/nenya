//! Three R1CS reductions from `Ring_zkSNARKs_using_Multiplicative_Embedding.pdf` §3.2:
//! SIMD (`simd_to_packed`, §3.2.1), MIMD (`mimd_to_packed`, §3.2.2), and
//! general single-circuit (`general_to_packed`, §3.2.3).
//!
//! All three target `κ = 32` matched to the `φ_2: R_0^32 → R_2 = GR(2^32, 128)`
//! embedding in [`crate::embedding`]. Output kernel-error pairs `(q1, q2)` are
//! the explicit decomposition `ê = q1·v_α + q2·v_β` from §3.3.4 eq. (13).

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::constants::{constants, HasPackingConstants};
use crate::embedding::{mul_with_quotients_only, pack_r2, rect_from_flat};
use crate::instance::{
    GeneralR1csInstance, MimdR1csInstance, PackedGeneralR1csInstance, PackedMimdR1csInstance,
    PackedSimdR1csInstance, SimdR1csInstance,
};
use crate::matrix::{matvec_gr1, CsrMatrix, SparseMatrix};
use galois_ring::{Gr1, Gr128, Limb};

// ---------------------------------------------------------------------------
// §3.2.1 SIMD
// ---------------------------------------------------------------------------

/// Paper §3.2.1 reduction: pack a 32-instance SIMD batch (shared `A, B, C`,
/// 32 witnesses) into a single packed instance.
///
/// Produces:
/// - `w_phi[j] = φ_2((w_0[j], …, w_31[j]))` for `j ∈ [m]`,
/// - `q1[i], q2[i]` for `i ∈ [n]`, the explicit kernel decomposition of
///   the multiplication-relation error.
pub fn simd_to_packed<L: HasPackingConstants>(
    batch: &SimdR1csInstance<L>,
) -> PackedSimdR1csInstance<L> {
    let m = batch.a.cols;
    let n = batch.a.rows;
    debug_assert_eq!(batch.b.cols, m);
    debug_assert_eq!(batch.c.cols, m);
    debug_assert_eq!(batch.b.rows, n);
    debug_assert_eq!(batch.c.rows, n);

    // 1. Pack the witnesses column-wise: w_phi[j] = φ_2((w_0[j], ..., w_31[j])).
    //    Each column `j` is independent → parallelize over `j`. The order-
    //    preserving `collect` keeps `w_phi` bit-identical to the serial loop.
    let w_phi: Vec<Gr128<L>> = (0..m)
        .into_par_iter()
        .map(|j| {
            let mut col = [Gr1::<L>::ZERO; 32];
            for k in 0..32 {
                col[k] = batch.witnesses[k][j];
            }
            pack_r2(col)
        })
        .collect();

    // 2. For each k ∈ [32], compute (A·w_k), (B·w_k) as length-n R_0 vectors.
    //    The mult-relation kernel comes from `mul_with_quotients` per row. The
    //    32 (matvec_a, matvec_b) pairs are independent → parallelize over `k`.
    let (aw, bw): (Vec<Vec<Gr1<L>>>, Vec<Vec<Gr1<L>>>) = (0..32)
        .into_par_iter()
        .map(|k| {
            (
                matvec_gr1(&batch.a, &batch.witnesses[k]),
                matvec_gr1(&batch.b, &batch.witnesses[k]),
            )
        })
        .unzip();

    // 3. For each constraint row i ∈ [n], assemble the 32-vectors, get the
    //    kernel pair `(q1, q2)` from `mul_with_quotients_only`, and fold it into
    //    the single combined kernel error `e[i] = q1[i]·v_α + q2[i]·v_β`. The
    //    verifier only uses this combination, so the SNARK commits `e` alone
    //    (one PCS / one query). `q1·v_α` uses the cheap lifted-R_1 fast path
    //    (`v_α ∈ R_1 ↪ R_2`); `q2·v_β` is a full R_2 mul (`v_β = β^16 − β`).
    //    Per-row independent → parallelize over `i`. `constants::<L>()` is forced
    //    here on the main thread so its `OnceLock` is warm before the parallel
    //    region (concurrent reads are then `Sync`).
    let cst = constants::<L>();
    let v_alpha = cst.v_alpha; // Gr4 (R_1)
    let v_beta = cst.v_beta; // Gr128 (R_2)
    let e: Vec<Gr128<L>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut x_row = [Gr1::<L>::ZERO; 32];
            let mut y_row = [Gr1::<L>::ZERO; 32];
            for k in 0..32 {
                x_row[k] = aw[k][i];
                y_row[k] = bw[k][i];
            }
            let (q1_i, q2_i) =
                mul_with_quotients_only(rect_from_flat(&x_row), rect_from_flat(&y_row));
            q1_i.mul_by_gr4(v_alpha) + q2_i * v_beta
        })
        .collect();

    PackedSimdR1csInstance {
        a: batch.a.clone(),
        b: batch.b.clone(),
        c: batch.c.clone(),
        w_phi,
        e,
    }
}

// ---------------------------------------------------------------------------
// §3.2.2 MIMD
// ---------------------------------------------------------------------------

/// Walk the 32 input matrices and return both:
/// - the packed `Â` as a `SparseMatrix<GR32_128>` whose support is the union
///   of the 32 input supports, and
/// - the unpacked 16×2 view of each packed entry, indexed by `(j, k)` —
///   needed downstream for kernel-quotient computation without reinvoking
///   `unpack_r2`.
/// `(r,c) → 16×2 view` of the packed matrix entries (for kernel-quotient accumulation).
pub type MatViews<L> = BTreeMap<(usize, usize), [[Gr1<L>; 2]; 16]>;

fn pack_matrix_set<L: HasPackingConstants>(
    rows: usize,
    cols: usize,
    matrices: &[SparseMatrix<Gr1<L>>; 32],
) -> (SparseMatrix<Gr128<L>>, MatViews<L>) {
    let mut entries: BTreeMap<(usize, usize), [Gr1<L>; 32]> = BTreeMap::new();
    for k in 0..32 {
        debug_assert_eq!(matrices[k].rows, rows);
        debug_assert_eq!(matrices[k].cols, cols);
        for &(r, c, v) in &matrices[k].triples {
            let entry = entries.entry((r, c)).or_insert([Gr1::<L>::ZERO; 32]);
            entry[k] = entry[k] + v;
        }
    }
    let mut packed = SparseMatrix::<Gr128<L>>::new(rows, cols);
    let mut views: MatViews<L> = BTreeMap::new();
    for ((r, c), col) in entries {
        let view = rect_from_flat(&col);
        packed.push(r, c, pack_r2(col));
        views.insert((r, c), view);
    }
    (packed, views)
}

/// Paper §3.2.2 reduction: pack 32 R1CS instances with independent
/// `(A_i, B_i, C_i)` matrices.
///
/// Produces a packed witness `w_phi`, packed matrices `Â, B̂, Ĉ` (each
/// entry in `Im(φ)`), four kernel-error pairs (one per linear relation
/// plus one for the multiplication relation), all of length `n`.
pub fn mimd_to_packed<L: HasPackingConstants>(
    mimd: &MimdR1csInstance<L>,
) -> PackedMimdR1csInstance<L> {
    let n = mimd.a_per_instance[0].rows;
    let m = mimd.a_per_instance[0].cols;

    // 1. Pack witnesses index-wise — same as SIMD.
    let mut w_phi: Vec<Gr128<L>> = Vec::with_capacity(m);
    let mut w_views: Vec<[[Gr1<L>; 2]; 16]> = Vec::with_capacity(m);
    for j in 0..m {
        let mut col = [Gr1::<L>::ZERO; 32];
        for k in 0..32 {
            col[k] = mimd.witnesses[k][j];
        }
        w_views.push(rect_from_flat(&col));
        w_phi.push(pack_r2(col));
    }

    // 2. Pack the three matrix sets.
    let (a_hat, a_views) = pack_matrix_set(n, m, &mimd.a_per_instance);
    let (b_hat, b_views) = pack_matrix_set(n, m, &mimd.b_per_instance);
    let (c_hat, c_views) = pack_matrix_set(n, m, &mimd.c_per_instance);

    // 3. For each linear relation, accumulate per-row kernel quotients.
    //    Each scalar product Â[r,c]·w_phi[c] contributes
    //    `mul_with_quotients_only(Â_view[r,c], w_view[c])`; sum over `c`
    //    gives the row-r kernel error pair.
    let zero = Gr128::<L>::ZERO;
    let accumulate =
        |hat: &SparseMatrix<Gr128<L>>, views: &MatViews<L>| -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
            let mut q1 = vec![zero; n];
            let mut q2 = vec![zero; n];
            for &(r, c, _) in &hat.triples {
                let m_view = views[&(r, c)];
                let w_view = w_views[c];
                let (dq1, dq2) = mul_with_quotients_only(m_view, w_view);
                q1[r] = q1[r] + dq1;
                q2[r] = q2[r] + dq2;
            }
            (q1, q2)
        };
    let (q1_a, q2_a) = accumulate(&a_hat, &a_views);
    let (q1_b, q2_b) = accumulate(&b_hat, &b_views);
    let (q1_c, q2_c) = accumulate(&c_hat, &c_views);

    // 4. Multiplication relation — identical to SIMD: per-instance matvec
    //    over R_0, then per-row 32-vectors.
    let mut aw: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    let mut bw: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    for k in 0..32 {
        aw[k] = matvec_gr1(&mimd.a_per_instance[k], &mimd.witnesses[k]);
        bw[k] = matvec_gr1(&mimd.b_per_instance[k], &mimd.witnesses[k]);
    }
    let mut q1_mult = Vec::with_capacity(n);
    let mut q2_mult = Vec::with_capacity(n);
    for i in 0..n {
        let mut x_row = [Gr1::<L>::ZERO; 32];
        let mut y_row = [Gr1::<L>::ZERO; 32];
        for k in 0..32 {
            x_row[k] = aw[k][i];
            y_row[k] = bw[k][i];
        }
        let (dq1, dq2) = mul_with_quotients_only(rect_from_flat(&x_row), rect_from_flat(&y_row));
        q1_mult.push(dq1);
        q2_mult.push(dq2);
    }

    PackedMimdR1csInstance {
        a_hat,
        b_hat,
        c_hat,
        w_phi,
        q1_a,
        q2_a,
        q1_b,
        q2_b,
        q1_c,
        q2_c,
        q1_mult,
        q2_mult,
    }
}

/// **Precomputed** (witness-independent) packed MIMD matrices `Â, B̂, Ĉ` and
/// their entry views. Deterministic from the 32 public matrix sets — setup, not
/// per-proof cost.
pub struct MimdMatrices<L: Limb> {
    pub a_hat: SparseMatrix<Gr128<L>>,
    pub b_hat: SparseMatrix<Gr128<L>>,
    pub c_hat: SparseMatrix<Gr128<L>>,
    pub a_views: MatViews<L>,
    pub b_views: MatViews<L>,
    pub c_views: MatViews<L>,
    /// Row-grouped copies of `Â,B̂,Ĉ` for the parallel matvec (setup cost).
    pub a_csr: CsrMatrix<Gr128<L>>,
    pub b_csr: CsrMatrix<Gr128<L>>,
    pub c_csr: CsrMatrix<Gr128<L>>,
}

/// Setup-time packing of the 32 MIMD matrix sets. Run once; reused across proofs.
pub fn pack_mimd_matrices<L: HasPackingConstants>(
    a_per: &[SparseMatrix<Gr1<L>>; 32],
    b_per: &[SparseMatrix<Gr1<L>>; 32],
    c_per: &[SparseMatrix<Gr1<L>>; 32],
) -> MimdMatrices<L> {
    let n = a_per[0].rows;
    let m = a_per[0].cols;
    let (a_hat, a_views) = pack_matrix_set(n, m, a_per);
    let (b_hat, b_views) = pack_matrix_set(n, m, b_per);
    let (c_hat, c_views) = pack_matrix_set(n, m, c_per);
    let a_csr = CsrMatrix::from_coo(&a_hat);
    let b_csr = CsrMatrix::from_coo(&b_hat);
    let c_csr = CsrMatrix::from_coo(&c_hat);
    MimdMatrices {
        a_hat,
        b_hat,
        c_hat,
        a_views,
        b_views,
        c_views,
        a_csr,
        b_csr,
        c_csr,
    }
}

/// **Per-proof** witness reduction for the MIMD SNARK: the packed witness `ŵ`
/// and the A,B linear-relation kernel quotients (against the precomputed
/// matrices). C and mult quotients are not produced — the SNARK folds the
/// C-error into `ê_mult'` and computes the mult error directly.
pub struct MimdWitnessParts<L: Limb> {
    pub w_phi: Vec<Gr128<L>>,
    pub q1_a: Vec<Gr128<L>>,
    pub q2_a: Vec<Gr128<L>>,
    pub q1_b: Vec<Gr128<L>>,
    pub q2_b: Vec<Gr128<L>>,
}

pub fn mimd_reduce_witness<L: HasPackingConstants>(
    witnesses: &[Vec<Gr1<L>>; 32],
    mats: &MimdMatrices<L>,
) -> MimdWitnessParts<L> {
    let n = mats.a_hat.rows;
    let m = mats.a_hat.cols;

    // Pack the witness columns in parallel (each `j` is an independent
    // `pack_r2`); order-preserving collect ⇒ bit-identical to the serial loop.
    let built: Vec<(Gr128<L>, [[Gr1<L>; 2]; 16])> = (0..m)
        .into_par_iter()
        .map(|j| {
            let mut col = [Gr1::<L>::ZERO; 32];
            for k in 0..32 {
                col[k] = witnesses[k][j];
            }
            let view = rect_from_flat(&col);
            (pack_r2(col), view)
        })
        .collect();
    let mut w_phi: Vec<Gr128<L>> = Vec::with_capacity(m);
    let mut w_views: Vec<[[Gr1<L>; 2]; 16]> = Vec::with_capacity(m);
    for (wp, wv) in built {
        w_phi.push(wp);
        w_views.push(wv);
    }

    let zero = Gr128::<L>::ZERO;
    // Per-triple kernel quotients are independent; collect them, then fold into
    // the row accumulators in fixed order (bit-identical to serial).
    let accumulate =
        |hat: &SparseMatrix<Gr128<L>>, views: &MatViews<L>| -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
            let contribs: Vec<(usize, Gr128<L>, Gr128<L>)> = hat
                .triples
                .par_iter()
                .map(|&(r, c, _)| {
                    let (dq1, dq2) = mul_with_quotients_only(views[&(r, c)], w_views[c]);
                    (r, dq1, dq2)
                })
                .collect();
            let mut q1 = vec![zero; n];
            let mut q2 = vec![zero; n];
            for &(r, dq1, dq2) in &contribs {
                q1[r] = q1[r] + dq1;
                q2[r] = q2[r] + dq2;
            }
            (q1, q2)
        };
    let (q1_a, q2_a) = accumulate(&mats.a_hat, &mats.a_views);
    let (q1_b, q2_b) = accumulate(&mats.b_hat, &mats.b_views);

    MimdWitnessParts {
        w_phi,
        q1_a,
        q2_a,
        q1_b,
        q2_b,
    }
}

// ---------------------------------------------------------------------------
// §3.2.3 General circuit
// ---------------------------------------------------------------------------

/// `rot_ℓ((x_0, …, x_31)) = (x_ℓ, x_{ℓ+1}, …, x_{ℓ-1})` (cyclic left shift).
fn rot_l<L: Limb>(x: &[Gr1<L>; 32], ell: usize) -> [Gr1<L>; 32] {
    let mut out = [Gr1::<L>::ZERO; 32];
    for i in 0..32 {
        out[i] = x[(i + ell) % 32];
    }
    out
}

/// Build the 32 packed cyclic-diagonal layers of an R_0 matrix `m` of size
/// `(n × n)` with `n` divisible by 32. Layer `ℓ` is the
/// `(n_blocks × n_blocks)` packed matrix whose `(p, q)` entry is the packed
/// `ℓ`-th cyclic diagonal of the κ×κ block `m^{(p,q)}`.
///
/// Returns the packed layers plus a per-layer `BTreeMap<(p, q), 16×2-view>`
/// of the packed entries — needed for kernel-quotient accumulation.
/// Per-layer `(p,q) → 16×2 view` of the packed diagonal entries.
/// Per-layer views, **parallel to** the corresponding `layers[ℓ].triples` (same
/// order) so the prover indexes them positionally — no `(p,q)` map lookup.
pub type DiagViews<L> = [Vec<[[Gr1<L>; 2]; 16]>; 32];

pub fn pack_diagonal_layers<L: HasPackingConstants>(
    m: &SparseMatrix<Gr1<L>>,
) -> ([SparseMatrix<Gr128<L>>; 32], DiagViews<L>) {
    let n = m.rows;
    debug_assert_eq!(m.cols, n);
    debug_assert_eq!(n % 32, 0);
    let n_blocks = n / 32;

    let mut layers: [SparseMatrix<Gr128<L>>; 32] =
        std::array::from_fn(|_| SparseMatrix::<Gr128<L>>::new(n_blocks, n_blocks));
    let mut views: DiagViews<L> = std::array::from_fn(|_| Vec::new());
    if m.triples.is_empty() {
        return (layers, views);
    }

    // 1. Map each nonzero to its diagonal-cell slot: ℓ = (j+32−i) mod 32 (where
    //    j = (i+ℓ) mod 32), with d^{(ℓ,p,q)}[i] += v. Parallel over the nonzeros.
    let mut items: Vec<(u8, usize, usize, u8, Gr1<L>)> = m
        .triples
        .par_iter()
        .map(|&(r, c, v)| {
            let (p, i) = (r / 32, r % 32);
            let (q, j) = (c / 32, c % 32);
            ((j + 32 - i) as u8 % 32, p, q, i as u8, v)
        })
        .collect();

    // 2. Sort by (ℓ, p, q) — parallel — so each diagonal cell's slot entries are
    //    contiguous, and within a layer the cells are already (p,q)-sorted
    //    (identical order to the previous BTreeMap version).
    items.par_sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));

    // 3. Cell boundaries (one entry per distinct (ℓ,p,q)); cheap serial scan.
    let mut bounds = vec![0usize];
    for k in 1..items.len() {
        if (items[k].0, items[k].1, items[k].2) != (items[k - 1].0, items[k - 1].1, items[k - 1].2)
        {
            bounds.push(k);
        }
    }
    bounds.push(items.len());

    // 4. Pack each cell in parallel: accumulate the 32-slot vector (sum is
    //    order-independent in the ring ⇒ bit-identical), then pack_r2 + view.
    let cells: Vec<(usize, usize, usize, Gr128<L>, [[Gr1<L>; 2]; 16])> = (0..bounds.len() - 1)
        .into_par_iter()
        .map(|g| {
            let (lo, hi) = (bounds[g], bounds[g + 1]);
            let (ell, p, q, _, _) = items[lo];
            let mut d = [Gr1::<L>::ZERO; 32];
            for k in lo..hi {
                let (_, _, _, i, v) = items[k];
                d[i as usize] = d[i as usize] + v;
            }
            (ell as usize, p, q, pack_r2(d), rect_from_flat(&d))
        })
        .collect();

    // 5. Distribute into the 32 layers — cells are in (ℓ,p,q) order, so each
    //    layer fills contiguously in (p,q) order, and views[ℓ] stays parallel to
    //    layers[ℓ].triples. Cheap serial Vec pushes (no map, no sort).
    for (ell, p, q, pk, view) in cells {
        layers[ell].push(p, q, pk);
        views[ell].push(view);
    }
    (layers, views)
}

/// Paper §3.2.3 reduction: pack a single R1CS over R_0 (witness length `n`
/// divisible by 32) using the Halevi–Shoup cyclic-diagonal decomposition.
///
/// Produces:
/// - block-packed witness `w_phi` of length `n / 32`,
/// - 32 packed rotation witnesses `rho[ℓ]` of length `n / 32`,
/// - 32 packed cyclic-diagonal layers per matrix,
/// - four kernel-error pairs (linear A/B/C + mult).
///
/// Each `rho[ℓ][q]` is built directly via `pack_r2(rot_ℓ(...))`, so it
/// lives in `Im(φ)` by construction. The verifier-side rotation relation
/// from §3.2.3 only constrains `ψ(rho)`; image membership is established
/// at the PCS layer (out of scope here).
pub fn general_to_packed<L: HasPackingConstants>(
    g: &GeneralR1csInstance<L>,
) -> PackedGeneralR1csInstance<L> {
    let n = g.a.rows;
    debug_assert_eq!(g.a.cols, n);
    debug_assert_eq!(g.b.rows, n);
    debug_assert_eq!(g.b.cols, n);
    debug_assert_eq!(g.c.rows, n);
    debug_assert_eq!(g.c.cols, n);
    debug_assert_eq!(g.w.len(), n);
    assert_eq!(
        n % 32,
        0,
        "general_to_packed requires witness length divisible by 32"
    );
    let n_blocks = n / 32;

    // 1. Block-pack the witness.
    let mut w_phi: Vec<Gr128<L>> = Vec::with_capacity(n_blocks);
    let mut w_blocks: Vec<[Gr1<L>; 32]> = Vec::with_capacity(n_blocks);
    for q in 0..n_blocks {
        let mut block = [Gr1::<L>::ZERO; 32];
        for i in 0..32 {
            block[i] = g.w[32 * q + i];
        }
        w_blocks.push(block);
        w_phi.push(pack_r2(block));
    }

    // 2. Build packed rotation witnesses rho[ℓ][q] = pack_r2(rot_ℓ(block_q))
    //    and their 16×2 views for quotient accumulation.
    let mut rho: [Vec<Gr128<L>>; 32] = std::array::from_fn(|_| Vec::with_capacity(n_blocks));
    let mut rho_views: [Vec<[[Gr1<L>; 2]; 16]>; 32] =
        std::array::from_fn(|_| Vec::with_capacity(n_blocks));
    for ell in 0..32 {
        for q in 0..n_blocks {
            let rotated = rot_l(&w_blocks[q], ell);
            rho_views[ell].push(rect_from_flat(&rotated));
            rho[ell].push(pack_r2(rotated));
        }
    }

    // 3. Pack diagonal layers for A, B, C.
    let (d_hat_a, a_views) = pack_diagonal_layers(&g.a);
    let (d_hat_b, b_views) = pack_diagonal_layers(&g.b);
    let (d_hat_c, c_views) = pack_diagonal_layers(&g.c);

    // 4. For each linear relation M ∈ {A, B, C}, accumulate kernel quotients
    //    across (ℓ, q): each (D̂_M^{(ℓ)}[p, q] · rho[ℓ][q]) contributes
    //    `mul_with_quotients_only` to the row-p pair.
    let zero = Gr128::<L>::ZERO;
    let accumulate = |layers: &[SparseMatrix<Gr128<L>>; 32], views: &DiagViews<L>|
     -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
        let mut q1 = vec![zero; n_blocks];
        let mut q2 = vec![zero; n_blocks];
        for ell in 0..32 {
            for (idx, &(p, q, _)) in layers[ell].triples.iter().enumerate() {
                let d_view = views[ell][idx];
                let r_view = rho_views[ell][q];
                let (dq1, dq2) = mul_with_quotients_only(d_view, r_view);
                q1[p] = q1[p] + dq1;
                q2[p] = q2[p] + dq2;
            }
        }
        (q1, q2)
    };
    let (q1_a, q2_a) = accumulate(&d_hat_a, &a_views);
    let (q1_b, q2_b) = accumulate(&d_hat_b, &b_views);
    let (q1_c, q2_c) = accumulate(&d_hat_c, &c_views);

    // 5. Multiplication relation: per block-row p, the 32-vector inputs are
    //    just (A·w)[32p..32p+32] and (B·w)[32p..32p+32] over R_0.
    let aw = matvec_gr1(&g.a, &g.w);
    let bw = matvec_gr1(&g.b, &g.w);
    let mut q1_mult = Vec::with_capacity(n_blocks);
    let mut q2_mult = Vec::with_capacity(n_blocks);
    // We need c_phi here — without it the verifier has nothing to compare
    // against, but in fact the explicit ã = Σ D̂·rho - ê^A construction
    // puts ã, b̃, ĉ directly in `Im(φ)` and `is_satisfied` does the rest.
    // So per block-row we only need the kernel pair from `mul_with_quotients_only`.
    for p in 0..n_blocks {
        let mut x_row = [Gr1::<L>::ZERO; 32];
        let mut y_row = [Gr1::<L>::ZERO; 32];
        for i in 0..32 {
            x_row[i] = aw[32 * p + i];
            y_row[i] = bw[32 * p + i];
        }
        let (dq1, dq2) = mul_with_quotients_only(rect_from_flat(&x_row), rect_from_flat(&y_row));
        q1_mult.push(dq1);
        q2_mult.push(dq2);
    }

    PackedGeneralR1csInstance {
        d_hat_a,
        d_hat_b,
        d_hat_c,
        w_phi,
        rho,
        q1_a,
        q2_a,
        q1_b,
        q2_b,
        q1_c,
        q2_c,
        q1_mult,
        q2_mult,
    }
}

/// **Precomputed** (witness-independent) packed cyclic-diagonal layers of
/// `A, B, C`. Deterministic from the public matrices — a one-time *setup* cost,
/// **not** a per-proof cost (paper §3.1.3: the packed diagonals "are
/// deterministic and incur no prover cost").
pub struct GeneralDiagonals<L: Limb> {
    pub d_hat_a: [SparseMatrix<Gr128<L>>; 32],
    pub d_hat_b: [SparseMatrix<Gr128<L>>; 32],
    pub d_hat_c: [SparseMatrix<Gr128<L>>; 32],
    /// CSR (grouped by output row) of each layer — lets the prover's
    /// `Σ_ℓ D̂·ρ̂` matvec parallelize over output rows (the big axis) instead of
    /// the 32 layers. Setup-precomputed alongside the COO layers.
    pub csr_a: [CsrMatrix<Gr128<L>>; 32],
    pub csr_b: [CsrMatrix<Gr128<L>>; 32],
    pub csr_c: [CsrMatrix<Gr128<L>>; 32],
    /// CSC (grouped by column = `from_coo` of the transposed layer) — lets
    /// `build_layer_rows` (`M^(ℓ)[q] = Σ_p eq[p]·D̂[p,q]`) parallelize over the
    /// output column `q`. For these, `col_idx` holds the original row `p`.
    pub csc_a: [CsrMatrix<Gr128<L>>; 32],
    pub csc_b: [CsrMatrix<Gr128<L>>; 32],
    pub csc_c: [CsrMatrix<Gr128<L>>; 32],
    pub a_views: DiagViews<L>,
    pub b_views: DiagViews<L>,
    pub c_views: DiagViews<L>,
}

/// CSR layers (by output row) of each diagonal layer.
fn csr_layers<L: Limb>(layers: &[SparseMatrix<Gr128<L>>; 32]) -> [CsrMatrix<Gr128<L>>; 32] {
    let v: Vec<CsrMatrix<Gr128<L>>> = (0..32)
        .into_par_iter()
        .map(|ell| CsrMatrix::from_coo(&layers[ell]))
        .collect();
    let mut it = v.into_iter();
    std::array::from_fn(|_| it.next().unwrap())
}

/// CSC layers (by column): transpose each layer (swap `(p,q)`), then group by
/// row — so `row_ptr[q]` indexes column `q`'s entries with `col_idx = p`.
fn csc_layers<L: Limb>(layers: &[SparseMatrix<Gr128<L>>; 32]) -> [CsrMatrix<Gr128<L>>; 32] {
    let v: Vec<CsrMatrix<Gr128<L>>> = (0..32)
        .into_par_iter()
        .map(|ell| {
            let m = &layers[ell];
            let mut t = SparseMatrix::new(m.cols, m.rows);
            for &(p, q, v) in &m.triples {
                t.push(q, p, v);
            }
            CsrMatrix::from_coo(&t)
        })
        .collect();
    let mut it = v.into_iter();
    std::array::from_fn(|_| it.next().unwrap())
}

/// Setup-time diagonal packing for the general circuit. Run once per public
/// `(A, B, C)`; reused across all proofs.
pub fn pack_general_matrices<L: HasPackingConstants>(
    a: &SparseMatrix<Gr1<L>>,
    b: &SparseMatrix<Gr1<L>>,
    c: &SparseMatrix<Gr1<L>>,
) -> GeneralDiagonals<L> {
    // Each matrix's pack + CSR + CSC is one unit; run the three concurrently so a
    // unit's serial tail (bounds scan, layer distribute) overlaps the others'
    // parallel phases. Each unit is itself parallel (par_sort + par pack + 32-way
    // CSR/CSC), and rayon work-stealing balances the nested parallelism.
    let unit = |m: &SparseMatrix<Gr1<L>>| {
        let (d, v) = pack_diagonal_layers(m);
        let csr = csr_layers(&d);
        let csc = csc_layers(&d);
        (d, v, csr, csc)
    };
    let (
        (d_hat_a, a_views, csr_a, csc_a),
        ((d_hat_b, b_views, csr_b, csc_b), (d_hat_c, c_views, csr_c, csc_c)),
    ) = rayon::join(|| unit(a), || rayon::join(|| unit(b), || unit(c)));
    GeneralDiagonals {
        d_hat_a,
        d_hat_b,
        d_hat_c,
        csr_a,
        csr_b,
        csr_c,
        csc_a,
        csc_b,
        csc_c,
        a_views,
        b_views,
        c_views,
    }
}

/// The **per-proof** witness reduction output for the general-circuit SNARK:
/// the 32 packed block-rotations `ρ̂` and the A,B linear-relation kernel
/// quotients. (C and mult quotients are intentionally *not* produced — the
/// SNARK folds the C-kernel error into `ê_mult' = â∘b̂ − C_sum` and computes the
/// mult error directly, so they are never needed.)
pub struct GeneralWitnessParts<L: Limb> {
    pub rho: [Vec<Gr128<L>>; 32],
    pub q1_a: Vec<Gr128<L>>,
    pub q2_a: Vec<Gr128<L>>,
    pub q1_b: Vec<Gr128<L>>,
    pub q2_b: Vec<Gr128<L>>,
}

/// Per-proof witness reduction: block-rotations `ρ̂^(ℓ)` and the A,B kernel
/// quotients, against the **precomputed** diagonals. This — *not*
/// [`general_to_packed`] — is the prover's reduction cost; the diagonal packing
/// in [`pack_general_matrices`] is setup.
pub fn general_reduce_witness<L: HasPackingConstants>(
    w: &[Gr1<L>],
    diag: &GeneralDiagonals<L>,
) -> GeneralWitnessParts<L> {
    let n = w.len();
    assert_eq!(n % 32, 0);
    let n_blocks = n / 32;

    // Block-pack the witness and build the 32 packed rotations + their views.
    let mut w_blocks: Vec<[Gr1<L>; 32]> = Vec::with_capacity(n_blocks);
    for q in 0..n_blocks {
        let mut block = [Gr1::<L>::ZERO; 32];
        for i in 0..32 {
            block[i] = w[32 * q + i];
        }
        w_blocks.push(block);
    }
    // Build the 32 packed rotations + views. Serial over the 32 layers, but each
    // layer parallelizes over its `n_blocks` independent rot/pack ops (the big
    // axis, ≫ #cores) — so all cores are used instead of capping at 32.
    let built: Vec<(Vec<Gr128<L>>, Vec<[[Gr1<L>; 2]; 16]>)> = (0..32)
        .map(|ell| {
            (0..n_blocks)
                .into_par_iter()
                .map(|q| {
                    let rotated = rot_l(&w_blocks[q], ell);
                    let view = rect_from_flat(&rotated);
                    (pack_r2(rotated), view)
                })
                .unzip()
        })
        .collect();
    let mut rho: [Vec<Gr128<L>>; 32] = std::array::from_fn(|_| Vec::new());
    let mut rho_views: [Vec<[[Gr1<L>; 2]; 16]>; 32] = std::array::from_fn(|_| Vec::new());
    for (ell, (r, v)) in built.into_iter().enumerate() {
        rho[ell] = r;
        rho_views[ell] = v;
    }

    // Accumulate the A,B linear-relation kernel quotients. Serial over the 32
    // layers, parallelizing the expensive per-triple `mul_with_quotients_only`
    // over each layer's nonzeros (the big axis), then folding the cheap adds.
    // Fold order fixed by `ℓ` then triple order ⇒ bit-identical to serial.
    let zero = Gr128::<L>::ZERO;
    let accumulate = |layers: &[SparseMatrix<Gr128<L>>; 32], views: &DiagViews<L>| {
        let mut q1 = vec![zero; n_blocks];
        let mut q2 = vec![zero; n_blocks];
        for ell in 0..32 {
            let contrib: Vec<(usize, Gr128<L>, Gr128<L>)> = layers[ell]
                .triples
                .par_iter()
                .enumerate()
                .map(|(idx, &(p, q, _))| {
                    let (dq1, dq2) =
                        mul_with_quotients_only(views[ell][idx], rho_views[ell][q]);
                    (p, dq1, dq2)
                })
                .collect();
            for (p, dq1, dq2) in contrib {
                q1[p] = q1[p] + dq1;
                q2[p] = q2[p] + dq2;
            }
        }
        (q1, q2)
    };
    let (q1_a, q2_a) = accumulate(&diag.d_hat_a, &diag.a_views);
    let (q1_b, q2_b) = accumulate(&diag.d_hat_b, &diag.b_views);

    GeneralWitnessParts {
        rho,
        q1_a,
        q2_a,
        q1_b,
        q2_b,
    }
}
