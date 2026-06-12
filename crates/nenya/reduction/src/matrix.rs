//! Sparse R_0 matrices and their R_0/R_2 mat·vec operators.
//!
//! Two layouts: [`SparseMatrix`] (COO triples, for construction / `is_satisfied`
//! / row-projection) and [`CsrMatrix`] (row-grouped, for the **parallel**
//! mat·vec on the prover's hot path — each output row is independent, so the
//! matvec parallelizes over rows with no contention and bounded memory).

use galois_ring::{Gr1, Gr128, Limb};
use rayon::prelude::*;

/// Coordinate-format sparse matrix. Duplicate `(r, c)` entries are summed in
/// `matvec_*`; callers should avoid pushing them unless that's the intent.
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub triples: Vec<(usize, usize, T)>,
}

impl<T: Copy> SparseMatrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triples: vec![],
        }
    }
    /// Append a non-zero entry. Bounds (`r < rows && c < cols`) are checked
    /// only in debug builds.
    pub fn push(&mut self, r: usize, c: usize, v: T) {
        debug_assert!(r < self.rows && c < self.cols);
        self.triples.push((r, c, v));
    }
}

/// Row-grouped (CSR) sparse matrix: `row_ptr[r]..row_ptr[r+1]` indexes the
/// `(col_idx, vals)` entries of row `r`. Built once from a [`SparseMatrix`];
/// the `matvec_*_csr` operators then parallelize over output rows.
#[derive(Clone, Debug)]
pub struct CsrMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<T>,
}

impl<T: Copy + Send + Sync> CsrMatrix<T> {
    /// Group COO triples by row (stable, so duplicate `(r,c)` keep their COO
    /// order — `matvec_*_csr` then sums them, matching the COO operators).
    pub fn from_coo(m: &SparseMatrix<T>) -> Self {
        let mut triples = m.triples.clone();
        triples.sort_by_key(|&(r, _, _)| r);
        let mut row_ptr = vec![0usize; m.rows + 1];
        for &(r, _, _) in &triples {
            row_ptr[r + 1] += 1;
        }
        for r in 0..m.rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        let col_idx = triples.iter().map(|&(_, c, _)| c).collect();
        let vals = triples.iter().map(|&(_, _, v)| v).collect();
        Self {
            rows: m.rows,
            cols: m.cols,
            row_ptr,
            col_idx,
            vals,
        }
    }
}

/// `m · w`  (R_0 matrix, R_0 vector → R_0 vector), parallel over rows.
pub fn matvec_gr1_csr<L: Limb>(m: &CsrMatrix<Gr1<L>>, w: &[Gr1<L>]) -> Vec<Gr1<L>> {
    assert_eq!(w.len(), m.cols);
    (0..m.rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = Gr1::<L>::ZERO;
            for k in m.row_ptr[r]..m.row_ptr[r + 1] {
                acc = acc + m.vals[k] * w[m.col_idx[k]];
            }
            acc
        })
        .collect()
}

/// `m · w`  (R_0 matrix lifted to R_2, R_2 vector → R_2 vector), parallel over rows.
///
/// Each entry is an `R_0` scalar times an `R_2` vector element, so it uses the
/// `mul_by_gr1` fast path (~128 limb-muls, no reduction) rather than lifting the
/// scalar to `R_2` and running the full schoolbook multiply (~16k limb-muls).
pub fn matvec_gr128_csr<L: Limb>(m: &CsrMatrix<Gr1<L>>, w: &[Gr128<L>]) -> Vec<Gr128<L>> {
    assert_eq!(w.len(), m.cols);
    (0..m.rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = Gr128::<L>::ZERO;
            for k in m.row_ptr[r]..m.row_ptr[r + 1] {
                acc = acc + w[m.col_idx[k]].mul_by_gr1(m.vals[k]);
            }
            acc
        })
        .collect()
}

/// `m · w`  (R_2 matrix, R_2 vector → R_2 vector), parallel over rows.
pub fn matvec_gr128_dense_csr<L: Limb>(
    m: &CsrMatrix<Gr128<L>>,
    w: &[Gr128<L>],
) -> Vec<Gr128<L>> {
    assert_eq!(w.len(), m.cols);
    (0..m.rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = Gr128::<L>::ZERO;
            for k in m.row_ptr[r]..m.row_ptr[r + 1] {
                acc = acc + m.vals[k] * w[m.col_idx[k]];
            }
            acc
        })
        .collect()
}

/// `m · w`  (R_0 matrix, R_0 vector → R_0 vector)
pub fn matvec_gr1<L: Limb>(m: &SparseMatrix<Gr1<L>>, w: &[Gr1<L>]) -> Vec<Gr1<L>> {
    assert_eq!(w.len(), m.cols);
    let mut out = vec![Gr1::<L>::ZERO; m.rows];
    for &(r, c, v) in &m.triples {
        out[r] = out[r] + v * w[c];
    }
    out
}

/// `m · w`  (R_0 matrix lifted to R_2 entries, R_2 vector → R_2 vector).
/// Each `R_0` scalar acts on its `R_2` vector element via the `mul_by_gr1`
/// fast path (~128 limb-muls) instead of a lift-then-schoolbook multiply.
pub fn matvec_gr128<L: Limb>(m: &SparseMatrix<Gr1<L>>, w: &[Gr128<L>]) -> Vec<Gr128<L>> {
    assert_eq!(w.len(), m.cols);
    let mut out = vec![Gr128::<L>::ZERO; m.rows];
    for &(r, c, v) in &m.triples {
        out[r] = out[r] + w[c].mul_by_gr1(v);
    }
    out
}

/// `m · w`  (R_2 matrix, R_2 vector → R_2 vector). Used by the MIMD and
/// general-circuit reductions, whose packed matrices are `R_2`-valued
/// (each entry already in `Im(φ)`).
pub fn matvec_gr128_dense<L: Limb>(m: &SparseMatrix<Gr128<L>>, w: &[Gr128<L>]) -> Vec<Gr128<L>> {
    assert_eq!(w.len(), m.cols);
    let mut out = vec![Gr128::<L>::ZERO; m.rows];
    for &(r, c, v) in &m.triples {
        out[r] = out[r] + v * w[c];
    }
    out
}
