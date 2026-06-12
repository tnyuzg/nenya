//! Sparse matrix with `R₀ = GR(2³², 1) = u32` entries.
//!
//! The PIOP linear check is `∑_h A(r, h) · w(h)` where `A` is an R1CS constraint
//! matrix. R1CS matrices are sparse (~3 nonzeros per row in typical Rank-1
//! constraint systems) and their entries are *small* (the constraint
//! coefficients fit in u32). The sumcheck challenge `r` is still drawn from
//! `T_2` for soundness, but the matrix itself stays in `R₀`.
//!
//! That u32-ness is exploited exactly once — during the matvec
//! `eq_row · A → A_row` step in [`crate::linear_check_simd`] — by calling
//! [`galois_ring::GR32_128::mul_by_gr1`] on each `(eq_row[y], A[y, x])` pair
//! instead of the generic schoolbook. Once the matvec is done, `A_row` is a
//! plain `R₂`-valued vector and the rest of the sumcheck has no use for the
//! u32 structure.
//!
//! Storage is the simplest thing that works: a flat list of `(row, col, val)`
//! triples. No CSR/CSC; the matvec walks the triples in order and accumulates
//! into `A_row[col]`. At the bench size (`n = 15`, ~96K nonzeros) the cache
//! cost is negligible compared to the per-entry `mul_by_gr1`.

use galois_ring::GR32_1;

#[derive(Clone, Debug)]
pub struct SparseMatrix {
    /// `2^log_num_rows` rows, `2^log_num_cols` cols.
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    /// Flat triples. No ordering requirement; duplicates are summed.
    pub entries: Vec<(usize, usize, GR32_1)>,
}

impl SparseMatrix {
    pub fn new(log_num_rows: usize, log_num_cols: usize) -> Self {
        Self {
            log_num_rows,
            log_num_cols,
            entries: Vec::new(),
        }
    }

    pub fn num_rows(&self) -> usize {
        1usize << self.log_num_rows
    }

    pub fn num_cols(&self) -> usize {
        1usize << self.log_num_cols
    }

    /// Push one entry. Asserts the indices are in range.
    pub fn push(&mut self, row: usize, col: usize, val: GR32_1) {
        assert!(row < self.num_rows(), "row out of range");
        assert!(col < self.num_cols(), "col out of range");
        self.entries.push((row, col, val));
    }
}
