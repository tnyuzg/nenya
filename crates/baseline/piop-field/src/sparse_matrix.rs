//! Sparse matrix with BB4 entries.
//!
//! Field-side mirror of [`piop_ring::SparseMatrix`]. Entries here are full
//! BB4 elements — there is no "small operand" structure to exploit, so the
//! linear-check matvec just calls `eq_row[y] * matrix[y, x]`. The shape of
//! the matvec inner loop is identical to the ring side; the only thing that
//! differs is the per-entry mul cost.

use crate::BB4;

#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub entries: Vec<(usize, usize, BB4)>,
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

    pub fn push(&mut self, row: usize, col: usize, val: BB4) {
        assert!(row < self.num_rows(), "row out of range");
        assert!(col < self.num_cols(), "col out of range");
        self.entries.push((row, col, val));
    }
}
