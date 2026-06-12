//! Field R1CS instance over BabyBear: `(A·w) ∘ (B·w) = C·w`, all over `BB`.

use field::{AbstractField, BB};

/// Coordinate-format sparse matrix with `BB` entries. (R1CS structure matrices
/// are over the base field; the witness is too.)
#[derive(Clone, Debug)]
pub struct SparseMatrixBB {
    pub rows: usize,
    pub cols: usize,
    pub triples: Vec<(usize, usize, BB)>,
}

impl SparseMatrixBB {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triples: vec![],
        }
    }
    pub fn push(&mut self, r: usize, c: usize, v: BB) {
        debug_assert!(r < self.rows && c < self.cols);
        self.triples.push((r, c, v));
    }
}

/// `m · w` over `BB`.
pub fn matvec_bb(m: &SparseMatrixBB, w: &[BB]) -> Vec<BB> {
    assert_eq!(w.len(), m.cols);
    let mut out = vec![BB::zero(); m.rows];
    for &(r, c, v) in &m.triples {
        out[r] += v * w[c];
    }
    out
}

/// A field R1CS instance: square `n × m` matrices, witness of length `m`.
#[derive(Clone, Debug)]
pub struct FieldR1csInstance {
    pub a: SparseMatrixBB,
    pub b: SparseMatrixBB,
    pub c: SparseMatrixBB,
    pub w: Vec<BB>,
}

impl FieldR1csInstance {
    /// True iff `(A·w) ∘ (B·w) == C·w`.
    pub fn is_satisfied(&self) -> bool {
        let aw = matvec_bb(&self.a, &self.w);
        let bw = matvec_bb(&self.b, &self.w);
        let cw = matvec_bb(&self.c, &self.w);
        for i in 0..aw.len() {
            if aw[i] * bw[i] != cw[i] {
                return false;
            }
        }
        true
    }
}
