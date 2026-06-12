//! Binary Merkle tree over the columns of an encoded matrix.
//!
//! Leaves are `blake3` over the `k_1` packed-u32 entries of column `j` of
//! the encoded matrix `M̂`. Each entry is 32 u32 words = 128 bytes (vs 512
//! bytes for a full `GR32_128`). Internal nodes are `blake3(left || right)`.
//!
//! `n` (the column count) is required to be a power of two.

use blake3::Hasher;
use galois_ring::Limb;
use rayon::prelude::*;

pub type Hash = [u8; 32];

/// Hash a single column of `k1` packed entries into one Merkle leaf. Reads
/// column-wise from a row-major matrix `M̂` of shape `k1 × n` stored in `flat`
/// (length `k1 * n`, each entry `[L; 32]`). Each limb is serialized as its low
/// `BITS/8` bytes, so the leaf is width-correct (32 bytes/entry for `u8`, 128
/// for `u32`).
pub fn hash_column<L: Limb>(flat: &[[L; 32]], k1: usize, n: usize, j: usize) -> Hash {
    let nbytes = (L::BITS / 8) as usize;
    let mut h = Hasher::new();
    for i in 0..k1 {
        let entry = &flat[i * n + j];
        for w in entry.iter() {
            h.update(&w.to_u64().to_le_bytes()[..nbytes]);
        }
    }
    *h.finalize().as_bytes()
}

/// Hash a stand-alone column (the `k_1` entries already gathered into a
/// `Vec`). Used verifier-side to re-derive a leaf from a proof's
/// `query_columns[q]`.
pub fn hash_packed_column<L: Limb>(col: &[[L; 32]]) -> Hash {
    let nbytes = (L::BITS / 8) as usize;
    let mut h = Hasher::new();
    for entry in col {
        for w in entry.iter() {
            h.update(&w.to_u64().to_le_bytes()[..nbytes]);
        }
    }
    *h.finalize().as_bytes()
}

/// A binary Merkle tree. `nodes` is the level-order layout:
/// `nodes[0..n]` = leaves, `nodes[n..n+n/2]` = level above, etc.
#[derive(Clone)]
pub struct MerkleTree {
    pub n: usize,
    pub nodes: Vec<Hash>,
}

impl MerkleTree {
    pub fn build(mut leaves: Vec<Hash>) -> Self {
        assert!(!leaves.is_empty());
        let n = leaves.len().next_power_of_two();
        leaves.resize(n, [0u8; 32]);
        let total = 2 * n - 1;
        let mut nodes = vec![[0u8; 32]; total];
        nodes[..n].copy_from_slice(&leaves);

        let mut level_start = 0;
        let mut level_len = n;
        while level_len > 1 {
            let next_start = level_start + level_len;
            // The current level [level_start, next_start) is fully built and the
            // next level [next_start, next_start + level_len/2) is disjoint from
            // it. Split at `next_start` so we can read the lower half while
            // writing the upper half in parallel; the per-pair hashes are
            // independent, so the result is identical to the serial fold.
            let (lower, upper) = nodes.split_at_mut(next_start);
            let cur = &lower[level_start..next_start];
            upper[..level_len / 2]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| {
                    let mut h = Hasher::new();
                    h.update(&cur[2 * i]);
                    h.update(&cur[2 * i + 1]);
                    *out = *h.finalize().as_bytes();
                });
            level_start = next_start;
            level_len /= 2;
        }
        Self { n, nodes }
    }

    pub fn root(&self) -> Hash {
        *self.nodes.last().unwrap()
    }

    /// Open leaf index `j`: return the sibling hashes from leaf to just-below-root.
    pub fn open(&self, j: usize) -> Vec<Hash> {
        assert!(j < self.n);
        let mut path = Vec::with_capacity(self.n.trailing_zeros() as usize);
        let mut idx = j;
        let mut level_start = 0;
        let mut level_len = self.n;
        while level_len > 1 {
            let sibling = idx ^ 1;
            path.push(self.nodes[level_start + sibling]);
            idx /= 2;
            level_start += level_len;
            level_len /= 2;
        }
        path
    }
}

/// Recompute the Merkle root from a leaf, its index, and the sibling path.
pub fn verify_path(leaf: Hash, mut idx: usize, path: &[Hash], root: Hash) -> bool {
    let mut cur = leaf;
    for sib in path {
        let mut h = Hasher::new();
        if idx & 1 == 0 {
            h.update(&cur);
            h.update(sib);
        } else {
            h.update(sib);
            h.update(&cur);
        }
        cur = *h.finalize().as_bytes();
        idx /= 2;
    }
    cur == root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_verify_roundtrip() {
        let leaves: Vec<Hash> = (0..16u8).map(|b| [b; 32]).collect();
        let tree = MerkleTree::build(leaves.clone());
        let root = tree.root();
        for j in 0..16 {
            let path = tree.open(j);
            assert!(verify_path(leaves[j], j, &path, root));
            assert!(!verify_path(leaves[j], j ^ 1, &path, root));
        }
    }
}
