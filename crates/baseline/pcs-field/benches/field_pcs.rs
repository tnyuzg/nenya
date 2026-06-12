//! **Field baseline for the tensor PCS** — same Brakedown-style construction
//! and the *same RAA fast-path code* as `pcs.rs`, but committing to
//! `BabyBear` (BB) field elements instead of the 32-u32 *packed* ring
//! representation. This is the apples-to-apples answer to "is the ring PCS
//! actually faster than a field PCS, or was that just the 32x-larger
//! commitment?".
//!
//! ## Why this is the fair comparison
//!
//! The ring PCS commits `N_ring` *packed* entries, each holding 32 base
//! values (u32 ~ one BB), so it covers `32 * N_ring` base values. A field
//! SNARK proving the same statement has no phi-packing and must commit all
//! `32 * N_ring` base values directly. So:
//!
//! - ring operating point  : `N_ring  = 2^15` packed (k1=128,  k2=256)   -> 2^20 base values
//! - field fair comparison : `N_field = 2^20` BB     (k1=1024, k2=1024)  -> 2^20 base values
//!
//! Same inverse-rate `q`, same query count `t`. The element type is the only
//! deliberate difference: BB (4 bytes, mod-p arithmetic) vs packed `[u32;32]`
//! (128 bytes, wrapping-u32 arithmetic). RAA permutations/diagonals,
//! Merkle layout (blake3 over columns), and Fiat-Shamir are structurally
//! identical to `pcs.rs`.
//!
//! Run: `cargo bench -p pcs-field --bench field_pcs`.

use blake3::Hasher;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use field::{AbstractExtensionField, AbstractField, BB, BB4};
use p3_field::PrimeField32;
use pcs_ring::merkle::{verify_path, Hash, MerkleTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const P: u32 = 0x7800_0001; // BabyBear modulus (2^31 - 2^27 + 1)

fn rand_bb(rng: &mut StdRng) -> BB {
    BB::from_canonical_u32(rng.gen::<u32>() % P)
}

fn rand_bb4(rng: &mut StdRng) -> BB4 {
    let b: [BB; 4] = [rand_bb(rng), rand_bb(rng), rand_bb(rng), rand_bb(rng)];
    <BB4 as AbstractExtensionField<BB>>::from_base_slice(&b)
}

/// Lift a base element into the extension (BB -> BB4).
#[inline]
fn lift(b: BB) -> BB4 {
    <BB4 as AbstractExtensionField<BB>>::from_base(b)
}

/// Multilinear eq-table, MSB-first in `z` (matches `piop_ring::build_eq_table`
/// convention so the row/column split is an outer product).
fn eq_table(z: &[BB4]) -> Vec<BB4> {
    let mut t = vec![BB4::one()];
    for &zi in z {
        let mut nt = Vec::with_capacity(t.len() * 2);
        let one_minus = BB4::one() - zi;
        for &e in &t {
            nt.push(e * one_minus);
            nt.push(e * zi);
        }
        t = nt;
    }
    t
}

// ---------------------------------------------------------------------------
// RAA code over BB (mirror of raa.rs, element type = BabyBear)
// ---------------------------------------------------------------------------

struct FieldRaa {
    k: usize,
    q: usize,
    n: usize,
    perm1: Vec<u32>,
    perm2: Vec<u32>,
    diag1: Vec<BB>,
    diag2: Vec<BB>,
}

impl FieldRaa {
    fn setup(k: usize, q: usize, seed: u64) -> Self {
        assert!(k > 0 && q >= 2);
        let n = k * q;
        let mut rng = StdRng::seed_from_u64(seed);
        let perm1 = sample_perm(&mut rng, n);
        let perm2 = sample_perm(&mut rng, n);
        // Diagonals are units (nonzero); matches raa.rs which forces odd u32s.
        let diag1 = (0..n).map(|_| nonzero_bb(&mut rng)).collect();
        let diag2 = (0..n).map(|_| nonzero_bb(&mut rng)).collect();
        Self {
            k,
            q,
            n,
            perm1,
            perm2,
            diag1,
            diag2,
        }
    }

    /// Encode a length-`k` BB message into a length-`n` BB codeword
    /// (committer side).
    fn encode_base(&self, msg: &[BB]) -> Vec<BB> {
        debug_assert_eq!(msg.len(), self.k);
        let n = self.n;
        let mut y = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..self.q {
                y.push(m);
            }
        }
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            t.push(y[self.perm1[i] as usize] * self.diag1[i]);
        }
        for i in 1..n {
            t[i] = t[i] + t[i - 1];
        }
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            z.push(t[self.perm2[i] as usize] * self.diag2[i]);
        }
        for i in 1..n {
            z[i] = z[i] + z[i - 1];
        }
        z
    }

    /// Encode a length-`k` BB4 message (verifier side: `m̄` is in the
    /// extension). Same perms/diagonals; diagonal scalars lifted to BB4.
    fn encode_ext(&self, msg: &[BB4]) -> Vec<BB4> {
        debug_assert_eq!(msg.len(), self.k);
        let n = self.n;
        let mut y = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..self.q {
                y.push(m);
            }
        }
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            t.push(y[self.perm1[i] as usize] * lift(self.diag1[i]));
        }
        for i in 1..n {
            t[i] = t[i] + t[i - 1];
        }
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            z.push(t[self.perm2[i] as usize] * lift(self.diag2[i]));
        }
        for i in 1..n {
            z[i] = z[i] + z[i - 1];
        }
        z
    }
}

fn sample_perm(rng: &mut StdRng, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        p.swap(i, j);
    }
    p
}

fn nonzero_bb(rng: &mut StdRng) -> BB {
    let mut x = rng.gen::<u32>() % P;
    if x == 0 {
        x = 1;
    }
    BB::from_canonical_u32(x)
}

// ---------------------------------------------------------------------------
// Field tensor PCS (mirror of pcs.rs)
// ---------------------------------------------------------------------------

struct FieldPcs {
    raa: FieldRaa,
    k1: usize,
    k2: usize,
    ell: usize,
}

struct FieldCommitment {
    root: Hash,
    /// Encoded matrix, row-major `k1 x n` over BB.
    encoded: Vec<BB>,
    /// Original message, row-major `k1 x k2` over BB.
    message: Vec<BB>,
    tree: MerkleTree,
}

struct FieldOpening {
    mbar: Vec<BB4>,
    query_indices: Vec<u32>,
    query_columns: Vec<Vec<BB>>,
    query_paths: Vec<Vec<Hash>>,
}

impl FieldOpening {
    fn size_bytes(&self) -> usize {
        let mbar = self.mbar.len() * 16; // BB4 = 4 * u32
        let cols: usize = self.query_columns.iter().map(|c| c.len() * 4).sum();
        let paths: usize = self.query_paths.iter().map(|p| p.len() * 32).sum();
        let idx = self.query_indices.len() * 4;
        mbar + cols + paths + idx
    }
}

impl FieldPcs {
    fn new(k1: usize, k2: usize, q: usize, ell: usize, seed: u64) -> Self {
        assert!(k1.is_power_of_two() && k2.is_power_of_two());
        Self {
            raa: FieldRaa::setup(k2, q, seed),
            k1,
            k2,
            ell,
        }
    }

    fn num_vars(&self) -> usize {
        (self.k1 * self.k2).trailing_zeros() as usize
    }

    fn split_eq(&self, z: &[BB4]) -> (Vec<BB4>, Vec<BB4>) {
        let log_k1 = self.k1.trailing_zeros() as usize;
        (eq_table(&z[..log_k1]), eq_table(&z[log_k1..]))
    }

    fn commit(&self, evals: &[BB]) -> FieldCommitment {
        assert_eq!(evals.len(), self.k1 * self.k2);
        let n = self.raa.n;
        let mut encoded = vec![BB::zero(); self.k1 * n];
        for i in 0..self.k1 {
            let row = &evals[i * self.k2..(i + 1) * self.k2];
            let c = self.raa.encode_base(row);
            encoded[i * n..(i + 1) * n].copy_from_slice(&c);
        }
        let leaves: Vec<Hash> = (0..n)
            .map(|j| hash_bb_column_strided(&encoded, self.k1, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        FieldCommitment {
            root,
            encoded,
            message: evals.to_vec(),
            tree,
        }
    }

    fn open(&self, comm: &FieldCommitment, z: &[BB4]) -> (BB4, FieldOpening) {
        let (e0, e1) = self.split_eq(z);
        let n = self.raa.n;

        // m̄ = Σ_i e0[i] · M[i, :]   ∈ BB4^{k2}
        let mut mbar = vec![BB4::zero(); self.k2];
        for i in 0..self.k1 {
            let e = e0[i];
            let row = &comm.message[i * self.k2..(i + 1) * self.k2];
            for j in 0..self.k2 {
                mbar[j] = mbar[j] + e * lift(row[j]);
            }
        }
        // f(z) = ⟨m̄, e1⟩
        let mut y = BB4::zero();
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);
        let mut query_columns = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<BB> = (0..self.k1).map(|i| comm.encoded[i * n + j]).collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }
        (
            y,
            FieldOpening {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    fn verify(&self, root: Hash, z: &[BB4], y_claimed: BB4, proof: &FieldOpening) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.raa.n;
        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y = BB4::zero();
        for j in 0..self.k2 {
            y = y + proof.mbar[j] * e1[j];
        }
        if y != y_claimed {
            return false;
        }
        let expected = derive_query_indices(root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }
        // ū = Enc(m̄) over BB4.
        let ubar = self.raa.encode_ext(&proof.mbar);
        for q in 0..self.ell {
            let j = proof.query_indices[q] as usize;
            let col = &proof.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }
            // acc_j = Σ_i e0[i] · col[i]   (no 32-lane basis loop — unpacked)
            let mut acc = BB4::zero();
            for i in 0..self.k1 {
                acc = acc + e0[i] * lift(col[i]);
            }
            if acc != ubar[j] {
                return false;
            }
            let leaf = hash_bb_column(col);
            if !verify_path(leaf, j, &proof.query_paths[q], root) {
                return false;
            }
        }
        true
    }
}

// blake3 over a strided column of the row-major encoded matrix.
fn hash_bb_column_strided(flat: &[BB], k1: usize, n: usize, j: usize) -> Hash {
    let mut h = Hasher::new();
    for i in 0..k1 {
        h.update(&flat[i * n + j].as_canonical_u32().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn hash_bb_column(col: &[BB]) -> Hash {
    let mut h = Hasher::new();
    for e in col {
        h.update(&e.as_canonical_u32().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn derive_query_indices(root: Hash, mbar: &[BB4], n: usize, ell: usize) -> Vec<u32> {
    let mut h = Hasher::new();
    h.update(b"field-pcs-ring/fs/queries");
    h.update(&root);
    for elem in mbar {
        for c in <BB4 as AbstractExtensionField<BB>>::as_base_slice(elem) {
            h.update(&c.as_canonical_u32().to_le_bytes());
        }
    }
    let mut xof = h.finalize_xof();
    let mut out = Vec::with_capacity(ell);
    let n32 = n as u32;
    let mask = (n.next_power_of_two() - 1) as u32;
    let mut buf = [0u8; 4];
    while out.len() < ell {
        xof.fill(&mut buf);
        let v = u32::from_le_bytes(buf) & mask;
        if v < n32 {
            out.push(v);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Self-check (runs once; turns the bench into its own sanity test)
// ---------------------------------------------------------------------------

fn eval_mle_ref(evals: &[BB], z: &[BB4]) -> BB4 {
    let eq = eq_table(z);
    assert_eq!(eq.len(), evals.len());
    let mut acc = BB4::zero();
    for i in 0..evals.len() {
        acc = acc + eq[i] * lift(evals[i]);
    }
    acc
}

fn self_check() {
    let mut rng = StdRng::seed_from_u64(0xABCD_1234);
    let pcs = FieldPcs::new(8, 8, 4, 16, 0xDEAD_BEEF);
    let evals: Vec<BB> = (0..64).map(|_| rand_bb(&mut rng)).collect();
    let z: Vec<BB4> = (0..pcs.num_vars()).map(|_| rand_bb4(&mut rng)).collect();

    let comm = pcs.commit(&evals);
    let (y, proof) = pcs.open(&comm, &z);
    assert_eq!(
        y,
        eval_mle_ref(&evals, &z),
        "open returned wrong evaluation"
    );
    assert!(
        pcs.verify(comm.root, &z, y, &proof),
        "verify rejected a valid proof"
    );
    assert!(
        !pcs.verify(comm.root, &z, y + BB4::one(), &proof),
        "verify accepted a tampered evaluation"
    );
    eprintln!("[field_pcs self-check] OK (eval correct, verify accepts, tamper rejects)");
}

// ---------------------------------------------------------------------------
// Criterion
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct Shape {
    k1: usize,
    k2: usize,
    label: &'static str,
}

fn bench_shape(c: &mut Criterion, group_name: &str, q: usize, t: usize, shape: &Shape) {
    let mut rng = StdRng::seed_from_u64(1);
    let pcs = FieldPcs::new(shape.k1, shape.k2, q, t, 0xC0DE);
    let evals: Vec<BB> = (0..shape.k1 * shape.k2)
        .map(|_| rand_bb(&mut rng))
        .collect();
    let z: Vec<BB4> = (0..pcs.num_vars()).map(|_| rand_bb4(&mut rng)).collect();

    let comm = pcs.commit(&evals);
    let (_y, proof) = pcs.open(&comm, &z);
    eprintln!(
        "[{}] N={}, k1={}, k2={}, n={}, t={} -> proof = {:.2} MB",
        shape.label,
        shape.k1 * shape.k2,
        shape.k1,
        shape.k2,
        pcs.raa.n,
        t,
        proof.size_bytes() as f64 / (1024.0 * 1024.0),
    );

    let mut g = c.benchmark_group(group_name);
    g.sample_size(10);
    g.bench_with_input(BenchmarkId::new("commit", shape.label), shape, |b, _| {
        b.iter(|| pcs.commit(&evals))
    });
    g.bench_with_input(BenchmarkId::new("open", shape.label), shape, |b, _| {
        b.iter(|| pcs.open(&comm, &z))
    });
    g.bench_with_input(BenchmarkId::new("verify", shape.label), shape, |b, _| {
        let (y, proof) = pcs.open(&comm, &z);
        b.iter(|| pcs.verify(comm.root, &z, y, &proof))
    });
    g.finish();
}

// Field-side sizes. The headline fair comparison is N=2^20 (k1=1024,k2=1024)
// vs the ring's N=2^15 packed (k1=128,k2=256) — both cover 2^20 base values.
fn shapes() -> &'static [Shape] {
    &[
        Shape {
            k1: 128,
            k2: 256,
            label: "N=2^15 (k1=128, k2=256)",
        },
        Shape {
            k1: 512,
            k2: 512,
            label: "N=2^18 (k1=512, k2=512)",
        },
        Shape {
            k1: 1024,
            k2: 1024,
            label: "N=2^20 (k1=1024,k2=1024)",
        },
    ]
}

fn bench_field_opt(c: &mut Criterion) {
    self_check();
    for shape in shapes() {
        bench_shape(c, "field-opt", 4, 487, shape);
    }
}

fn bench_field_con(c: &mut Criterion) {
    for shape in shapes() {
        bench_shape(c, "field-con", 6, 842, shape);
    }
}

criterion_group!(benches, bench_field_opt, bench_field_con);
criterion_main!(benches);
