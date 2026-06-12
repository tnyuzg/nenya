//! Cross-prover grid benchmark: {field, ring-u8, ring-u32, ring-u64} × {SIMD, MIMD, general},
//! each prover's cost split into (reduction, piop, pcs, total). Parallel (all cores).
//!
//! Usage: `snark_grid [log_scalar]`  (default 25 ⇒ a 2^25-constraint R1CS).
//!
//! Sizing: all 9 cells prove the same `2^log_scalar` *scalar* constraints.
//!   - SIMD / MIMD: 32 instances of `2^(log_scalar-5)` each.
//!   - general:     one circuit of `2^log_scalar` (log_blocks = log_scalar-5).
//!   - field:       one R1CS of `2^log_scalar` (identical across the 3 columns,
//!                  since the field doesn't see the SIMD/MIMD/general structure).
//!
//! Buckets: reduction = witness→packed reduce; piop = matvec + sumchecks +
//! row-build; pcs = commit + opens. `total` is measured directly (so
//! red+piop+pcs ≤ total — the remainder is transcript/eq/glue).

use field::{AbstractField, BB};
use galois_ring::{Gr1, Limb};
use reduction::constants::HasPackingConstants;
use pcs_field::FieldPcs;
use pcs_ring::Pcs;
use reduction::{
    matrix::SparseMatrix, pack_general_matrices, pack_mimd_matrices, GeneralR1csInstance,
    MimdR1csInstance, SimdR1csInstance,
};
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{prove_field_with_timing, FieldSnarkParams};
use snark_ring::{
    prove_general_with_timing, prove_mimd_with_timing, prove_simd_with_timing, GeneralSnarkParams,
    MimdSnarkParams, R2Pcs, SimdSnarkParams,
};

#[derive(Clone, Copy, Default)]
struct Cost {
    red: f64,
    piop: f64,
    pcs: f64,
    total: f64,
}

const ELL: usize = 16;

fn dims(log: usize) -> (usize, usize) {
    let half = log / 2;
    (1 << half, 1 << (log - half))
}

// ----- ring instance builders (x·x = y), generic over L -----

fn xy_witness<L: Limb>(m: usize, seed: usize) -> Vec<Gr1<L>> {
    let k = (m - 1) / 2;
    let mut w = vec![Gr1::<L>::ZERO; m];
    w[0] = Gr1::<L>::ONE;
    for i in 0..k {
        let x = Gr1(L::from_u64((((seed * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1)));
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    w
}

fn xy_matrices<L: Limb>(n: usize, m: usize) -> [SparseMatrix<Gr1<L>>; 3] {
    let k = (m - 1) / 2;
    let mut a = SparseMatrix::new(n, m);
    let mut b = SparseMatrix::new(n, m);
    let mut c = SparseMatrix::new(n, m);
    for i in 0..k {
        a.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        b.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        c.push(i, 2 + 2 * i, Gr1::<L>::ONE);
    }
    [a, b, c]
}

fn run_simd<L: HasPackingConstants>(log: usize) -> Cost {
    let n = 1usize << log;
    let [a, b, c] = xy_matrices::<L>(n, n);
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|k| xy_witness::<L>(n, k));
    let simd = SimdR1csInstance { a, b, c, witnesses };
    let (k1, k2) = dims(log);
    let params = SimdSnarkParams {
        log_n: log,
        log_m: log,
        pcs_w: Pcs::<L>::new(k1, k2, 4, ELL, 0xDEAD),
        pcs_q: R2Pcs::<L>::new(k1, k2, 4, ELL, 0xCAFE),
    };
    let (_p, t) = prove_simd_with_timing(&simd, &params);
    Cost {
        red: t.reduce_ms,
        piop: t.matvec_abc_ms
            + t.eq_r_ms
            + t.mult_sumcheck_ms
            + t.linear_arow_ms
            + t.linear_sumcheck_ms,
        pcs: t.commit_w_encode_ms
            + t.commit_w_merkle_ms
            + t.commit_e_encode_ms
            + t.commit_e_merkle_ms
            + t.open_w_ms
            + t.open_e_ms,
        total: t.total_ms,
    }
}

#[allow(dead_code)] // superseded by examples/exp_cell.rs; kept for ad-hoc MIMD grid runs
fn run_mimd<L: HasPackingConstants>(log: usize) -> Cost {
    let n = 1usize << log;
    let a: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[0].clone());
    let b: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[1].clone());
    let c: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[2].clone());
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|k| xy_witness::<L>(n, k));
    let mimd = MimdR1csInstance {
        a_per_instance: a,
        b_per_instance: b,
        c_per_instance: c,
        witnesses,
    };
    let (k1, k2) = dims(log);
    let params = MimdSnarkParams {
        log_n: log,
        log_m: log,
        pcs_w: Pcs::<L>::new(k1, k2, 4, ELL, 0xDEAD),
        pcs_abc: Pcs::<L>::new(k1, k2, 4, ELL, 0xABCD),
        pcs_e: R2Pcs::<L>::new(k1, k2, 4, ELL, 0xCAFE),
        mats: pack_mimd_matrices(&mimd.a_per_instance, &mimd.b_per_instance, &mimd.c_per_instance),
    };
    let (_p, t) = prove_mimd_with_timing(&mimd.witnesses, &params);
    Cost {
        red: t.reduce_ms,
        piop: t.matvec_ms + t.mult_sumcheck_ms + t.linear_ms,
        pcs: t.commit_ms + t.opens_ms,
        total: t.total_ms,
    }
}

fn run_general<L: HasPackingConstants>(log_blocks: usize) -> Cost {
    let n = 32usize << log_blocks; // = 2^(log_blocks+5)
    let [a, b, c] = xy_matrices::<L>(n, n);
    let w = xy_witness::<L>(n, 1);
    let g = GeneralR1csInstance { a, b, c, w };
    let (k1, k2) = dims(log_blocks);
    let params = GeneralSnarkParams {
        log_blocks,
        pcs_w: Pcs::<L>::new(k1, k2, 4, ELL, 0xDEAD),
        pcs_abc: Pcs::<L>::new(k1, k2, 4, ELL, 0xABCD),
        pcs_e: R2Pcs::<L>::new(k1, k2, 4, ELL, 0xCAFE),
        diag: pack_general_matrices(&g.a, &g.b, &g.c),
    };
    let (_p, t) = prove_general_with_timing(&g.w, &params);
    Cost {
        red: t.reduce_ms,
        piop: t.matvec_ms + t.mult_sumcheck_ms + t.linear_ms,
        pcs: t.commit_ms + t.opens_ms,
        total: t.total_ms,
    }
}

fn run_field(log: usize) -> Cost {
    let n = 1usize << log;
    let k = (n - 1) / 2;
    let mut a = SparseMatrixBB::new(n, n);
    let mut b = SparseMatrixBB::new(n, n);
    let mut c = SparseMatrixBB::new(n, n);
    for i in 0..k {
        a.push(i, 1 + 2 * i, BB::one());
        b.push(i, 1 + 2 * i, BB::one());
        c.push(i, 2 + 2 * i, BB::one());
    }
    let mut w = vec![BB::zero(); n];
    w[0] = BB::one();
    for i in 0..k {
        let x = BB::from_canonical_u32(((i as u32) ^ 0x85EB).wrapping_add(1) % 0x7800_0001);
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    let inst = FieldR1csInstance { a, b, c, w };
    let (k1, k2) = dims(log);
    let params = FieldSnarkParams {
        log_n: log,
        log_m: log,
        pcs_w: FieldPcs::new(k1, k2, 4, ELL, 0xDEAD),
    };
    let (_p, t) = prove_field_with_timing(&inst, &params);
    Cost {
        red: 0.0,
        piop: t.matvec_abc_ms
            + t.eq_r_ms
            + t.mult_sumcheck_ms
            + t.linear_arow_ms
            + t.linear_sumcheck_ms,
        pcs: t.commit_w_ms + t.open_w_ms,
        total: t.total_ms,
    }
}

fn print_bucket(name: &str, get: impl Fn(&Cost) -> f64, grid: &[[Cost; 3]; 4]) {
    let rows = ["field", "u8", "u32", "u64"];
    println!("\n=== {name} (ms) ===");
    println!("  {:<8} {:>12} {:>12} {:>12}", "", "SIMD", "MIMD", "general");
    for (ri, row) in grid.iter().enumerate() {
        println!(
            "  {:<8} {:>12.1} {:>12.1} {:>12.1}",
            rows[ri],
            get(&row[0]),
            get(&row[1]),
            get(&row[2])
        );
    }
}

fn main() {
    let log_scalar: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let ring = log_scalar - 5; // SIMD/MIMD log_size; general log_blocks
    // Optional 2nd arg = thread count (default = all cores; pass 1 for serial).
    let nthreads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    println!(
        "Grid @ 2^{log_scalar} scalar constraints, parallel ({nthreads} threads). \
         SIMD/MIMD log_size={ring}, general log_blocks={ring}, field log={log_scalar}."
    );

    let f = pool.install(|| run_field(log_scalar)); // identical across the 3 columns
    let mut grid = [[Cost::default(); 3]; 4];
    grid[0] = [f, f, f];
    grid[1] = [
        pool.install(|| run_simd::<u8>(ring)),
        Cost::default(),
        pool.install(|| run_general::<u8>(ring)),
    ];
    grid[2] = [
        pool.install(|| run_simd::<u32>(ring)),
        Cost::default(),
        pool.install(|| run_general::<u32>(ring)),
    ];

    print_bucket("reduction", |c| c.red, &grid);
    print_bucket("piop", |c| c.piop, &grid);
    print_bucket("pcs", |c| c.pcs, &grid);
    print_bucket("TOTAL prove", |c| c.total, &grid);

    // Machine-readable: `CSV,log_scalar,row,col,red,piop,pcs,total` (ms).
    let rows = ["field", "u8", "u32", "u64"];
    let cols = ["simd", "mimd", "general"];
    for (ri, row) in grid.iter().enumerate() {
        for (ci, c) in row.iter().enumerate() {
            println!(
                "CSV,{log_scalar},{},{},{:.3},{:.3},{:.3},{:.3}",
                rows[ri], cols[ci], c.red, c.piop, c.pcs, c.total
            );
        }
    }
}
