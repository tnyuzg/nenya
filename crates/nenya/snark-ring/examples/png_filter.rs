//! PNG-filter application: prove that a batch of grayscale images was correctly
//! PNG-filtered (Sub / Up), comparing the **ring-u8 SIMD** SNARK against the
//! **field (BabyBear)** baseline on the *same* batch.
//!
//! ## Why PNG filters are a `Z_{2^8}`-native workload
//!
//! PNG row filters (<https://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html>)
//! predict each byte from a neighbour and store the residue **mod 256**:
//!
//! - **Sub**  (filter type 1): `Filt[x] = Orig[x] − Orig[left]   (mod 256)`
//! - **Up**   (filter type 2): `Filt[x] = Orig[x] − Orig[above]  (mod 256)`
//!
//! Reconstruction is the inverse recurrence `Orig[x] = Filt[x] + pred (mod 256)`.
//! This is *exactly* arithmetic in `R₀ = Z_{2^8}`. The relation we prove (private
//! raw image `Orig`, public filtered bytes `Filt`) is the single constraint
//!
//! ```text
//!     Orig[x] = Filt[x] + pred[x]        (mod 256)
//! ```
//!
//! ## The two circuits — where the field pays an "emulation tax"
//!
//! * **ring-u8 (native).** Every witness element already *is* a byte and ring
//!   arithmetic already wraps mod 256, so each pixel is **one linear constraint**
//!   `Orig[x] = Filt[x] + pred`. No range checks, no wraparound bookkeeping.
//!   The whole circuit has **zero multiplications**. 32 images pack into the 32
//!   SIMD slots (shared matrices, per-image witness).
//!
//! * **field (BabyBear ≈ 2³¹).** A prime field does *not* wrap at 256, and a
//!   malicious prover could use out-of-range "bytes". To *faithfully* compute the
//!   mod-256 semantics the field must, **per private byte**:
//!     1. recurrence with a borrow:  `Orig + 256·borrow = Filt + pred`   (1 row)
//!     2. borrow booleanity:         `borrow·borrow = borrow`            (1 row)
//!     3. 8-bit range decomposition: `Orig = Σ 2^j·bit_j`                (1 row)
//!     4. bit booleanity ×8:         `bit_j·bit_j = bit_j`               (8 rows)
//!   → **11 constraints/byte** (8 of them genuine multiplications). The public
//!   `Filt` bytes are *not* range-checked here (the verifier holds them), so 11
//!   is a conservative, field-favourable count.
//!
//! Both prove the identical batch of images; we hold the **image data fixed** and
//! compare prover time. PCS queries `ell = 16` on both sides (same methodology as
//! `BENCHMARKS.md`; at equal `ell` the field is actually *under*-secured).
//!
//! Run:  `cargo run --release --example png_filter -- [log_bytes] [nthreads]`
//!   `log_bytes` = log₂(total raw bytes in the batch), default 18 (256 KiB).
//!   The batch is 32 images of `2^(log_bytes−5)` bytes each.

use field::{AbstractField, BB};
use galois_ring::{Gr1, Limb};
use reduction::{matrix::SparseMatrix, SimdR1csInstance};
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{prove_field_with_timing, verify_field, FieldSnarkParams};
use snark_ring::{prove_simd_with_timing, verify_simd, SimdSnarkParams};

const SLOTS: usize = 32;

/// Smallest `k` with `2^k ≥ x`.
fn ceil_log2(x: usize) -> usize {
    let mut k = 0;
    while (1usize << k) < x {
        k += 1;
    }
    k
}

/// Deterministic pseudo-random byte (splitmix64-style; no RNG dependency).
fn rnd_byte(seed: u64) -> u8 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x & 0xFF) as u8
}

/// Image geometry for `b` bytes: `W = min(256, b)` columns, `H = b/W` rows.
fn image_dims(b: usize) -> (usize, usize) {
    let w = b.min(256);
    (b / w, w)
}

/// Predecessor map for one image: row `r` uses **Sub** (left neighbour) when `r`
/// is even, **Up** (neighbour above) when odd. Identical across all 32 images, so
/// the SIMD matrices are shared. `None` = no predecessor (border ⇒ pred 0).
fn predecessors(h: usize, w: usize) -> Vec<Option<usize>> {
    let mut p = vec![None; h * w];
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            p[i] = if r % 2 == 0 {
                if c > 0 {
                    Some(i - 1) // Sub: left
                } else {
                    None
                }
            } else if r > 0 {
                Some(i - w) // Up: above
            } else {
                None
            };
        }
    }
    p
}

/// Generate one image: a random raw `orig`, then its PNG-filtered `filt`
/// (`filt = orig − pred mod 256`). The recurrence `orig = filt + pred` then holds
/// by construction.
fn gen_image(slot: usize, b: usize, preds: &[Option<usize>]) -> (Vec<u8>, Vec<u8>) {
    let orig: Vec<u8> = (0..b)
        .map(|i| rnd_byte(((slot as u64) << 40) ^ (i as u64).wrapping_mul(0x1000_0001)))
        .collect();
    let filt: Vec<u8> = (0..b)
        .map(|i| {
            let pred = preds[i].map_or(0u8, |p| orig[p]);
            orig[i].wrapping_sub(pred)
        })
        .collect();
    (orig, filt)
}

// ---------------------------------------------------------------------------
// Ring-u8 SIMD circuit: one linear constraint per byte, 32 images packed.
//   witness/slot: w[0] = 1, w[1+2i] = Orig[i], w[2+2i] = Filt[i]
//   row i:  (Orig[i]) · 1  =  Filt[i] + Orig[pred(i)]          (all mod 256)
// ---------------------------------------------------------------------------
fn build_ring<L: Limb>(
    images: &[(Vec<u8>, Vec<u8>)],
    preds: &[Option<usize>],
    b: usize,
    log_n: usize,
    log_m: usize,
) -> SimdR1csInstance<L> {
    let n = 1usize << log_n;
    let m = 1usize << log_m;
    let one = Gr1::<L>::ONE;
    let mut a = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut bmat = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, m);
    for i in 0..b {
        a.push(i, 1 + 2 * i, one); // Orig[i]
        bmat.push(i, 0, one); // constant 1
        c.push(i, 2 + 2 * i, one); // Filt[i]
        if let Some(p) = preds[i] {
            c.push(i, 1 + 2 * p, one); // + Orig[pred]
        }
    }
    let witnesses: [Vec<Gr1<L>>; SLOTS] = std::array::from_fn(|s| {
        let (orig, filt) = &images[s];
        let mut w = vec![Gr1::<L>::ZERO; m];
        w[0] = one;
        for i in 0..b {
            w[1 + 2 * i] = Gr1(L::from_u64(orig[i] as u64));
            w[2 + 2 * i] = Gr1(L::from_u64(filt[i] as u64));
        }
        w
    });
    SimdR1csInstance {
        a,
        b: bmat,
        c,
        witnesses,
    }
}

// ---------------------------------------------------------------------------
// Field circuit: all 32 images in one R1CS, 11 constraints/byte (full u8 tax).
//   per global byte g: vars [Orig, Filt, borrow, bit_0..bit_7] at 1+11g..
// ---------------------------------------------------------------------------
fn build_field(
    images: &[(Vec<u8>, Vec<u8>)],
    preds: &[Option<usize>],
    b: usize,
    log_n: usize,
    log_m: usize,
) -> FieldR1csInstance {
    let n = 1usize << log_n;
    let m = 1usize << log_m;
    let one = BB::one();
    let c256 = BB::from_canonical_u32(256);
    let mut a = SparseMatrixBB::new(n, m);
    let mut bmat = SparseMatrixBB::new(n, m);
    let mut c = SparseMatrixBB::new(n, m);
    let mut w = vec![BB::zero(); m];
    w[0] = one;

    let recon_col = |g: usize| 1 + 11 * g; // Orig[g]
    for (s, (orig, filt)) in images.iter().enumerate() {
        for i in 0..b {
            let g = s * b + i;
            let base = 1 + 11 * g;
            let rb = 11 * g; // first of this byte's 11 rows
            let (o_col, f_col, brw_col) = (base, base + 1, base + 2);

            // --- witness values ---
            let pred = preds[i].map_or(0u8, |p| orig[p]);
            let sum = filt[i] as u32 + pred as u32; // 0..=510
            let borrow = sum >> 8; // 0 or 1 (Orig = sum - 256·borrow)
            w[o_col] = BB::from_canonical_u32(orig[i] as u32);
            w[f_col] = BB::from_canonical_u32(filt[i] as u32);
            w[brw_col] = BB::from_canonical_u32(borrow);
            for j in 0..8 {
                w[base + 3 + j] = BB::from_canonical_u32(((orig[i] >> j) & 1) as u32);
            }

            // --- row rb:  Orig + 256·borrow = Filt + pred ---
            a.push(rb, o_col, one);
            a.push(rb, brw_col, c256);
            bmat.push(rb, 0, one);
            c.push(rb, f_col, one);
            if let Some(p) = preds[i] {
                c.push(rb, recon_col(s * b + p), one);
            }
            // --- row rb+1:  Orig = Σ 2^j · bit_j ---
            a.push(rb + 1, o_col, one);
            bmat.push(rb + 1, 0, one);
            for j in 0..8 {
                c.push(rb + 1, base + 3 + j, BB::from_canonical_u32(1 << j));
            }
            // --- row rb+2:  borrow · borrow = borrow ---
            a.push(rb + 2, brw_col, one);
            bmat.push(rb + 2, brw_col, one);
            c.push(rb + 2, brw_col, one);
            // --- rows rb+3+j:  bit_j · bit_j = bit_j ---
            for j in 0..8 {
                let row = rb + 3 + j;
                a.push(row, base + 3 + j, one);
                bmat.push(row, base + 3 + j, one);
                c.push(row, base + 3 + j, one);
            }
        }
    }
    FieldR1csInstance {
        a,
        b: bmat,
        c,
        w,
    }
}

#[derive(Clone, Copy, Default)]
struct Cost {
    red: f64,
    piop: f64,
    pcs: f64,
    total: f64,
}

fn main() {
    let log_bytes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let nthreads: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or_else(
        || {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        },
    );
    assert!(
        log_bytes >= 10,
        "log_bytes too small; use >= 10 (32 images need room)"
    );

    let log_slot = log_bytes - 5; // bytes per image = 2^log_slot
    let b = 1usize << log_slot;
    let total_bytes = SLOTS * b;
    let (h, wd) = image_dims(b);
    let preds = predecessors(h, wd);

    // Shared batch of 32 images (same data fed to both provers).
    let images: Vec<(Vec<u8>, Vec<u8>)> =
        (0..SLOTS).map(|s| gen_image(s, b, &preds)).collect();

    // ----- dimensions -----
    // ring: n = b rows (one per byte), m = pad(1 + 2b) witness, 32 slots.
    let log_n_ring = log_slot;
    let log_m_ring = ceil_log2(1 + 2 * b);
    // field: one R1CS over all 32·b bytes, 11 rows + 11 vars per byte.
    let log_n_field = ceil_log2(11 * total_bytes);
    let log_m_field = ceil_log2(1 + 11 * total_bytes);

    let ring = build_ring::<u8>(&images, &preds, b, log_n_ring, log_m_ring);
    let fld = build_field(&images, &preds, b, log_n_field, log_m_field);
    assert!(ring.is_satisfied(), "ring R1CS unsatisfied");
    assert!(fld.is_satisfied(), "field R1CS unsatisfied");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    // ----- ring-u8 SIMD prover -----
    // Secure operating point (δ=0.19, λ=100 ⇒ t=1060, proof-size-optimal shape),
    // same as the rest of the Section-5 evaluation. Prover time is ~t-robust, so
    // the heavier query budget only inflates proof size (not reported by Exp 3).
    let ring_params = SimdSnarkParams::<u8>::secure(log_n_ring, log_m_ring, 0xDEAD, 0xCAFE);
    let (ring_proof, rt) = pool.install(|| prove_simd_with_timing(&ring, &ring_params));
    let ring_cost = Cost {
        red: rt.reduce_ms,
        piop: rt.matvec_abc_ms
            + rt.eq_r_ms
            + rt.mult_sumcheck_ms
            + rt.linear_arow_ms
            + rt.linear_sumcheck_ms,
        pcs: rt.commit_w_encode_ms
            + rt.commit_w_merkle_ms
            + rt.commit_e_encode_ms
            + rt.commit_e_merkle_ms
            + rt.open_w_ms
            + rt.open_e_ms,
        total: rt.total_ms,
    };
    let ring_ok =
        pool.install(|| verify_simd(&ring.a, &ring.b, &ring.c, &ring_params, &ring_proof)).is_ok();

    // ----- field prover -----
    let field_params = FieldSnarkParams::secure(log_n_field, log_m_field, 0xDEAD);
    let (field_proof, ft) = pool.install(|| prove_field_with_timing(&fld, &field_params));
    let field_cost = Cost {
        red: 0.0,
        piop: ft.matvec_abc_ms
            + ft.eq_r_ms
            + ft.mult_sumcheck_ms
            + ft.linear_arow_ms
            + ft.linear_sumcheck_ms,
        pcs: ft.commit_w_ms + ft.open_w_ms,
        total: ft.total_ms,
    };
    let field_ok =
        pool.install(|| verify_field(&fld.a, &fld.b, &fld.c, &field_params, &field_proof)).is_ok();

    // ----------------------------- report -----------------------------
    println!(
        "\nPNG filter (Sub/Up) — prove a batch of {SLOTS} images, ring-u8 SIMD vs field baseline"
    );
    println!(
        "  batch : 2^{log_bytes} bytes ({total_bytes} B) = {SLOTS} images × {b} B  (image {h}×{wd}, \
         Sub on even rows / Up on odd)"
    );
    println!("  threads: {nthreads}\n");

    println!("  R1CS sizes:");
    println!(
        "    ring-u8 SIMD : n=2^{log_n_ring} rows × {SLOTS} slots, m=2^{log_m_ring} ; \
         ALL-LINEAR (0 mults) → {total_bytes} byte-constraints"
    );
    println!(
        "    field (BB)   : n=2^{log_n_field} rows, m=2^{log_m_field} ; \
         11 rows/byte (recurrence+borrow, 8 range bits, recompose, 2 bool) → {} constraints",
        11 * total_bytes
    );
    println!(
        "    → field circuit is {:.1}× the constraint count of the native ring circuit\n",
        (11 * total_bytes) as f64 / total_bytes as f64
    );

    println!("  prove time (ms):");
    println!(
        "    {:<10} {:>12} {:>12}",
        "stage", "ring-u8", "field"
    );
    let row = |name: &str, r: f64, f: f64| {
        println!("    {name:<10} {r:>12.1} {f:>12.1}");
    };
    row("reduction", ring_cost.red, field_cost.red);
    row("PIOP", ring_cost.piop, field_cost.piop);
    row("PCS", ring_cost.pcs, field_cost.pcs);
    row("TOTAL", ring_cost.total, field_cost.total);

    println!(
        "\n  speedup (field / ring-u8): {:.1}×",
        field_cost.total / ring_cost.total
    );
    println!(
        "  verify: ring-u8 {} | field {}",
        if ring_ok { "OK" } else { "FAIL" },
        if field_ok { "OK" } else { "FAIL" }
    );
    println!(
        "\n  note: secure operating point — t=1060 queries on both (δ=0.19, λ=100). Field\n  \
         range-checks only the private raw bytes."
    );
    // Machine-parseable line for the app_filter.dat driver (logsize = log_bytes).
    println!(
        "APP_RESULT {log_bytes} {:.6} {:.6}",
        ring_cost.total / 1e3,
        field_cost.total / 1e3
    );
}
