//! A/B microbenchmark: does **per-round Fiat–Shamir** cost anything in the
//! sumcheck hot loop, serial and parallel?
//!
//! The two columns run the *same* mult-check sumcheck over the same inputs; the
//! only difference is the challenger:
//!
//! - **per-round** — `snark_ring::transcript::Transcript`: every round absorbs
//!   its message (Blake3) and squeezes a transcript-bound `T_2` challenge. This
//!   is the current, sound FS discipline.
//! - **lock-step** — `R2Sampler`: `observe` is a no-op; challenges come straight
//!   from a PRNG seeded once. This reproduces the *old* per-sumcheck FS cost.
//!
//! Everything else (the deg-3 round compute, the slim-eq build, the internal
//! `a.clone()`/`b.clone()`, the fold) is identical in both columns, so the
//! `per-round − lock-step` delta isolates exactly the per-round FS overhead.
//! Each column is timed under a 1-thread rayon pool (serial) and an all-cores
//! pool (parallel).
//!
//! Usage: `fs_overhead [log_size] [width] [reps]`   (defaults: 20, 8, 7)
//!   log_size = sumcheck variables n (= a 2^log_size mult-check, i.e. the
//!   mult-check of a 2^(log_size+5)-scalar SIMD instance); width ∈ {8,16,32,64}.

use galois_ring::{Gr128, Limb, R2Sampler};
use piop_ring::{prove_mult_check, Mle};
use snark_ring::transcript::Transcript;
use std::hint::black_box;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median-of-`reps` wall time (ms) of `run`, after `warmup` untimed calls.
fn time_ms(reps: usize, warmup: usize, run: impl Fn()) -> f64 {
    for _ in 0..warmup {
        run();
    }
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        run();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(ts)
}

fn run<L: Limb>(width: u32, n: usize, reps: usize) {
    let len = 1usize << n;

    // Inputs (built once, outside every timed region). Teichmüller draws are a
    // convenient source of valid R_2 elements; the sumcheck cost is independent
    // of whether they are Teichmüller. The claim is irrelevant to *prover*
    // timing (the prover does not validate it), so use 0.
    let mut setup = R2Sampler::<L>::new(0x5EED_1234);
    let a = Mle::new((0..len).map(|_| setup.sample()).collect::<Vec<_>>());
    let b = Mle::new((0..len).map(|_| setup.sample()).collect::<Vec<_>>());
    let r_target: Vec<Gr128<L>> = (0..n).map(|_| setup.sample()).collect();
    let claim = Gr128::<L>::ZERO;

    let nthreads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    // The two challenger flavours, as closures producing one fresh prove call.
    let per_round = || {
        let mut tr = Transcript::new(b"fs-bench");
        black_box(prove_mult_check(&a, &b, &r_target, claim, &mut tr));
    };
    let lock_step = || {
        let mut s = R2Sampler::<L>::new(0xBEEF_CAFE);
        black_box(prove_mult_check(&a, &b, &r_target, claim, &mut s));
    };

    let warmup = 2;
    let pr_ser = time_ms(reps, warmup, || pool1.install(&per_round));
    let ls_ser = time_ms(reps, warmup, || pool1.install(&lock_step));
    let pr_par = time_ms(reps, warmup, || pooln.install(&per_round));
    let ls_par = time_ms(reps, warmup, || pooln.install(&lock_step));

    let pct = |new: f64, base: f64| {
        if base > 0.0 {
            (new - base) / base * 100.0
        } else {
            0.0
        }
    };

    println!(
        "\n##### u{width}  mult-check sumcheck  n={n}  (= 2^{} scalar SIMD)  reps={reps}, median #####",
        n + 5
    );
    println!(
        "  {:<10} {:>14} {:>14} {:>14}",
        "threads", "lock-step ms", "per-round ms", "FS overhead"
    );
    println!(
        "  {:<10} {:>14.3} {:>14.3} {:>13.2}%",
        format!("1 (serial)"),
        ls_ser,
        pr_ser,
        pct(pr_ser, ls_ser)
    );
    println!(
        "  {:<10} {:>14.3} {:>14.3} {:>13.2}%",
        format!("{nthreads} (par)"),
        ls_par,
        pr_par,
        pct(pr_par, ls_par)
    );
    println!(
        "  (FS overhead = (per-round − lock-step)/lock-step; the round compute is \
         identical, so this is the cost of {n} per-round absorb+squeeze pairs.)"
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let width: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reps: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    match width {
        8 => run::<u8>(8, n, reps),
        16 => run::<u16>(16, n, reps),
        32 => run::<u32>(32, n, reps),
        64 => run::<u64>(64, n, reps),
        _ => panic!("width must be 8, 16, 32, or 64"),
    }
}
