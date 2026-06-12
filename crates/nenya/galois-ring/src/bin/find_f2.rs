//! Search for a sparse irreducible polynomial of degree 32 over F_16.
//!
//! (Standalone one-off tool; a few F_16 helpers are kept for completeness even
//! when the current search path doesn't call them.)
#![allow(dead_code)]
//!
//! F_16 = F_2[w] / (w^4 + w + 1).  Elements are u8 in 0..16.
//! Arithmetic is carry-less (XOR-based) over F_2.
//!
//! We search trinomials X^32 + a*X^k + b (b != 0) for each k in 1..32,
//! then pentanomials if none found.
//!
//! Irreducibility test: a degree-d polynomial over F_{2^4} is irreducible iff
//!   1. gcd(f, X^{(2^4)^i} - X) = 1  for all prime p | d, and
//!   2. gcd(f, X^{(2^4)^{d/p}} - X) = f  for all prime p | d.
//!
//! Equivalently (Ben-Or): f is irreducible iff
//!   (a) f | X^{q^d} - X  (q = 16, d = 32), and
//!   (b) gcd(f, X^{q^k} - X) = 1 for k = 1..d/2.
//!
//! For d=32, prime divisors are {2}. So we check:
//!   (a) f | X^{16^32} - X   and
//!   (b) gcd(f, X^{16^16} - X) = 1.
//!
//! We do all polynomial ops modulo f (or unreduced), using Vec<u8> coefficients in F_16,
//! index = power of X (index 0 = constant term).

// ---------------------------------------------------------------------------
// F_16 arithmetic  (carry-less, primitive poly = x^4+x+1 = 0b10011 = 19)
// ---------------------------------------------------------------------------

const F16_PRIM: u8 = 0b0001_0011; // x^4 + x + 1 (without the leading x^4 bit)

fn f16_add(a: u8, b: u8) -> u8 {
    a ^ b
}

fn f16_mul(a: u8, b: u8) -> u8 {
    let mut a = a as u16;
    let mut b = b as u16;
    let mut result = 0u16;
    while b > 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & 0x10 != 0 {
            a ^= F16_PRIM as u16;
        }
    }
    result as u8
}

fn f16_neg(a: u8) -> u8 {
    a
} // char 2, neg = identity

fn f16_inv(a: u8) -> u8 {
    assert!(a != 0);
    // Fermat: a^{14} = a^{-1} in F_{16}^*
    let mut r = 1u8;
    let mut base = a;
    let mut exp = 14u32;
    while exp > 0 {
        if exp & 1 == 1 {
            r = f16_mul(r, base);
        }
        base = f16_mul(base, base);
        exp >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Polynomial arithmetic over F_16
// Polynomials are Vec<u8> with coeff[i] = coefficient of X^i.
// Degree = len-1 (trailing zeros are stripped by normalize).
// ---------------------------------------------------------------------------

fn poly_normalize(p: &mut Vec<u8>) {
    while p.last() == Some(&0) {
        p.pop();
    }
}

fn poly_zero() -> Vec<u8> {
    vec![]
}
fn poly_one() -> Vec<u8> {
    vec![1]
}
fn poly_x() -> Vec<u8> {
    vec![0, 1]
}

fn poly_degree(p: &[u8]) -> Option<usize> {
    if p.is_empty() {
        None
    } else {
        Some(p.len() - 1)
    }
}

fn poly_add(a: &[u8], b: &[u8]) -> Vec<u8> {
    let n = a.len().max(b.len());
    let mut c = vec![0u8; n];
    for i in 0..a.len() {
        c[i] ^= a[i];
    }
    for i in 0..b.len() {
        c[i] ^= b[i];
    }
    poly_normalize(&mut c);
    c
}

fn poly_scale(a: &[u8], s: u8) -> Vec<u8> {
    let mut c: Vec<u8> = a.iter().map(|&x| f16_mul(x, s)).collect();
    poly_normalize(&mut c);
    c
}

fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() {
        return poly_zero();
    }
    let mut c = vec![0u8; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            c[i + j] ^= f16_mul(ai, bj);
        }
    }
    poly_normalize(&mut c);
    c
}

/// Divide a by b, return (quotient, remainder).
fn poly_divrem(a: &[u8], b: &[u8]) -> (Vec<u8>, Vec<u8>) {
    assert!(!b.is_empty());
    let mut rem = a.to_vec();
    let db = b.len() - 1;
    let lc_inv = f16_inv(*b.last().unwrap());
    let mut quot_coeffs = vec![];
    while rem.len() >= b.len() {
        let dr = rem.len() - 1;
        let c = f16_mul(*rem.last().unwrap(), lc_inv);
        quot_coeffs.push((dr - db, c));
        // rem -= c * X^{dr-db} * b
        for (i, &bi) in b.iter().enumerate() {
            rem[i + dr - db] ^= f16_mul(c, bi);
        }
        poly_normalize(&mut rem);
    }
    // Reconstruct quotient (coefficients were added from high to low)
    let max_deg = quot_coeffs.iter().map(|&(d, _)| d).max().unwrap_or(0);
    let mut quot = vec![0u8; max_deg + 1];
    for (d, c) in quot_coeffs {
        quot[d] ^= c;
    }
    poly_normalize(&mut quot);
    (quot, rem)
}

fn poly_rem(a: &[u8], b: &[u8]) -> Vec<u8> {
    poly_divrem(a, b).1
}

fn poly_gcd(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    while !b.is_empty() {
        let r = poly_rem(&a, &b);
        a = b;
        b = r;
    }
    // Normalize to monic
    if let Some(&lc) = a.last() {
        if lc != 1 {
            let inv = f16_inv(lc);
            a = poly_scale(&a, inv);
        }
    }
    a
}

/// Compute X^{16^k} mod f using repeated squaring in the ring F_16[X]/(f).
/// Returns the polynomial representative of X^{16^k} in that ring.
fn x_pow_16k_mod(k: u32, f: &[u8]) -> Vec<u8> {
    // We want (X)^{16^k} mod f.
    // Use: x^{16^k} = (x^{16^{k-1}})^16 mod f
    // Start with x^1 mod f = x, then raise to the 16th power k times.
    // Raising to the 16th power = 4 squarings.
    let mut cur = poly_x(); // X^1 mod f
    for _ in 0..k {
        // square 4 times (= raise to 16th power)
        for _ in 0..4 {
            cur = poly_rem(&poly_mul(&cur, &cur), f);
        }
    }
    cur
}

/// Ben-Or irreducibility test for a monic degree-32 polynomial over F_16.
/// d = 32, prime divisors of 32 = {2}.
/// Conditions:
///   (a) gcd(f, X^{16^(32/2)} - X) = gcd(f, X^{16^16} - X) = 1
///   (b) f | X^{16^32} - X  i.e. X^{16^32} ≡ X (mod f)
fn is_irreducible_deg32(f: &[u8]) -> bool {
    // Condition (a): gcd(f, X^{16^16} - X) = 1
    let x16_16 = x_pow_16k_mod(16, f);
    // X^{16^16} - X in F_16 char 2: subtract = XOR = add
    let x16_16_minus_x = poly_add(&x16_16, &poly_x());
    let g = poly_gcd(f, &x16_16_minus_x);
    if poly_degree(&g) != Some(0) {
        return false; // gcd != 1, reducible
    }

    // Condition (b): X^{16^32} ≡ X (mod f)
    let x16_32 = x_pow_16k_mod(32, f);
    x16_32 == poly_x()
}

/// Build the polynomial X^32 + a*X^k + b (monic degree 32 trinomial).
fn trinomial(k: usize, a: u8, b: u8) -> Vec<u8> {
    let mut p = vec![0u8; 33];
    p[32] = 1;
    p[k] ^= a;
    p[0] ^= b;
    poly_normalize(&mut p);
    p
}

/// Build monic degree-32 quadrinomial X^32 + a*X^j + b*X^k + c  (j > k >= 1, a,b,c != 0).
fn quadrinomial(j: usize, a: u8, k: usize, b: u8, c: u8) -> Vec<u8> {
    let mut p = vec![0u8; 33];
    p[32] = 1;
    p[j] ^= a;
    p[k] ^= b;
    p[0] ^= c;
    poly_normalize(&mut p);
    p
}

fn print_poly(f: &[u8]) {
    let deg = f.len() - 1;
    print!("X^{}", deg);
    for (i, &c) in f.iter().enumerate().rev().skip(1) {
        if c != 0 {
            if i == 0 {
                print!(" + {}", c);
            } else if i == 1 {
                print!(" + {}*X", c);
            } else {
                print!(" + {}*X^{}", c, i);
            }
        }
    }
    println!();
}

fn main() {
    // --- Trinomial search ---
    println!("Searching trinomials X^32 + a*X^k + b over F_16...");
    for k in 1..32usize {
        for a in 1u8..16 {
            for b in 1u8..16 {
                let f = trinomial(k, a, b);
                if f.len() != 33 {
                    continue;
                }
                if is_irreducible_deg32(&f) {
                    print!("FOUND trinomial: ");
                    print_poly(&f);
                    println!("  k={} a={} b={}", k, a, b);
                    return;
                }
            }
        }
    }
    println!("No trinomial found.\n");

    // --- Quadrinomial search ---
    println!("Searching quadrinomials X^32 + a*X^j + b*X^k + c over F_16...");
    println!("(j > k, a,b,c in 1..15)");
    let mut count = 0u64;
    for j in 2..32usize {
        for k in 1..j {
            for a in 1u8..16 {
                for b in 1u8..16 {
                    for c in 1u8..16 {
                        let f = quadrinomial(j, a, k, b, c);
                        if f.len() != 33 {
                            continue;
                        }
                        if is_irreducible_deg32(&f) {
                            print!("FOUND quadrinomial: ");
                            print_poly(&f);
                            println!("  j={} a={} k={} b={} c={}", j, a, k, b, c);
                            return;
                        }
                        count += 1;
                        if count % 100_000 == 0 {
                            eprint!("\r  tested {}K (j={}, k={})   ", count / 1000, j, k);
                        }
                    }
                }
            }
        }
    }
    println!("\nNo quadrinomial found either. Need pentanomial.");
}
