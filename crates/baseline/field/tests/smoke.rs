//! Axiom-level smoke tests for the BabyBear field and its degree-4 binomial
//! extension. Locks in the API surface that downstream crates rely on.

use field::{AbstractExtensionField, AbstractField, Field, BB, BB4};

#[test]
fn babybear_basic_axioms() {
    let a = BB::from_canonical_u32(7);
    let b = BB::from_canonical_u32(11);
    let zero = BB::zero();
    let one = BB::one();

    assert_eq!(a + zero, a);
    assert_eq!(a * one, a);
    assert_eq!(a + b, b + a);
    assert_eq!(a * b, b * a);
    assert_eq!((a + b) * a, a * a + a * b);
    assert_eq!(a - a, zero);
    assert_eq!(a * a.inverse(), one);
}

#[test]
fn bb4_basic_axioms() {
    let a = BB4::from_canonical_u32(7);
    let b = BB4::from_canonical_u32(11);
    let zero = BB4::zero();
    let one = BB4::one();

    assert_eq!(a + zero, a);
    assert_eq!(a * one, a);
    assert_eq!(a + b, b + a);
    assert_eq!(a * b, b * a);
    assert_eq!((a + b) * a, a * a + a * b);
    assert_eq!(a - a, zero);
    assert_eq!(a * a.inverse(), one);
}

#[test]
fn bb4_is_actually_quartic() {
    let basis = <BB4 as AbstractExtensionField<BB>>::from_base_slice(&[
        BB::from_canonical_u32(1),
        BB::from_canonical_u32(2),
        BB::from_canonical_u32(3),
        BB::from_canonical_u32(4),
    ]);
    let coeffs: &[BB] = basis.as_base_slice();
    assert_eq!(coeffs.len(), 4);
    assert_eq!(coeffs[0], BB::from_canonical_u32(1));
    assert_eq!(coeffs[3], BB::from_canonical_u32(4));
}
