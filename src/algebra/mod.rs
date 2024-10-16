//! Contains the abstract algebric structures supporting arithmetics for Groth-Sahai commitment groups
//! `(B1, B2, BT)`, its concrete representation within the SXDH instantiation `(Com1, Com2, ComT)`.
//!
//! # Properties
//!
//! In a Type-III pairing setting, the Groth-Sahai instantiation requires the SXDH assumption,
//! implementing the commitment group using elements of the bilinear group over an elliptic curve.
//! [`Com1`](crate::algebra::Com1) and [`Com2`](crate::algebra::Com2) are represented by 2 x 1 vectors of elements
//! in the corresponding groups [`G1Affine`](ark_ec::pairing::Pairing::G1Affine) and [`G2Affine`](ark_ec::pairing::Pairing::G2Affine).
//! [`ComT`](crate::algebra::ComT) represents a 2 x 2 matrix of elements in [`GT`](ark_ec::pairing::PairingOutput).
//!
//! All of `Com1`, `Com2`, `ComT` are expressed as follows:
//! * Addition is defined by entry-wise addition of elements in `G1Affine`, `G2Affine`:
//!     * The equality is equality of all elements
//!     * The zero element is the zero vector
//!     * The negation is the additive inverse (i.e. negation) of all elements
//!
//! The Groth-Sahai proof system uses matrices of commitment group elements in its computations as
//! well.

pub mod com;
pub use com::*;

pub mod com_t;
pub use com_t::*;

pub mod matrix;
pub use matrix::*;

// type alias for `Com1` and `Com2` which uses the generic struct `Com`.
pub type Com1<E> = Com<<E as ark_ec::pairing::Pairing>::G1>;
pub type Com2<E> = Com<<E as ark_ec::pairing::Pairing>::G2>;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use crate::{AbstractCrs, CRS};

    use super::*;

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::{
        pairing::{Pairing, PairingOutput},
        AffineRepr, CurveGroup,
    };
    use ark_ff::{UniformRand, Zero};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::test_rng;
    use std::ops::Mul;

    type G1Affine = <F as Pairing>::G1Affine;
    type G1 = <F as Pairing>::G1;
    type G2Affine = <F as Pairing>::G2Affine;
    type G2 = <F as Pairing>::G2;
    type GT = PairingOutput<F>;
    type Fr = <F as Pairing>::ScalarField;

    #[test]
    fn test_B1_add_zero() {
        let mut rng = test_rng();
        let a = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let zero = Com::<G1>(G1Affine::zero(), G1Affine::zero());
        let asub = a + zero;

        assert_eq!(zero, Com::<G1>::zero());
        assert!(zero.is_zero());
        assert_eq!(a, asub);
    }

    #[test]
    fn test_B2_add_zero() {
        let mut rng = test_rng();
        let a = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let zero = Com::<G2>(G2Affine::zero(), G2Affine::zero());
        let asub = a + zero;

        assert_eq!(zero, Com::<G2>::zero());
        assert!(zero.is_zero());
        assert_eq!(a, asub);
    }

    #[test]
    fn test_BT_add_zero() {
        let mut rng = test_rng();
        let a = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let zero = ComT::<F>(GT::zero(), GT::zero(), GT::zero(), GT::zero());
        let asub = a + zero;

        assert_eq!(zero, ComT::<F>::zero());
        assert!(zero.is_zero());
        assert_eq!(a, asub);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_add() {
        let mut rng = test_rng();
        let a = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let ab = a + b;
        let ba = b + a;

        assert_eq!(ab, Com::<G1>((a.0 + b.0).into(), (a.1 + b.1).into()));
        assert_eq!(ab, ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_add() {
        let mut rng = test_rng();
        let a = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let b = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let ab = a + b;
        let ba = b + a;

        assert_eq!(ab, Com::<G2>((a.0 + b.0).into(), (a.1 + b.1).into()));
        assert_eq!(ab, ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_BT_add() {
        let mut rng = test_rng();
        let a = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let b = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let ab = a + b;
        let ba = b + a;

        assert_eq!(ab, ComT::<F>(a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
        assert_eq!(ab, ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_sum() {
        let mut rng = test_rng();
        let a = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let c = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );

        let abc_vec = vec![a, b, c];
        let abc: Com1<F> = abc_vec.into_iter().sum();

        assert_eq!(abc, a + b + c);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_sum() {
        let mut rng = test_rng();
        let a = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let b = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let c = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );

        let abc_vec = vec![a, b, c];
        let abc: Com2<F> = abc_vec.into_iter().sum();

        assert_eq!(abc, a + b + c);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_BT_sum() {
        let mut rng = test_rng();
        let a = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let b = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let c = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );

        let abc_vec = vec![a, b, c];
        let abc: ComT<F> = abc_vec.into_iter().sum();

        assert_eq!(abc, a + b + c);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_neg() {
        let mut rng = test_rng();
        let b = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let bneg = -b;
        let zero = b + bneg;

        assert!(zero.is_zero());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_neg() {
        let mut rng = test_rng();
        let b = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let bneg = -b;
        let zero = b + bneg;

        assert!(zero.is_zero());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_BT_neg() {
        let mut rng = test_rng();
        let b = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let bneg = -b;
        let zero = b + bneg;

        assert!(zero.is_zero());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_sub() {
        let mut rng = test_rng();
        let a = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let ab = a - b;
        let ba = b - a;

        assert_eq!(ab, Com::<G1>((a.0 + -b.0).into(), (a.1 + -b.1).into()));
        assert_eq!(ab, -ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_sub() {
        let mut rng = test_rng();
        let a = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let b = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let ab = a - b;
        let ba = b - a;

        assert_eq!(ab, Com::<G2>((a.0 + -b.0).into(), (a.1 + -b.1).into()));
        assert_eq!(ab, -ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_BT_sub() {
        let mut rng = test_rng();
        let a = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let b = ComT::<F>(
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
            GT::rand(&mut rng),
        );
        let ab = a - b;
        let ba = b - a;

        assert_eq!(ab, ComT::<F>(a.0 - b.0, a.1 - b.1, a.2 - b.2, a.3 - b.3));
        assert_eq!(ab, -ba);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_scalar_mul() {
        let mut rng = test_rng();
        let b = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let scalar = Fr::rand(&mut rng);
        let b0 = b.0.mul(scalar);
        let b1 = b.1.mul(scalar);
        let bres = b.scalar_mul(&scalar);
        let bexp = Com::<G1>(b0.into_affine(), b1.into_affine());

        assert_eq!(bres, bexp);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_scalar_mul() {
        let mut rng = test_rng();
        let b = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let scalar = Fr::rand(&mut rng);
        let b0 = b.0.mul(scalar);
        let b1 = b.1.mul(scalar);
        let bres = b.scalar_mul(&scalar);
        let bexp = Com::<G2>(b0.into_affine(), b1.into_affine());

        assert_eq!(bres, bexp);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_serde() {
        let mut rng = test_rng();
        let a = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );

        // Serialize and deserialize Com1.

        let mut c_bytes = Vec::new();
        a.serialize_compressed(&mut c_bytes).unwrap();
        let a_de = Com::<G1>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(a, a_de);

        let mut u_bytes = Vec::new();
        a.serialize_uncompressed(&mut u_bytes).unwrap();
        let a_de = Com::<G1>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(a, a_de);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B2_serde() {
        let mut rng = test_rng();
        let a = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );

        // Serialize and deserialize Com2.

        let mut c_bytes = Vec::new();
        a.serialize_compressed(&mut c_bytes).unwrap();
        let a_de = Com::<G2>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(a, a_de);

        let mut u_bytes = Vec::new();
        a.serialize_uncompressed(&mut u_bytes).unwrap();
        let a_de = Com::<G2>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(a, a_de);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G1() {
        let mut rng = test_rng();
        let b1 = Com::<G1>(G1Affine::zero(), G1Affine::zero());
        let b2 = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let bt = ComT::pairing(b1, b2);

        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, GT::zero());
        assert_eq!(bt.2, GT::zero());
        assert_eq!(bt.3, GT::zero());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G2() {
        let mut rng = test_rng();
        let b1 = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b2 = Com::<G2>(G2Affine::zero(), G2Affine::zero());
        let bt = ComT::pairing(b1, b2);

        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, GT::zero());
        assert_eq!(bt.2, GT::zero());
        assert_eq!(bt.3, GT::zero());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_commit() {
        let mut rng = test_rng();
        let b1 = Com::<G1>(G1Affine::zero(), G1::rand(&mut rng).into_affine());
        let b2 = Com::<G2>(G2Affine::zero(), G2::rand(&mut rng).into_affine());
        let bt = ComT::pairing(b1, b2);

        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, GT::zero());
        assert_eq!(bt.2, GT::zero());
        assert_eq!(bt.3, F::pairing(b1.1, b2.1));
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_rand() {
        let mut rng = test_rng();
        let b1 = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b2 = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let bt = ComT::pairing(b1, b2);

        assert_eq!(bt.0, F::pairing(b1.0, b2.0));
        assert_eq!(bt.1, F::pairing(b1.0, b2.1));
        assert_eq!(bt.2, F::pairing(b1.1, b2.0));
        assert_eq!(bt.3, F::pairing(b1.1, b2.1));
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_sum() {
        let mut rng = test_rng();
        let x1 = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let x2 = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let y1 = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let y2 = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let x = vec![x1, x2];
        let y = vec![y1, y2];
        let exp: ComT<F> = vec![ComT::<F>::pairing(x1, y1), ComT::<F>::pairing(x2, y2)]
            .into_iter()
            .sum();
        let res: ComT<F> = ComT::<F>::pairing_sum(&x, &y);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B_into_matrix() {
        let mut rng = test_rng();
        let b1 = Com::<G1>(
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        );
        let b2 = Com::<G2>(
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        );
        let bt = ComT::<F>::pairing(b1, b2);

        // B1 and B2 can be representing as 2-dim column vectors
        assert_eq!(b1.as_col_vec(), Matrix::new(&[[b1.0], [b1.1]]));
        assert_eq!(b2.as_col_vec(), Matrix::new(&[[b2.0], [b2.1]]));
        // BT can be represented as a 2 x 2 matrix
        assert_eq!(bt.as_matrix(), Matrix::new(&[[bt.0, bt.1], [bt.2, bt.3]]));
    }

    #[test]
    fn test_B_from_matrix() {
        let mut rng = test_rng();
        let b1_vec = Matrix::new(&[
            [G1::rand(&mut rng).into_affine()],
            [G1::rand(&mut rng).into_affine()],
        ]);

        let b2_vec = Matrix::new(&[
            [G2::rand(&mut rng).into_affine()],
            [G2::rand(&mut rng).into_affine()],
        ]);
        let bt_vec = Matrix::new(&[
            [
                F::pairing(b1_vec[(0, 0)], b2_vec[(0, 0)]),
                F::pairing(b1_vec[(0, 0)], b2_vec[(1, 0)]),
            ],
            [
                F::pairing(b1_vec[(1, 0)], b2_vec[(0, 0)]),
                F::pairing(b1_vec[(1, 0)], b2_vec[(1, 0)]),
            ],
        ]);

        let b1 = Com::<G1>::from(b1_vec.clone());
        let b2 = Com::<G2>::from(b2_vec.clone());
        let bt = ComT::<F>::from(bt_vec.clone());

        assert_eq!(b1.0, b1_vec[(0, 0)]);
        assert_eq!(b1.1, b1_vec[(1, 0)]);
        assert_eq!(b2.0, b2_vec[(0, 0)]);
        assert_eq!(b2.1, b2_vec[(1, 0)]);
        assert_eq!(bt.0, bt_vec[(0, 0)]);
        assert_eq!(bt.1, bt_vec[(0, 1)]);
        assert_eq!(bt.2, bt_vec[(1, 0)]);
        assert_eq!(bt.3, bt_vec[(1, 1)]);
    }

    #[test]
    fn test_batched_linear_maps() {
        let mut rng = test_rng();
        let vec_g1 = vec![
            G1::rand(&mut rng).into_affine(),
            G1::rand(&mut rng).into_affine(),
        ];
        let vec_g2 = vec![
            G2::rand(&mut rng).into_affine(),
            G2::rand(&mut rng).into_affine(),
        ];
        let vec_b1 = Com::<G1>::batch_linear_map(&vec_g1);
        let vec_b2 = Com::<G2>::batch_linear_map(&vec_g2);

        assert_eq!(vec_b1[0], Com::<G1>::linear_map(&vec_g1[0]));
        assert_eq!(vec_b1[1], Com::<G1>::linear_map(&vec_g1[1]));
        assert_eq!(vec_b2[0], Com::<G2>::linear_map(&vec_g2[0]));
        assert_eq!(vec_b2[1], Com::<G2>::linear_map(&vec_g2[1]));
    }

    #[test]
    fn test_batched_scalar_linear_maps() {
        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let vec_scalar = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let vec_b1 = key.u[1].batch_scalar_linear_map(&vec_scalar, &key.g1_gen);
        let vec_b2 = key.v[1].batch_scalar_linear_map(&vec_scalar, &key.g2_gen);

        assert_eq!(
            vec_b1[0],
            key.u[1].scalar_linear_map(&vec_scalar[0], &key.g1_gen)
        );
        assert_eq!(
            vec_b1[1],
            key.u[1].scalar_linear_map(&vec_scalar[1], &key.g1_gen)
        );
        assert_eq!(
            vec_b2[0],
            key.v[1].scalar_linear_map(&vec_scalar[0], &key.g2_gen)
        );
        assert_eq!(
            vec_b2[1],
            key.v[1].scalar_linear_map(&vec_scalar[1], &key.g2_gen)
        );
    }

    #[test]
    fn test_PPE_linear_maps() {
        let mut rng = test_rng();
        let a1 = G1::rand(&mut rng).into_affine();
        let a2 = G2::rand(&mut rng).into_affine();
        let at = F::pairing(a1, a2);
        let b1 = Com::<G1>::linear_map(&a1);
        let b2 = Com::<G2>::linear_map(&a2);
        let bt = ComT::<F>::linear_map_PPE(&at);

        assert_eq!(b1.0, G1Affine::zero());
        assert_eq!(b1.1, a1);
        assert_eq!(b2.0, G2Affine::zero());
        assert_eq!(b2.1, a2);
        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, GT::zero());
        assert_eq!(bt.2, GT::zero());
        assert_eq!(bt.3, F::pairing(a1, a2));
    }

    // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
    #[test]
    fn test_MSMEG1_linear_maps() {
        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = G1::rand(&mut rng).into_affine();
        let a2 = Fr::rand(&mut rng);
        let at = a1.mul(a2).into_affine();
        let b1 = Com::<G1>::linear_map(&a1);
        let b2 = key.v[1].scalar_linear_map(&a2, &key.g2_gen);
        let bt = ComT::<F>::linear_map_MSMEG1(&at, &key.v[1], &key.g2_gen);

        assert_eq!(b1.0, G1Affine::zero());
        assert_eq!(b1.1, a1);
        assert_eq!(b2.0, key.v[1].0.mul(a2));
        assert_eq!(b2.1, (key.v[1].1 + key.g2_gen).mul(a2));
        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, GT::zero());
        assert_eq!(bt.2, F::pairing(at, key.v[1].0));
        assert_eq!(bt.3, F::pairing(at, key.v[1].1 + key.g2_gen));
    }

    // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
    #[test]
    fn test_MSMEG2_linear_maps() {
        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = G2::rand(&mut rng).into_affine();
        let at = a2.mul(a1).into_affine();
        let b1 = key.u[1].scalar_linear_map(&a1, &key.g1_gen);
        let b2 = Com::<G2>::linear_map(&a2);
        let bt = ComT::<F>::linear_map_MSMEG2(&at, &key.u[1], &key.g1_gen);

        assert_eq!(b1.0, key.u[1].0.mul(a1));
        assert_eq!(b1.1, (key.u[1].1 + key.g1_gen).mul(a1));
        assert_eq!(b2.0, G2Affine::zero());
        assert_eq!(b2.1, a2);
        assert_eq!(bt.0, GT::zero());
        assert_eq!(bt.1, F::pairing(key.u[1].0, at));
        assert_eq!(bt.2, GT::zero());
        assert_eq!(bt.3, F::pairing(key.u[1].1 + key.g1_gen, at));
    }

    // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
    #[test]
    fn test_QuadEqu_linear_maps() {
        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = Fr::rand(&mut rng);
        let at = a1 * a2;
        let b1 = key.u[1].scalar_linear_map(&a1, &key.g1_gen);
        let b2 = key.v[1].scalar_linear_map(&a2, &key.g2_gen);
        let bt = ComT::<F>::linear_map_quad(&at, &key.u[1], &key.g1_gen, &key.v[1], &key.g2_gen);
        let W1 = Com::<G1>(key.u[1].0, (key.u[1].1 + key.g1_gen).into());
        let W2 = Com::<G2>(key.v[1].0, (key.v[1].1 + key.g2_gen).into());
        assert_eq!(b1.0, W1.0.mul(a1));
        assert_eq!(b1.1, W1.1.mul(a1));
        assert_eq!(b2.0, W2.0.mul(a2));
        assert_eq!(b2.1, W2.1.mul(a2));
        assert_eq!(
            bt,
            ComT::<F>::pairing(W1.scalar_mul(&a1), W2.scalar_mul(&a2))
        );
        assert_eq!(bt, ComT::<F>::pairing(W1, W2.scalar_mul(&at)));
    }
}
