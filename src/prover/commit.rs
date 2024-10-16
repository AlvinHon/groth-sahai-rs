//! Commit from scalar field [`Fr`](ark_ec::Pairing::Fr) or bilinear group `G1, G2`
//! into the Groth-Sahai commitment group `B1, B2` for the SXDH instantiation.
#![allow(non_snake_case)]

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{fmt::Debug, rand::Rng, UniformRand};

use crate::algebra::{col_vec_to_vec, vec_to_col_vec, Com, Com1, Com2, Mat, Matrix};
use crate::generator::CRS;

/// Contains both the commitment's values (as [`Com1`](crate::algebra::Com1)) and its randomness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commit1<E: Pairing> {
    pub coms: Vec<Com1<E>>,
    pub(super) rand: Matrix<E::ScalarField>,
}

impl<E: Pairing> PartialEq for Commit1<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.coms == other.coms && self.rand == other.rand
    }
}
impl<E: Pairing> Eq for Commit1<E> {}

/// Contains both the commitment's values (as [`Com2`](crate::algebra::Com2)) and its randomness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commit2<E: Pairing> {
    pub coms: Vec<Com2<E>>,
    pub(super) rand: Matrix<E::ScalarField>,
}

impl<E: Pairing> PartialEq for Commit2<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.coms == other.coms && self.rand == other.rand
    }
}
impl<E: Pairing> Eq for Commit2<E> {}

/// Commit a single [`G1`](ark_ec::Pairing::G1Affine) element to [`B1`](crate::algebra::Com1).
pub fn commit_G1<CR, E>(xvar: &E::G1Affine, key: &CRS<E>, rng: &mut CR) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let (r1, r2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));

    // c := i_1(x) + r_1 u_1 + r_2 u_2
    Commit1::<E> {
        coms: vec![
            Com::<E::G1>::linear_map(xvar)
                + vec_to_col_vec(&key.u)[(0, 0)].scalar_mul(&r1)
                + vec_to_col_vec(&key.u)[(1, 0)].scalar_mul(&r2),
        ],
        rand: Matrix::new(&[[r1, r2]]),
    }
}

/// Commit all [`G1`](ark_ec::Pairing::G1Affine) elements in list to corresponding element in [`B1`](crate::algebra::Com1).
pub fn batch_commit_G1<CR, E>(xvars: &[E::G1Affine], key: &CRS<E>, rng: &mut CR) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    // R is a random scalar m x 2 matrix
    let m = xvars.len();
    let R = Matrix::<E::ScalarField>::rand(rng, m, 2);

    // i_1(X) = [ (O, X_1), ..., (O, X_m) ] (m x 1 matrix)
    let lin_x: Matrix<Com1<E>> = vec_to_col_vec(&Com1::<E>::batch_linear_map(xvars));

    // c := i_1(X) + Ru (m x 1 matrix)
    let coms = lin_x.add(&vec_to_col_vec(&key.u).left_mul(&R));

    Commit1::<E> {
        coms: col_vec_to_vec(&coms),
        rand: R,
    }
}

/// Commit a single [scalar field](ark_ec::Pairing::Fr) element to [`B1`](crate::algebra::Com1).
pub fn commit_scalar_to_B1<CR, E>(
    scalar_xvar: &E::ScalarField,
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let r: E::ScalarField = E::ScalarField::rand(rng);

    // c := i_1'(x) + r u_1
    Commit1::<E> {
        coms: vec![
            key.u[1].scalar_linear_map(scalar_xvar, &key.g1_gen)
                + vec_to_col_vec(&key.u)[(0, 0)].scalar_mul(&r),
        ],
        rand: Matrix::new(&[[r]]),
    }
}

/// Commit all [scalar field](ark_ec::Pairing::Fr) elements in list to corresponding element in [`B1`](crate::algebra::Com1).
pub fn batch_commit_scalar_to_B1<CR, E>(
    scalar_xvars: &[E::ScalarField],
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let mprime = scalar_xvars.len();
    let r = Matrix::rand(rng, mprime, 1);
    let slin_x: Matrix<Com<E::G1>> =
        vec_to_col_vec(&key.u[1].batch_scalar_linear_map(scalar_xvars, &key.g1_gen));
    let ru: Matrix<Com1<E>> = vec_to_col_vec(
        &col_vec_to_vec(&r)
            .into_iter()
            .map(|sca| vec_to_col_vec(&key.u)[(0, 0)].scalar_mul(&sca))
            .collect::<Vec<Com1<E>>>(),
    );

    // c := i_1'(x) + r u_1 (mprime x 1 matrix)
    let coms: Matrix<Com1<E>> = slin_x.add(&ru);

    Commit1::<E> {
        coms: col_vec_to_vec(&coms),
        rand: r,
    }
}

/// Commit a single [`G2`](ark_ec::Pairing::G2Affine) element to [`B2`](crate::algebra::Com2).
pub fn commit_G2<CR, E>(yvar: &E::G2Affine, key: &CRS<E>, rng: &mut CR) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let (s1, s2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));

    // d := i_2(y) + s_1 v_1 + s_2 v_2
    Commit2::<E> {
        coms: vec![
            Com2::<E>::linear_map(yvar)
                + vec_to_col_vec(&key.v)[(0, 0)].scalar_mul(&s1)
                + vec_to_col_vec(&key.v)[(1, 0)].scalar_mul(&s2),
        ],
        rand: Matrix::new(&[[s1, s2]]),
    }
}

/// Commit all [`G2`](ark_ec::Pairing::G2Affine) elements in list to corresponding element in [`B2`](crate::algebra::Com2).
pub fn batch_commit_G2<CR, E>(yvars: &[E::G2Affine], key: &CRS<E>, rng: &mut CR) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    // S is a random scalar n x 2 matrix
    let n = yvars.len();
    let S = Matrix::rand(rng, n, 2);

    // i_2(Y) = [ (O, Y_1), ..., (O, Y_m) ] (n x 1 matrix)
    let lin_y: Matrix<Com2<E>> = vec_to_col_vec(&Com2::<E>::batch_linear_map(yvars));

    // c := i_2(Y) + Sv (n x 1 matrix)
    let coms = lin_y.add(&vec_to_col_vec(&key.v).left_mul(&S));

    Commit2::<E> {
        coms: col_vec_to_vec(&coms),
        rand: S,
    }
}

/// Commit a single [scalar field](ark_ec::Pairing::Fr) element to [`B2`](crate::algebra::Com2).
pub fn commit_scalar_to_B2<CR, E>(
    scalar_yvar: &E::ScalarField,
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let s: E::ScalarField = E::ScalarField::rand(rng);
    // d := i_2'(y) + s v_1
    Commit2::<E> {
        coms: vec![
            key.v[1].scalar_linear_map(scalar_yvar, &key.g2_gen)
                + vec_to_col_vec(&key.v)[(0, 0)].scalar_mul(&s),
        ],
        rand: Matrix::new(&[[s]]),
    }
}

/// Commit all [scalar field](ark_ec::Pairing::Fr) elements in list to corresponding element in [`B2`](crate::algebra::Com2).
pub fn batch_commit_scalar_to_B2<CR, E>(
    scalar_yvars: &[E::ScalarField],
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let nprime = scalar_yvars.len();
    let s = Matrix::rand(rng, nprime, 1);
    let slin_y: Matrix<Com2<E>> =
        vec_to_col_vec(&key.v[1].batch_scalar_linear_map(scalar_yvars, &key.g2_gen));
    let sv: Matrix<Com2<E>> = vec_to_col_vec(
        &col_vec_to_vec(&s)
            .into_iter()
            .map(|sca| vec_to_col_vec(&key.v)[(0, 0)].scalar_mul(&sca))
            .collect::<Vec<Com2<E>>>(),
    );

    // d := i_2'(y) + s v_1 (nprime x 1 matrix)
    let coms: Matrix<Com2<E>> = slin_y.add(&sv);

    Commit2::<E> {
        coms: col_vec_to_vec(&coms),
        rand: s,
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::ops::Mul;

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::CurveGroup;
    use ark_std::test_rng;

    use crate::AbstractCrs;

    use super::*;

    type G1 = <F as Pairing>::G1;
    type G2 = <F as Pairing>::G2;
    type Fr = <F as Pairing>::ScalarField;

    // Uses an affine group generator to produce a projective group element represented by the numeric string.
    #[allow(unused_macros)]
    macro_rules! projective_group_new {
        ($gen:expr, $strnum:tt) => {
            $gen.mul(Fr::from_str($strnum).unwrap())
        };
    }

    #[test]
    fn test_commit_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);
        let r1 = Fr::rand(&mut rng);
        let r2 = Fr::rand(&mut rng);
        let com1 = Commit1::<F> {
            coms: vec![Com::<G1>(
                crs.g1_gen.mul(r1).into_affine(),
                crs.g1_gen.mul(r2).into_affine(),
            )],
            rand: Matrix::new(&[[r1, r2]]),
        };
        let com2 = Commit2::<F> {
            coms: vec![Com::<G2>(
                crs.g2_gen.mul(r1).into_affine(),
                crs.g2_gen.mul(r2).into_affine(),
            )],
            rand: Matrix::new(&[[r1, r2]]),
        };

        // Serialize and deserialize the commitment 1
        let mut c_bytes = Vec::new();
        com1.serialize_compressed(&mut c_bytes).unwrap();
        let com1_de = Commit1::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(com1, com1_de);

        let mut u_bytes = Vec::new();
        com1.serialize_uncompressed(&mut u_bytes).unwrap();
        let com1_de = Commit1::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(com1, com1_de);

        // Serialize and deserialize the commitment 2
        let mut c_bytes = Vec::new();
        com2.serialize_compressed(&mut c_bytes).unwrap();
        let com2_de = Commit2::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(com2, com2_de);

        let mut u_bytes = Vec::new();
        com2.serialize_uncompressed(&mut u_bytes).unwrap();
        let com2_de = Commit2::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(com2, com2_de);
    }
}
