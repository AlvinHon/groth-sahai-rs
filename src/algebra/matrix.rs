use std::fmt::Debug;

use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use ndarray::{Array, Ix2};

use super::Com;

// pub type Matrix<F> = Array<F, Ix2>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<F> {
    inner: Array<F, Ix2>,
}

impl<F> Matrix<F> {
    pub fn new<const N: usize>(xs: &[[F; N]]) -> Self
    where
        F: Clone,
    {
        Self {
            inner: ndarray::arr2(xs),
        }
    }

    pub fn to_vecs(&self) -> Vec<Vec<F>>
    where
        F: Clone,
    {
        self.inner
            .outer_iter()
            .map(|row| row.iter().cloned().collect())
            .collect()
    }

    pub fn from_vecs(vecs: Vec<Vec<F>>) -> Self
    where
        F: Clone,
    {
        Self {
            inner: Array::from_shape_vec(
                (vecs.len(), vecs[0].len()),
                vecs.into_iter().flatten().collect(),
            )
            .unwrap(),
        }
    }

    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        self.inner.dim()
    }
}

impl<F> From<Array<F, Ix2>> for Matrix<F>
where
    F: Clone,
{
    fn from(inner: Array<F, Ix2>) -> Self {
        Self { inner }
    }
}

impl<F> AsRef<Array<F, Ix2>> for Matrix<F>
where
    F: Clone,
{
    fn as_ref(&self) -> &Array<F, Ix2> {
        &self.inner
    }
}

// impls for CanonicalSerialize and CanonicalDeserialize for Matrix<F>

impl<F> Valid for Matrix<F>
where
    F: Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}

impl<F> CanonicalSerialize for Matrix<F>
where
    F: Clone + CanonicalSerialize,
{
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        Vec::<Vec<F>>::serialize_with_mode(&self.to_vecs(), writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        Vec::<Vec<F>>::serialized_size(&self.to_vecs(), compress)
    }
}

impl<F> CanonicalDeserialize for Matrix<F>
where
    F: Clone + CanonicalDeserialize,
{
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        Vec::<Vec<F>>::deserialize_with_mode(reader, compress, validate).map(Self::from_vecs)
    }
}

/// Collapse matrix into a single vector.
pub fn col_vec_to_vec<F: Clone>(mat: &Matrix<F>) -> Vec<F> {
    mat.as_ref().iter().cloned().collect()
}

/// Expand vector into column vector (in matrix form).
pub fn vec_to_col_vec<F: Clone>(vec: &[F]) -> Matrix<F> {
    Matrix::<F> {
        inner: Array::from_shape_vec((vec.len(), 1), vec.to_vec()).unwrap(),
    }
}

pub trait Mat<Elem: Clone>: Eq + Clone + Debug {
    type Other;

    fn add(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scalar_mul(&self, other: &Self::Other) -> Self;
    fn transpose(&self) -> Self;
    fn left_mul(&self, lhs: &Matrix<Self::Other>) -> Self;
    fn right_mul(&self, rhs: &Matrix<Self::Other>) -> Self;
}

impl<F: Field> Mat<F> for Matrix<F> {
    type Other = F;

    fn add(&self, other: &Self) -> Self {
        // assert_eq!(self.len(), other.len());
        // assert_eq!(self[0].len(), other[0].len());
        Self {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    #[inline]
    fn neg(&self) -> Self {
        Self {
            inner: <Array<F, Ix2> as std::ops::Neg>::neg(self.inner.clone()),
        } // TODO check if clone is necessary
    }

    fn scalar_mul(&self, other: &Self::Other) -> Self {
        let mut res = Array::zeros(self.inner.dim());
        res.scaled_add(*other, &self.inner);
        Self { inner: res }
    }

    fn transpose(&self) -> Self {
        Self {
            inner: self.inner.clone().reversed_axes(),
        } // TODO check if clone is necessary
    }

    fn right_mul(&self, rhs: &Matrix<Self::Other>) -> Self {
        Self {
            inner: self.inner.dot(&rhs.inner),
        }
    }

    fn left_mul(&self, lhs: &Matrix<Self::Other>) -> Self {
        Self {
            inner: lhs.inner.dot(&self.inner),
        }
    }
}

impl<G: CurveGroup> Mat<Com<G>> for Matrix<Com<G>> {
    type Other = <G::Affine as AffineRepr>::ScalarField;

    fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    #[inline]
    fn neg(&self) -> Self {
        Self {
            inner: <Array<Com<G>, Ix2> as std::ops::Neg>::neg(self.inner.clone()),
        } // TODO check if clone is necessary
    }

    fn scalar_mul(&self, other: &Self::Other) -> Self {
        Self {
            inner: self.inner.map(|com| com.scalar_mul(other)),
        }
    }

    fn transpose(&self) -> Self {
        Self {
            inner: self.inner.clone().reversed_axes(),
        } // TODO check if clone is necessary
    }

    fn right_mul(&self, rhs: &Matrix<Self::Other>) -> Self {
        let dim1 = self.inner.dim();
        let dim2 = rhs.inner.dim();
        let dim_out = (dim1.0, dim2.1);

        // TODO try using ndarray's capabilities to make this more efficient
        let res = (0..dim1.0)
            .flat_map(|i| {
                let row = &self.inner.row(i);
                (0..dim2.1)
                    .map(|j| {
                        (0..dim2.0)
                            .map(|k| row[k].scalar_mul(&rhs.inner[(k, j)]).into())
                            .sum()
                    })
                    .collect::<Vec<Com<G>>>()
            })
            .collect();

        Self {
            inner: Array::from_shape_vec(dim_out, res).unwrap(),
        }
    }

    fn left_mul(&self, lhs: &Matrix<Self::Other>) -> Self {
        let dim1 = lhs.inner.dim();
        let dim2 = self.inner.dim();
        let dim_out = (dim1.0, dim2.1);

        // TODO try using ndarray's capabilities to make this more efficient
        let res = (0..dim1.0)
            .flat_map(|i| {
                let row = &lhs.inner.row(i);
                (0..dim2.1)
                    .map(|j| {
                        (0..dim2.0)
                            .map(|k| self.inner[(k, j)].scalar_mul(&row[k]))
                            .sum()
                    })
                    .collect::<Vec<Com<G>>>()
            })
            .collect();

        Self {
            inner: Array::from_shape_vec(dim_out, res).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::pairing::Pairing;
    use ark_ff::{One, UniformRand};
    use ark_std::{ops::Mul, str::FromStr, test_rng};

    type G1Affine = <F as Pairing>::G1Affine;
    type G1 = <F as Pairing>::G1;
    type G2Affine = <F as Pairing>::G2Affine;
    type G2 = <F as Pairing>::G2;
    type Fr = <F as Pairing>::ScalarField;

    // Uses an affine group generator to produce an affine group element represented by the numeric string.
    #[allow(unused_macros)]
    macro_rules! affine_group_new {
        ($gen:expr, $strnum:tt) => {
            $gen.mul(Fr::from_str($strnum).unwrap()).into_affine()
        };
    }

    // Uses an affine group generator to produce a projective group element represented by the numeric string.
    #[allow(unused_macros)]
    macro_rules! projective_group_new {
        ($gen:expr, $strnum:tt) => {
            $gen.mul(Fr::from_str($strnum).unwrap())
        };
    }

    #[test]
    fn test_col_vec_to_vec() {
        let mat = Matrix::new(&[
            [Fr::from_str("1").unwrap()],
            [Fr::from_str("2").unwrap()],
            [Fr::from_str("3").unwrap()],
        ]);
        let vec: Vec<Fr> = col_vec_to_vec(&mat);
        let exp = vec![
            Fr::from_str("1").unwrap(),
            Fr::from_str("2").unwrap(),
            Fr::from_str("3").unwrap(),
        ];
        assert_eq!(vec, exp);
    }

    #[test]
    fn test_vec_to_col_vec() {
        let vec = vec![
            Fr::from_str("1").unwrap(),
            Fr::from_str("2").unwrap(),
            Fr::from_str("3").unwrap(),
        ];
        let mat = vec_to_col_vec(&vec);
        let exp = Matrix::new(&[
            [Fr::from_str("1").unwrap()],
            [Fr::from_str("2").unwrap()],
            [Fr::from_str("3").unwrap()],
        ]);
        assert_eq!(mat, exp);
    }

    #[test]
    fn test_matrix_serde() {
        let mat = Matrix::new(&[
            [Fr::from_str("1").unwrap(), Fr::from_str("2").unwrap()],
            [Fr::from_str("3").unwrap(), Fr::from_str("4").unwrap()],
        ]);
        let mut buf = vec![];
        mat.serialize_compressed(&mut buf).unwrap();

        let mat2 = Matrix::<Fr>::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_field_matrix_left_mul_entry() {
        // 1 x 3 (row) vector
        let one = Fr::one();
        let lhs = Matrix::new(&[[one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()]]);
        // 3 x 1 (column) vector
        let rhs = Matrix::new(&[
            [Fr::from_str("4").unwrap()],
            [Fr::from_str("5").unwrap()],
            [Fr::from_str("6").unwrap()],
        ]);
        let exp = Matrix::new(&[[Fr::from_str("32").unwrap()]]);
        let res = rhs.left_mul(&lhs);

        // 1 x 1 resulting matrix
        assert_eq!(res.dim(), (1, 1));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_right_mul_entry() {
        // 1 x 3 (row) vector
        let one = Fr::one();
        let lhs = Matrix::new(&[[one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()]]);
        // 3 x 1 (column) vector
        let rhs = Matrix::new(&[
            [Fr::from_str("4").unwrap()],
            [Fr::from_str("5").unwrap()],
            [Fr::from_str("6").unwrap()],
        ]);
        let exp = Matrix::new(&[[Fr::from_str("32").unwrap()]]);
        let res = lhs.right_mul(&rhs);

        // 1 x 1 resulting matrix
        assert_eq!(res.dim(), (1, 1));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_left_mul() {
        // 2 x 3 matrix
        let one = Fr::one();
        let lhs = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
        ]);
        // 3 x 4 matrix
        let rhs = Matrix::new(&[
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
                Fr::from_str("10").unwrap(),
            ],
            [
                Fr::from_str("11").unwrap(),
                Fr::from_str("12").unwrap(),
                Fr::from_str("13").unwrap(),
                Fr::from_str("14").unwrap(),
            ],
            [
                Fr::from_str("15").unwrap(),
                Fr::from_str("16").unwrap(),
                Fr::from_str("17").unwrap(),
                Fr::from_str("18").unwrap(),
            ],
        ]);
        let exp = Matrix::new(&[
            [
                Fr::from_str("74").unwrap(),
                Fr::from_str("80").unwrap(),
                Fr::from_str("86").unwrap(),
                Fr::from_str("92").unwrap(),
            ],
            [
                Fr::from_str("173").unwrap(),
                Fr::from_str("188").unwrap(),
                Fr::from_str("203").unwrap(),
                Fr::from_str("218").unwrap(),
            ],
        ]);
        let res = rhs.left_mul(&lhs);

        // 2 x 4 resulting matrix
        assert_eq!(res.dim(), (2, 4));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_right_mul() {
        // 2 x 3 matrix
        let one = Fr::one();
        let lhs = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
        ]);
        // 3 x 4 matrix
        let rhs = Matrix::new(&[
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
                Fr::from_str("10").unwrap(),
            ],
            [
                Fr::from_str("11").unwrap(),
                Fr::from_str("12").unwrap(),
                Fr::from_str("13").unwrap(),
                Fr::from_str("14").unwrap(),
            ],
            [
                Fr::from_str("15").unwrap(),
                Fr::from_str("16").unwrap(),
                Fr::from_str("17").unwrap(),
                Fr::from_str("18").unwrap(),
            ],
        ]);
        let exp = Matrix::new(&[
            [
                Fr::from_str("74").unwrap(),
                Fr::from_str("80").unwrap(),
                Fr::from_str("86").unwrap(),
                Fr::from_str("92").unwrap(),
            ],
            [
                Fr::from_str("173").unwrap(),
                Fr::from_str("188").unwrap(),
                Fr::from_str("203").unwrap(),
                Fr::from_str("218").unwrap(),
            ],
        ]);
        let res = lhs.right_mul(&rhs);

        // 2 x 4 resulting matrix
        assert_eq!(res.dim(), (2, 4));

        assert_eq!(exp, res);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_matrix_left_mul_entry() {
        // 1 x 3 (row) vector
        let one = Fr::one();
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();

        let lhs = Matrix::new(&[[one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()]]);
        // 3 x 1 (column) vector
        let rhs = Matrix::new(&[
            [Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "4"))],
            [Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "5"))],
            [Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "6"))],
        ]);
        let exp = Matrix::new(&[[Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "32"))]]);
        let res = rhs.left_mul(&lhs);

        // 1 x 1 resulting matrix
        assert_eq!(res.dim(), (1, 1));

        assert_eq!(exp, res);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B1_matrix_right_mul_entry() {
        // 1 x 3 (row) vector
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        let lhs = Matrix::new(&[[
            Com::<G1>(G1Affine::zero(), g1gen),
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
        ]]);
        // 3 x 1 (column) vector
        let rhs = Matrix::new(&[
            [Fr::from_str("4").unwrap()],
            [Fr::from_str("5").unwrap()],
            [Fr::from_str("6").unwrap()],
        ]);
        let exp = Matrix::new(&[[Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "32"))]]);
        let res = lhs.right_mul(&rhs);

        assert_eq!(res.dim(), (1, 1));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_scalar_mul() {
        // 3 x 3 matrices
        let one = Fr::one();
        let scalar: Fr = Fr::from_str("3").unwrap();
        let mat = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
        ]);

        let exp = Matrix::new(&[
            [
                Fr::from_str("3").unwrap(),
                Fr::from_str("6").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
            [
                Fr::from_str("12").unwrap(),
                Fr::from_str("15").unwrap(),
                Fr::from_str("18").unwrap(),
            ],
            [
                Fr::from_str("21").unwrap(),
                Fr::from_str("24").unwrap(),
                Fr::from_str("27").unwrap(),
            ],
        ]);
        let res = mat.scalar_mul(&scalar);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B1_matrix_scalar_mul() {
        let scalar: Fr = Fr::from_str("3").unwrap();

        // 3 x 3 matrix of Com1 elements (0, 3)
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        let mat = Matrix::from(ndarray::Array2::from_shape_fn((3, 3), |(_, _)| {
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "1"))
        }));

        let exp = Matrix::from(ndarray::Array2::from_shape_fn((3, 3), |(_, _)| {
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3"))
        }));

        let res = mat.scalar_mul(&scalar);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B2_matrix_scalar_mul() {
        let scalar: Fr = Fr::from_str("3").unwrap();

        // 3 x 3 matrix of Com2 elements (0, 3)
        let mut rng = test_rng();
        let g2gen = G2::rand(&mut rng).into_affine();
        let mat = Matrix::from(ndarray::Array2::from_shape_fn((3, 3), |(_, _)| {
            Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "1"))
        }));

        let exp = Matrix::from(ndarray::Array2::from_shape_fn((3, 3), |(_, _)| {
            Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3"))
        }));

        let res = mat.scalar_mul(&scalar);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B1_transpose_vec() {
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        // 1 x 3 (row) vector
        let mat = Matrix::new(&[[
            Com::<G1>(G1Affine::zero(), g1gen),
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
            Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
        ]]);
        // 3 x 1 transpose (column) vector
        let exp = Matrix::new(&[
            [Com::<G1>(G1Affine::zero(), g1gen)],
            [Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2"))],
            [Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3"))],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 1));
        assert_eq!(exp, res);
    }

    #[test]
    fn test_B1_matrix_transpose() {
        // 3 x 3 matrix
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        let mat = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), g1gen),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), g1gen),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
            ],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B2_transpose_vec() {
        let mut rng = test_rng();
        let g2gen = G2::rand(&mut rng).into_affine();
        // 1 x 3 (row) vector
        let mat = Matrix::new(&[[
            Com::<G2>(G2Affine::zero(), g2gen),
            Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
            Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
        ]]);
        // 3 x 1 transpose (column) vector
        let exp = Matrix::new(&[
            [Com::<G2>(G2Affine::zero(), g2gen)],
            [Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2"))],
            [Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3"))],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 1));
        assert_eq!(exp, res);
    }

    #[test]
    fn test_B2_matrix_transpose() {
        // 3 x 3 matrix
        let mut rng = test_rng();
        let g2gen = G2::rand(&mut rng).into_affine();
        let mat = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), g2gen),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), g2gen),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
            ],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_transpose_vec() {
        // 1 x 3 (row) vector
        let one = Fr::one();
        let mat = Matrix::new(&[[one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()]]);

        // 3 x 1 transpose (column) vector
        let exp = Matrix::new(&[
            [one],
            [Fr::from_str("2").unwrap()],
            [Fr::from_str("3").unwrap()],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 1));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_transpose() {
        // 3 x 3 matrix
        let one = Fr::one();
        let mat = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [one, Fr::from_str("4").unwrap(), Fr::from_str("7").unwrap()],
            [
                Fr::from_str("2").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("8").unwrap(),
            ],
            [
                Fr::from_str("3").unwrap(),
                Fr::from_str("6").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
        ]);
        let res = mat.transpose();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_neg() {
        // 3 x 3 matrix
        let one = Fr::one();
        let mat = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [
                -one,
                -Fr::from_str("2").unwrap(),
                -Fr::from_str("3").unwrap(),
            ],
            [
                -Fr::from_str("4").unwrap(),
                -Fr::from_str("5").unwrap(),
                -Fr::from_str("6").unwrap(),
            ],
            [
                -Fr::from_str("7").unwrap(),
                -Fr::from_str("8").unwrap(),
                -Fr::from_str("9").unwrap(),
            ],
        ]);
        let res = mat.neg();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B1_matrix_neg() {
        // 3 x 3 matrix
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        let mat = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), g1gen),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), -g1gen),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "2")),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "3")),
            ],
            [
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "4")),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "5")),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "6")),
            ],
            [
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "7")),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "8")),
                Com::<G1>(G1Affine::zero(), -affine_group_new!(g1gen, "9")),
            ],
        ]);
        let res = mat.neg();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_B2_matrix_neg() {
        // 3 x 3 matrix
        let mut rng = test_rng();
        let g2gen = G2::rand(&mut rng).into_affine();
        let mat = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), g2gen),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
            ],
        ]);
        // 3 x 3 transpose matrix
        let exp = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), -g2gen),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "2")),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "3")),
            ],
            [
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "4")),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "5")),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "6")),
            ],
            [
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "7")),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "8")),
                Com::<G2>(G2Affine::zero(), -affine_group_new!(g2gen, "9")),
            ],
        ]);
        let res = mat.neg();

        assert_eq!(res.dim(), (3, 3));

        assert_eq!(exp, res);
    }

    #[test]
    fn test_field_matrix_add() {
        // 3 x 3 matrices
        let one = Fr::one();
        let lhs = Matrix::new(&[
            [one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
            [
                Fr::from_str("4").unwrap(),
                Fr::from_str("5").unwrap(),
                Fr::from_str("6").unwrap(),
            ],
            [
                Fr::from_str("7").unwrap(),
                Fr::from_str("8").unwrap(),
                Fr::from_str("9").unwrap(),
            ],
        ]);
        let rhs = Matrix::new(&[
            [
                Fr::from_str("10").unwrap(),
                Fr::from_str("11").unwrap(),
                Fr::from_str("12").unwrap(),
            ],
            [
                Fr::from_str("13").unwrap(),
                Fr::from_str("14").unwrap(),
                Fr::from_str("15").unwrap(),
            ],
            [
                Fr::from_str("16").unwrap(),
                Fr::from_str("17").unwrap(),
                Fr::from_str("18").unwrap(),
            ],
        ]);

        let exp = Matrix::new(&[
            [
                Fr::from_str("11").unwrap(),
                Fr::from_str("13").unwrap(),
                Fr::from_str("15").unwrap(),
            ],
            [
                Fr::from_str("17").unwrap(),
                Fr::from_str("19").unwrap(),
                Fr::from_str("21").unwrap(),
            ],
            [
                Fr::from_str("23").unwrap(),
                Fr::from_str("25").unwrap(),
                Fr::from_str("27").unwrap(),
            ],
        ]);
        let lr = lhs.add(&rhs);
        let rl = rhs.add(&lhs);

        assert_eq!(lr.dim(), (3, 3));

        assert_eq!(exp, lr);
        assert_eq!(lr, rl);
    }

    #[test]
    fn test_B1_matrix_add() {
        // 3 x 3 matrices
        let mut rng = test_rng();
        let g1gen = G1::rand(&mut rng).into_affine();
        let lhs = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), g1gen),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
            ],
        ]);
        let rhs = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "10")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "11")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "12")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "13")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "14")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "15")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "16")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "17")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "18")),
            ],
        ]);

        let exp = Matrix::new(&[
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "11")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "13")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "15")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "17")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "19")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "21")),
            ],
            [
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "23")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "25")),
                Com::<G1>(G1Affine::zero(), affine_group_new!(g1gen, "27")),
            ],
        ]);
        let lr = lhs.add(&rhs);
        let rl = rhs.add(&lhs);

        assert_eq!(lr.dim(), (3, 3));

        assert_eq!(exp, lr);
        assert_eq!(lr, rl);
    }

    #[test]
    fn test_B2_matrix_add() {
        // 3 x 3 matrices
        let mut rng = test_rng();
        let g2gen = G2::rand(&mut rng).into_affine();
        let lhs = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), g2gen),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
            ],
        ]);
        let rhs = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "10")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "11")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "12")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "13")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "14")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "15")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "16")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "17")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "18")),
            ],
        ]);

        let exp = Matrix::new(&[
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "11")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "13")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "15")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "17")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "19")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "21")),
            ],
            [
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "23")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "25")),
                Com::<G2>(G2Affine::zero(), affine_group_new!(g2gen, "27")),
            ],
        ]);
        let lr = lhs.add(&rhs);
        let rl = rhs.add(&lhs);

        assert_eq!(lr.dim(), (3, 3));

        assert_eq!(exp, lr);
        assert_eq!(lr, rl);
    }
}
