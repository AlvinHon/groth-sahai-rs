use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, Zero};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::Matrix;

#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Com<G: CurveGroup>(pub G::Affine, pub G::Affine);

// Equality for Com group
impl<G: CurveGroup> PartialEq for Com<G> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<G: CurveGroup> Eq for Com<G> {}

// Addition for Com group
impl<G: CurveGroup> Add<Com<G>> for Com<G> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self((self.0 + other.0).into(), (self.1 + other.1).into())
    }
}
impl<G: CurveGroup> AddAssign<Com<G>> for Com<G> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = Self((self.0 + other.0).into(), (self.1 + other.1).into());
    }
}
impl<G: CurveGroup> Neg for Com<G> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(
            (-(self.0.into_group())).into(),
            (-(self.1.into_group())).into(),
        )
    }
}
impl<G: CurveGroup> Sub<Com<G>> for Com<G> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}
impl<G: CurveGroup> SubAssign<Com<G>> for Com<G> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}
impl<G: CurveGroup> Mul<&<G::Affine as AffineRepr>::ScalarField> for &Com<G> {
    type Output = Com<G>;

    fn mul(self, rhs: &<G::Affine as AffineRepr>::ScalarField) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

// Entry-wise scalar point-multiplication
impl<G: CurveGroup> MulAssign<&<G::Affine as AffineRepr>::ScalarField> for Com<G> {
    fn mul_assign(&mut self, rhs: &<G::Affine as AffineRepr>::ScalarField) {
        self.0 = (self.0.into_group() * rhs).into_affine();
        self.1 = (self.1.into_group() * rhs).into_affine();
    }
}

impl<G: CurveGroup> Sum for Com<G> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<G: CurveGroup> Zero for Com<G> {
    #[inline]
    fn zero() -> Self {
        Self(G::Affine::zero(), G::Affine::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<G: CurveGroup> UniformRand for Com<G> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(G::Affine::rand(rng), G::Affine::rand(rng))
    }
}

impl<G: CurveGroup> From<Matrix<G::Affine>> for Com<G> {
    fn from(mat: Matrix<G::Affine>) -> Self {
        Self(mat[(0, 0)], mat[(1, 0)])
    }
}

impl<G: CurveGroup> Com<G> {
    pub fn new(x: G::Affine, y: G::Affine) -> Self {
        Self(x, y)
    }

    pub fn as_col_vec(&self) -> Matrix<G::Affine> {
        ndarray::arr2(&[[self.0], [self.1]])
    }

    pub fn as_vec(&self) -> Vec<G::Affine> {
        vec![self.0, self.1]
    }

    #[inline]
    pub fn linear_map(x: &G::Affine) -> Self {
        Self(G::Affine::zero(), *x)
    }

    #[inline]
    pub fn batch_linear_map(x_vec: &[G::Affine]) -> Vec<Self> {
        x_vec.iter().map(Self::linear_map).collect::<Vec<Self>>()
    }

    /// Compute a commitment group element:
    /// - = xu, where u = u_2 + (O, P) for G1
    /// - = yv, where v = v_2 + (O, P) for G2
    #[inline]
    pub fn scalar_linear_map(
        x: &<G::Affine as AffineRepr>::ScalarField,
        uv: Self,
        p: &G::Affine,
    ) -> Self {
        (uv + Self::linear_map(p)).scalar_mul(x)
    }

    /// Compute a vector of commitment group elements:
    /// - = xu, where u = u_2 + (O, P) for G1
    /// - = yv, where v = v_2 + (O, P) for G2
    #[inline]
    pub fn batch_scalar_linear_map(
        x_vec: &[<G::Affine as AffineRepr>::ScalarField],
        uv: Self,
        p: &G::Affine,
    ) -> Vec<Self> {
        x_vec
            .iter()
            .map(|elem| Self::scalar_linear_map(elem, uv, p))
            .collect::<Vec<Self>>()
    }

    pub fn scalar_mul(&self, rhs: &<G::Affine as AffineRepr>::ScalarField) -> Self {
        let mut s1p = self.0.into_group();
        let mut s2p = self.1.into_group();
        s1p *= *rhs;
        s2p *= *rhs;
        Self(s1p.into_affine(), s2p.into_affine())
    }
}
