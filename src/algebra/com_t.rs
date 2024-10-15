use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::{UniformRand, Zero};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use super::{matrix::Matrix, Com1, Com2};

/// Target [`BT`](crate::data_structures::BT) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug)]
pub struct ComT<E: Pairing>(
    pub PairingOutput<E>,
    pub PairingOutput<E>,
    pub PairingOutput<E>,
    pub PairingOutput<E>,
);

// ComT<Com1, Com2> is an instantiation of BT<B1, B2>
impl<E: Pairing> PartialEq for ComT<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: Pairing> Eq for ComT<E> {}

impl<E: Pairing> Add<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}
impl<E: Pairing> Zero for ComT<E> {
    #[inline]
    fn zero() -> Self {
        Self(
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}
impl<E: Pairing> AddAssign<ComT<E>> for ComT<E> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
        self.3 += other.3;
    }
}
impl<E: Pairing> Neg for ComT<E> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1, -self.2, -self.3)
    }
}
impl<E: Pairing> Sub<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}
impl<E: Pairing> SubAssign<ComT<E>> for ComT<E> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
        self.2 -= other.2;
        self.3 -= other.3;
    }
}
impl<E: Pairing> From<Matrix<PairingOutput<E>>> for ComT<E> {
    fn from(mat: Matrix<PairingOutput<E>>) -> Self {
        Self(mat[(0, 0)], mat[(0, 1)], mat[(1, 0)], mat[(1, 1)])
    }
}

impl<E: Pairing> UniformRand for ComT<E> {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(
            PairingOutput::rand(rng),
            PairingOutput::rand(rng),
            PairingOutput::rand(rng),
            PairingOutput::rand(rng),
        )
    }
}

impl<E: Pairing> Sum for ComT<E> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<E: Pairing> ComT<E> {
    #[inline]
    pub fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            E::pairing(x.0, y.0),
            E::pairing(x.0, y.1),
            E::pairing(x.1, y.0),
            E::pairing(x.1, y.1),
        )
    }

    #[inline]
    pub fn pairing_sum(x_vec: &[Com1<E>], y_vec: &[Com2<E>]) -> Self {
        assert_eq!(x_vec.len(), y_vec.len());
        Self(
            E::multi_pairing(x_vec.iter().map(|x| x.0), y_vec.iter().map(|y| y.0)),
            E::multi_pairing(x_vec.iter().map(|x| x.0), y_vec.iter().map(|y| y.1)),
            E::multi_pairing(x_vec.iter().map(|x| x.1), y_vec.iter().map(|y| y.0)),
            E::multi_pairing(x_vec.iter().map(|x| x.1), y_vec.iter().map(|y| y.1)),
        )
    }

    pub fn as_matrix(&self) -> Matrix<PairingOutput<E>> {
        ndarray::arr2(&[[self.0, self.1], [self.2, self.3]])
    }

    #[allow(non_snake_case)]
    #[inline]
    pub fn linear_map_PPE(z: &PairingOutput<E>) -> Self {
        Self(
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
            *z,
        )
    }

    // #[inline]
    // fn linear_map_MSMEG1(z: &E::G1Affine, key: &CRS<E>) -> Self {
    //     Self::pairing(
    //         Com1::<E>::linear_map(z),
    //         Com2::<E>::scalar_linear_map(&E::ScalarField::one(), key),
    //     )
    // }

    // #[inline]
    // fn linear_map_MSMEG2(z: &E::G2Affine, key: &CRS<E>) -> Self {
    //     Self::pairing(
    //         Com1::<E>::scalar_linear_map(&E::ScalarField::one(), key),
    //         Com2::<E>::linear_map(z),
    //     )
    // }

    // #[inline]
    // fn linear_map_quad(z: &E::ScalarField, key: &CRS<E>) -> Self {
    //     Self::pairing(
    //         Com1::<E>::scalar_linear_map(&E::ScalarField::one(), key),
    //         Com2::<E>::scalar_linear_map(&E::ScalarField::one(), key).scalar_mul(z),
    //     )
    // }
}
