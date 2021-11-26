use ark_ec::{PairingEngine};
use ark_ff::{Zero, One};
use ark_std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign}
};


/// B1,B2,BT forms a bilinear group for GS commitments
pub trait B1: Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
//    + AddAssign<Self>
{}
pub trait B2: Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
//    + AddAssign<Self>
{}
pub trait BT<C1: B1, C2: B2>: 
    Eq
    + Clone
    + Debug
// TODO: What's multiplication for commitment group BT?
//    + One
//    + Mul<Com1<E>, Com2<E>>
//    + MulAssign<Self>
{
    fn pairing(x: C1, y: C2) -> Self;
}

// SXDH instantiation's bilinear group for commitments

// TODO: Expose randomness? (see example data_structures in Arkworks)
#[derive(Clone, Debug)]
pub struct Com1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);
#[derive(Clone, Debug)]
pub struct Com2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);
#[derive(Clone, Debug)]
pub struct ComT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);

// TODO: Combine this into a macro for Com1<E>: B1, Com2<E>: B2, ComT<E>: BT<B1,B2>
/*
macro_rules! impl_Com {
    (for $($t:ty),+) => {
        $(impl<E: PairingEngine> PartialEq for Com$t<E> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0 && self.1 == other.1
            }
        })*
        $(impl<E: PairingEngine> Eq for Com$t<E> {})*
        $(impl<E: PairingEngine> B$t for Com$t<E> {})*
    }
}
impl_Com!(for 1, 2);
*/

// Com1 implements B1
impl<E: PairingEngine> PartialEq for Com1<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com1<E> {}
impl<E: PairingEngine> Add<Com1<E>> for Com1<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}
impl<E: PairingEngine> Zero for Com1<E> {
    fn zero() -> Com1<E> {
        Com1::<E> (
            E::G1Affine::zero(),
            E::G1Affine::zero()
        )
    }

    fn is_zero(&self) -> bool {
        *self == Com1::<E>::zero()
    }
}
impl<E: PairingEngine> B1 for Com1<E> {}


// Com2 implements B2
impl<E: PairingEngine> PartialEq for Com2<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com2<E> {}
impl<E: PairingEngine> Add<Com2<E>> for Com2<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}
impl<E: PairingEngine> Zero for Com2<E> {
    fn zero() -> Com2<E> {
        Com2::<E> (
            E::G2Affine::zero(),
            E::G2Affine::zero()
        )
    }

    fn is_zero(&self) -> bool {
        *self == Com2::<E>::zero()
    }
}
impl<E: PairingEngine> B2 for Com2<E> {}

// ComT implements BT<B1, B2>
impl<E: PairingEngine> PartialEq for ComT<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: PairingEngine> Eq for ComT<E> {}
/*
impl<E: PairingEngine> One for ComT<E> {
    fn one() -> ComT<E> {
        ComT::<E> (
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one()
        )
    }
}
*/
impl<E: PairingEngine> BT<Com1<E>, Com2<E>> for ComT<E> {
    #[inline]
    /// B_pairing takes entry-wise pairing products
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
        )
    }
}

#[cfg(test)]
mod tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ff::{UniformRand, Zero, One};
    use ark_ec::{ProjectiveCurve, PairingEngine};
    use ark_std::test_rng;

    use crate::data_structures::*;

    type G1Projective = <F as PairingEngine>::G1Projective;
    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type GT = <F as PairingEngine>::Fqk;

    
    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_rand() {
        let mut rng = test_rng();
        let b1 = Com1::<F>(
            G1Projective::rand(&mut rng).into_affine(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = Com2::<F>(
            G2Projective::rand(&mut rng).into_affine(),
            G2Projective::rand(&mut rng).into_affine()
        );
        let bt = ComT::pairing(b1.clone(), b2.clone());

        assert_eq!(bt.0, F::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.0.clone()));
        assert_eq!(bt.1, F::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.1.clone()));
        assert_eq!(bt.2, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.0.clone()));
        assert_eq!(bt.3, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
    
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G1() {
        let mut rng = test_rng();
        let b1 = Com1::<F>(
            G1Affine::zero(),
            G1Affine::zero()
        );
        let b2 = Com2::<F>(
            G2Projective::rand(&mut rng).into_affine(),
            G2Projective::rand(&mut rng).into_affine()
        );        
        let bt = ComT::pairing(b1.clone(), b2.clone());

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G2() {
        let mut rng = test_rng();
        let b1 = Com1::<F>(
            G1Projective::rand(&mut rng).into_affine(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = Com2::<F>(
            G2Affine::zero(),
            G2Affine::zero()
        );        
        let bt = ComT::pairing(b1.clone(), b2.clone());

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_commit() {
        let mut rng = test_rng();
        let b1 = Com1::<F>(
            G1Affine::zero(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = Com2::<F>(
            G2Affine::zero(),
            G2Projective::rand(&mut rng).into_affine()
        );
        let bt = ComT::pairing(b1.clone(), b2.clone());

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
    }
}