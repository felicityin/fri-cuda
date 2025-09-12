//! Traits for polynomial commitment schemes.

extern crate alloc;
use alloc::vec::Vec;

use p3_commit::{OpenedValues, PolynomialSpace, Val};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::base::DeviceMatrix;
use crate::types::{Challenge, Challenger};

pub trait GpuPcs {
    type Domain: PolynomialSpace;

    /// The commitment that's sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData: Clone;

    /// The opening argument.
    type Proof: Clone + Serialize + DeserializeOwned;

    /// This should return a coset domain (s.t. Domain::next_point returns Some)
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, DeviceMatrix<Val<Self::Domain>>)>,
    ) -> (Self::Commitment, Self::ProverData);

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);
}
