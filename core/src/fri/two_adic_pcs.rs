extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use cuda_common::{d_buffer::DeviceBuffer, memory_manager::MemTracker};
use itertools::{Itertools, izip};
use p3_challenger::FieldChallenger;
use p3_commit::{OpenedValues, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{Field, FieldAlgebra, TwoAdicField, dot_product};
use p3_fri::{BatchOpening, FriProof};
use p3_util::{linear_map::LinearMap, log2_strict_usize};
use tracing::info_span;

use super::prover::prove;
use crate::{
    base::{DeviceMatrix, DevicePoly, MatrixDimensions},
    dft::GpuLde,
    fri::ops::{
        compute_inverse_denominators_on_gpu, compute_non_bitrev_inverse_denominators_on_gpu,
        matrix_evaluate, reduce_matrix_quotient_acc,
    },
    merkle_tree::{GpuMerkleTree, GpuMerkleTreeMmcs},
    pcs::GpuPcs,
    types::{Challenge, ChallengeMmcs, Challenger, DigestHash, EF, F, ValMmcs},
};

pub struct GpuTwoAdicFriPcs {
    fri: FriConfig,
}

#[derive(Debug, Clone)]
pub struct FriConfig {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

impl FriConfig {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}

impl GpuTwoAdicFriPcs {
    pub fn new(fri: FriConfig) -> Self {
        Self { fri }
    }

    pub fn fri_config(&self) -> &FriConfig {
        &self.fri
    }
}

impl GpuPcs for GpuTwoAdicFriPcs {
    type Domain = TwoAdicMultiplicativeCoset<F>;
    type Commitment = DigestHash;
    type ProverData = GpuMerkleTree;
    type Proof = FriProof<EF, ChallengeMmcs, F, Vec<BatchOpening<F, ValMmcs>>>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        TwoAdicMultiplicativeCoset { log_n, shift: F::ONE }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, DeviceMatrix<F>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let _mem = MemTracker::start("commit");

        let ldes: Vec<GpuLde> = evaluations
            .into_iter()
            .map(|(domain, trace)| {
                assert_eq!(domain.size(), trace.height());
                let shift = F::GENERATOR / domain.shift;
                GpuLde::new(trace, self.fri.log_blowup, shift)
            })
            .collect();

        GpuMerkleTreeMmcs::commit(ldes)
    }

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
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let mem = MemTracker::start("open");

        // For each matrix (with columns p_i) and opening point z, we want:
        // reduced[X] += alpha_offset * inv_denom[X] * (sum_i{alpha^i * p_i[X]} - sum_i{alpha^i * y[i]})

        // Batch combination challenge.
        let alpha: EF = challenger.sample_ext_element();
        tracing::debug!("alpha sampled in gpu_pcs::open(): {:?}", alpha);

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = data.leaves.iter().collect_vec();
                assert_eq!(mats.len(), points.len());
                (mats, points)
            })
            .collect_vec();

        // Hight of LDE matrices.
        let heights_and_points = mats_and_points
            .iter()
            .map(|(mats, points)| (mats.iter().map(|m| m.height()).collect_vec(), *points))
            .collect_vec();

        let global_max_height =
            heights_and_points.iter().flat_map(|(mats, _)| mats.iter().copied()).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Height of trace matrices.
        let trace_heights_and_points = mats_and_points
            .iter()
            .map(|(mats, points)| (mats.iter().map(|m| m.trace_height()).collect_vec(), *points))
            .collect_vec();

        let mut inv_denoms = LinearMap::new();
        let mut last_shift = None;
        let all_opened_values: OpenedValues<EF> = info_span!("evaluate matrix").in_scope(|| {
            mats_and_points
                .iter()
                .map(|(mats, points)| {
                    // Matrices that have same shift are grouped together.
                    let mut mats_by_shift: BTreeMap<F, Vec<usize>> = BTreeMap::new();
                    mats.iter().enumerate().for_each(|(i, mat)| {
                        if let Some(indices) = mats_by_shift.get_mut(&mat.shift()) {
                            indices.push(i);
                        } else {
                            mats_by_shift.insert(mat.shift(), vec![i]);
                        }
                    });

                    // BTreeMap guarantees eval_orders is deterministic
                    let eval_orders = mats_by_shift
                        .into_iter()
                        .flat_map(|(_, indices)| indices.into_iter())
                        .collect_vec();

                    let openings = eval_orders
                        .iter()
                        .map(|&idx| {
                            let mat = mats[idx];
                            let points_for_mat = &points[idx];

                            let shift = F::GENERATOR;

                            // If last_shift is not set or not equal to current shift.
                            if last_shift.is_none() || last_shift.unwrap() != shift {
                                // For each point and each log height, we will precompute 1/(X - z)
                                // for subgroup of order 2^log_height.
                                // TODO: reduce inv_denoms' size
                                inv_denoms = compute_non_bitrev_inverse_denominators_on_gpu(
                                    trace_heights_and_points.as_slice(),
                                    shift,
                                );
                            }
                            last_shift = Some(shift);

                            // Use Barycentric interpolation to evaluate the matrix at the given point.
                            // Matrix is evaluations on domain shift * H = { shift * g^i }.
                            points_for_mat
                                .iter()
                                .map(|z| {
                                    let trace_height = mat.trace_height();
                                    let log_height = log2_strict_usize(trace_height);
                                    let low_coset_mat = mat.get_lde(trace_height);
                                    let g = F::two_adic_generator(log_height);
                                    let inv_denom =
                                        inv_denoms.get(z).unwrap()[log_height].as_ref().unwrap();
                                    matrix_evaluate(
                                        &low_coset_mat,
                                        inv_denom,
                                        *z,
                                        shift,
                                        g,
                                        trace_height,
                                    )
                                    .unwrap()
                                })
                                .collect_vec()
                        })
                        .collect_vec();

                    let mut original_orders = vec![0; eval_orders.len()];
                    for (reorder_idx, original_idx) in eval_orders.iter().enumerate() {
                        original_orders[*original_idx] = reorder_idx;
                    }

                    original_orders
                        .iter()
                        .map(|reorder_idx| openings[*reorder_idx].clone())
                        .collect_vec()
                })
                .collect()
        });
        drop(inv_denoms);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms =
            compute_inverse_denominators_on_gpu(heights_and_points.as_slice(), F::GENERATOR);

        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        info_span!("build fri inputs").in_scope(|| {
            for ((mats, points), openings_for_round) in
                mats_and_points.iter().zip_eq(all_opened_values.iter())
            {
                for (mat, points_for_mat, openings_for_mat) in
                    izip!(mats.iter(), points.iter(), openings_for_round.iter())
                {
                    let log_height = log2_strict_usize(mat.height());
                    let reduced_opening = reduced_openings[log_height].get_or_insert_with(|| {
                        DevicePoly::new(true, DeviceBuffer::<EF>::with_capacity(mat.height()))
                    });

                    let mat = mat.get_lde(mat.height());

                    for (z, openings) in points_for_mat.iter().zip(openings_for_mat.iter()) {
                        let inv_denom = inv_denoms.get(z).unwrap();
                        let m_z = dot_product(alpha.powers(), openings.iter().copied());
                        reduce_matrix_quotient_acc(
                            reduced_opening,
                            &mat,
                            inv_denom,
                            m_z,
                            alpha,
                            num_reduced[log_height],
                            num_reduced[log_height] == 0,
                        )
                        .unwrap();
                        num_reduced[log_height] += mat.width();
                    }
                }
            }
        });

        let fri_inputs = reduced_openings.into_iter().rev().flatten().collect_vec();
        mem.tracing_info("after fri inputs");
        assert!(!fri_inputs.is_empty());
        assert!(
            fri_inputs.iter().tuple_windows().all(|(l, r)| l.len() >= r.len()),
            "Inputs are not sorted in descending order of length."
        );

        let fri_proof = prove(&self.fri, fri_inputs, challenger, rounds, log_global_max_height);

        (all_opened_values, fri_proof)
    }
}
