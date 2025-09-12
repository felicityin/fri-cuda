extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use cuda_common::copy::MemCopyD2H;
use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_field::{Field, FieldExtensionAlgebra, TwoAdicField};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof};
use p3_util::log2_strict_usize;
use tracing::info_span;

use crate::{
    base::{DevicePoly, ExtendedLagrangeCoeff},
    dft::GpuLde,
    fri::{
        FriConfig,
        ops::{fri_ext_poly_to_base_matrix, fri_fold},
    },
    merkle_tree::{GpuMerkleTree, GpuMerkleTreeMmcs},
    types::{ChallengeMmcs, Challenger, DigestHash, EF, F, ValMmcs},
};

pub fn prove(
    config: &FriConfig,
    inputs: Vec<DevicePoly<EF, ExtendedLagrangeCoeff>>,
    challenger: &mut Challenger,
    rounds: Vec<(&GpuMerkleTree, Vec<Vec<EF>>)>,
    log_global_max_height: usize,
) -> FriProof<EF, ChallengeMmcs, F, Vec<BatchOpening<F, ValMmcs>>> {
    let log_max_height = log2_strict_usize(inputs[0].len());

    // Commit to the folded polynomials.
    let commit_phase_result = commit_phase_on_gpu(config, inputs, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let extra_query_index_bits = 0;
    let query_proofs = info_span!("query phase").in_scope(|| {
        let query_indices =
            iter::repeat_with(|| challenger.sample_bits(log_max_height + extra_query_index_bits))
                .take(config.num_queries)
                .collect_vec();

        let mut input_proofs_for_rounds = rounds
            .iter()
            .map(|(tree, _)| {
                // For each round, query multiple indices at once.
                let log_max_height = log2_strict_usize(tree.get_max_height());
                let reduced_indices = query_indices
                    .iter()
                    .map(|index| {
                        let bits_reduced = log_global_max_height - log_max_height;
                        index >> bits_reduced
                    })
                    .collect_vec();
                tree.open_batch_at_multiple_indices(&reduced_indices)
                    .unwrap()
                    .into_iter()
                    .map(|(opened_values, opening_proof)| BatchOpening {
                        opened_values,
                        opening_proof,
                    })
                    .collect_vec()
            })
            .collect_vec();

        let input_proofs_rev: Vec<_> = query_indices
            .iter()
            // Reverse the indices to pop.
            .rev()
            .map(|_| {
                // The opening proof for last index comes at the end,
                // therefore we can get it by popping to avoid clone.
                input_proofs_for_rounds
                    .iter_mut()
                    .map(|input_proofs| input_proofs.pop().unwrap())
                    .collect()
            })
            .collect_vec();

        let commit_phase_openings_rev = answer_batch_queries_on_gpu(
            &commit_phase_result.data,
            query_indices
                .into_iter()
                .map(|index| index >> extra_query_index_bits)
                .collect_vec()
                .as_slice(),
        );

        input_proofs_rev
            .into_iter()
            .rev()
            .zip(commit_phase_openings_rev.into_iter().rev())
            .map(|(input_proof, commit_phase_openings)| QueryProof {
                input_proof,
                commit_phase_openings,
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

pub struct CommitPhaseGpuResult {
    pub commits: Vec<DigestHash>,
    pub data: Vec<GpuMerkleTree>,
    pub final_poly: EF,
}

pub fn commit_phase_on_gpu(
    config: &FriConfig,
    inputs: Vec<DevicePoly<EF, ExtendedLagrangeCoeff>>,
    challenger: &mut Challenger,
) -> CommitPhaseGpuResult {
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data: Vec<GpuMerkleTree> = vec![];

    while folded.len() > config.blowup() {
        // folded is converted to a matrix over base field with width = 8 and height = folded.len()/2
        let folded_as_matrix = fri_ext_poly_to_base_matrix(&folded).unwrap();
        let lde = GpuLde::new(folded_as_matrix, 0, F::GENERATOR);
        let (commit, prover_data) = GpuMerkleTreeMmcs::commit(vec![lde]);
        challenger.observe(commit);

        commits.push(commit);
        data.push(prover_data);

        let log_folded_len = log2_strict_usize(folded.len());
        let beta: EF = challenger.sample_ext_element();
        tracing::debug!("beta at gpu pcs (layer = {}): {:?}", log_folded_len, beta);

        let fri_input = inputs_iter.next_if(|v| v.len() == folded.len() / 2);
        let g_inv = EF::two_adic_generator(log_folded_len).inverse();

        folded = fri_fold(folded, fri_input, beta, g_inv).unwrap();
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), config.blowup());
    let folded_on_host: Vec<EF> = folded.coeff.to_host().unwrap();
    let final_poly = folded_on_host[0];
    for x in folded_on_host {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseGpuResult { commits, data, final_poly }
}

pub fn answer_batch_queries_on_gpu(
    commit_phase_commits: &[GpuMerkleTree],
    indices: &[usize],
) -> Vec<Vec<CommitPhaseProofStep<EF, ChallengeMmcs>>> {
    let mut proofs_per_phase = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let pair_indices = indices
                .iter()
                .map(|index| {
                    let index_i = index >> i;
                    index_i >> 1
                })
                .collect::<Vec<_>>();

            commit
                .open_batch_at_multiple_indices(&pair_indices)
                .unwrap()
                .into_iter()
                .zip(indices.iter())
                .map(|((opened_base_values, opening_proof), index)| {
                    let index_i = index >> i;
                    let index_i_sibling = index_i ^ 1;

                    let opened_ext_values: Vec<Vec<EF>> = opened_base_values
                        .into_iter()
                        .map(|row| row.chunks(4).map(EF::from_base_slice).collect())
                        .collect();
                    let mut opened_rows = opened_ext_values;

                    assert_eq!(opened_rows.len(), 1);

                    let opened_row = opened_rows.pop().unwrap();
                    assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
                    let sibling_value = opened_row[index_i_sibling % 2];

                    CommitPhaseProofStep { sibling_value, opening_proof }
                })
                .collect_vec()
        })
        .collect_vec();

    indices
        .iter()
        .rev()
        .map(|_| proofs_per_phase.iter_mut().map(|proofs| proofs.pop().unwrap()).collect_vec())
        .collect_vec()
}
