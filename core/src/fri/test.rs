#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use itertools::Itertools;
    use itertools::izip;
    use p3_challenger::CanObserve;
    use p3_challenger::FieldChallenger;
    use p3_commit::Mmcs;
    use p3_commit::Pcs;
    use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
    use p3_dft::Radix2DitParallel;
    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::Field;
    use p3_fri::FriConfig as P3FriConfig;
    use p3_fri::TwoAdicFriPcs;
    use p3_matrix::{Matrix, bitrev::BitReversableMatrix, dense::RowMajorMatrix};
    use p3_merkle_tree::MerkleTree;
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
    use zkm_primitives::poseidon2_init;

    use crate::types::ChallengeMmcs;
    use crate::types::Challenger;
    use crate::types::EF;
    use crate::{
        GpuPcs,
        data_transporter::transport_matrix_to_device,
        fri::{FriConfig, GpuTwoAdicFriPcs},
        types::{DigestHash, F, MyCompress, MyHash, ValMmcs},
    };

    type CpuDft = Radix2DitParallel<F>;
    type CpuPcs = TwoAdicFriPcs<F, CpuDft, ValMmcs, ChallengeMmcs>;

    #[test]
    fn test_fri_pcs_single() {
        const LOG_BLOWUP: usize = 1;
        const COLS: usize = 8;
        let log_degree = 12;
        let rows = 1 << log_degree;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, COLS);
        let d_matrix = transport_matrix_to_device(&h_matrix);

        let challenger = Challenger::new(poseidon2_init());
        let config = FriConfig { log_blowup: LOG_BLOWUP, num_queries: 10, proof_of_work_bits: 8 };

        // ------------------------ commit ------------------------

        let gpu_pcs = GpuTwoAdicFriPcs::new(config.clone());
        let domain = gpu_pcs.natural_domain_for_degree(h_matrix.height());
        let (gpu_root, gpu_data) = gpu_pcs.commit(vec![(domain, d_matrix)]);

        let (cpu_root, cpu_data) =
            p3_commit(CpuDft::default(), LOG_BLOWUP, vec![(domain, h_matrix)]);

        assert_eq!(gpu_root, cpu_root);

        // ------------------------ open ------------------------

        // cpu open

        let mut p_cpu_challenger = challenger.clone();
        p_cpu_challenger.observe(cpu_root.clone());
        let zeta: EF = p_cpu_challenger.sample_ext_element();

        let cpu_pcs = create_cpu_pcs(&config);
        let (cpu_open, cpu_proof) =
            cpu_pcs.open(vec![(&cpu_data, vec![vec![zeta]])], &mut p_cpu_challenger);

        // verify

        let mut v_cpu_challenger = challenger.clone();
        v_cpu_challenger.observe(cpu_root);
        let zeta: EF = v_cpu_challenger.sample_ext_element();

        cpu_pcs
            .verify(
                vec![(cpu_root, vec![(domain, vec![(zeta, cpu_open[0][0][0].clone())])])],
                &cpu_proof,
                &mut v_cpu_challenger,
            )
            .unwrap();

        // gpu open

        let mut p_gpu_challenger = challenger.clone();
        p_gpu_challenger.observe(gpu_root.clone());
        let zeta: EF = p_gpu_challenger.sample_ext_element();

        let (gpu_open, gpu_proof) =
            gpu_pcs.open(vec![(&gpu_data, vec![vec![zeta]])], &mut p_gpu_challenger);

        // verify

        let mut v_gpu_challenger = challenger.clone();
        v_gpu_challenger.observe(gpu_root);
        let zeta: EF = v_gpu_challenger.sample_ext_element();

        cpu_pcs
            .verify(
                vec![(gpu_root, vec![(domain, vec![(zeta, gpu_open[0][0][0].clone())])])],
                &gpu_proof,
                &mut v_gpu_challenger,
            )
            .unwrap();
    }

    #[test]
    fn test_fri_pcs_multi() {
        let degrees = (3..3 + 8).collect::<Vec<_>>();
        test_fri_pcs(degrees);
    }

    #[test]
    fn test_fri_pcs_multi_rev() {
        let degrees = (3..3 + 8).rev().collect::<Vec<_>>();
        test_fri_pcs(degrees);
    }

    fn test_fri_pcs(degrees: Vec<i32>) {
        const LOG_BLOWUP: usize = 1;

        let log_degrees_by_round = [degrees];
        let num_rounds = log_degrees_by_round.len();
        let mut rng = ChaCha20Rng::seed_from_u64(0);

        let config = FriConfig { log_blowup: LOG_BLOWUP, num_queries: 10, proof_of_work_bits: 8 };
        let cpu_pcs = create_cpu_pcs(&config);
        let gpu_pcs = GpuTwoAdicFriPcs::new(config);
        let challenger = Challenger::new(poseidon2_init());

        // ------------------------cpu commit ------------------------

        let cpu_domains_and_polys_by_round = log_degrees_by_round
            .iter()
            .map(|log_degrees| {
                log_degrees
                    .iter()
                    .map(|&log_degree| {
                        let d = 1 << log_degree;
                        // random width 5-15
                        // let width = 5 + rng.gen_range(0..=10);
                        let width = 8;
                        (
                            gpu_pcs.natural_domain_for_degree(d),
                            RowMajorMatrix::<F>::rand(&mut rng, d, width),
                        )
                    })
                    .collect_vec()
            })
            .collect_vec();

        let (cpu_commits_by_round, cpu_data_by_round): (Vec<_>, Vec<_>) =
            cpu_domains_and_polys_by_round
                .clone()
                .into_iter()
                .map(|domains_and_polys| {
                    p3_commit(CpuDft::default(), LOG_BLOWUP, domains_and_polys)
                })
                .unzip();
        assert_eq!(cpu_commits_by_round.len(), num_rounds);
        assert_eq!(cpu_data_by_round.len(), num_rounds);

        // ------------------------gpu commit ------------------------

        let gpu_domains_and_polys_by_round = cpu_domains_and_polys_by_round
            .iter()
            .map(|domains_and_polys| {
                domains_and_polys
                    .iter()
                    .map(|(domain, poly)| (domain.clone(), transport_matrix_to_device(&poly)))
                    .collect_vec()
            })
            .collect_vec();

        let (gpu_commits_by_round, gpu_data_by_round): (Vec<_>, Vec<_>) =
            gpu_domains_and_polys_by_round
                .clone()
                .into_iter()
                .map(|domains_and_polys| gpu_pcs.commit(domains_and_polys))
                .unzip();
        assert_eq!(gpu_commits_by_round.len(), num_rounds);
        assert_eq!(gpu_data_by_round.len(), num_rounds);

        assert_eq!(cpu_commits_by_round, gpu_commits_by_round);

        // ------------------------cpu open ------------------------

        let mut p_cpu_challenger = challenger.clone();
        p_cpu_challenger.observe_slice(&cpu_commits_by_round);
        let zeta: EF = p_cpu_challenger.sample_ext_element();

        let cpu_points_by_round = log_degrees_by_round
            .iter()
            .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
            .collect_vec();

        let cpu_data_and_points = cpu_data_by_round.iter().zip(cpu_points_by_round).collect();
        let (cpu_opening_by_round, proof) =
            cpu_pcs.open(cpu_data_and_points, &mut p_cpu_challenger);
        assert_eq!(cpu_opening_by_round.len(), num_rounds);

        // ------------------------ verify ------------------------

        let mut v_cpu_challenger = challenger.clone();
        v_cpu_challenger.observe_slice(&cpu_commits_by_round);
        let zeta: EF = v_cpu_challenger.sample_ext_element();

        let cpu_commits_and_claims_by_round =
            izip!(cpu_commits_by_round, cpu_domains_and_polys_by_round, cpu_opening_by_round)
                .map(|(commit, domains_and_polys, openings)| {
                    let claims = domains_and_polys
                        .iter()
                        .zip(openings)
                        .map(|((domain, _), mat_openings)| {
                            (*domain, vec![(zeta, mat_openings[0].clone())])
                        })
                        .collect_vec();
                    (commit, claims)
                })
                .collect_vec();
        assert_eq!(cpu_commits_and_claims_by_round.len(), num_rounds);

        cpu_pcs.verify(cpu_commits_and_claims_by_round, &proof, &mut v_cpu_challenger).unwrap();

        // ------------------------gpu open ------------------------

        let mut p_gpu_challenger = challenger.clone();
        p_gpu_challenger.observe_slice(&gpu_commits_by_round);
        let zeta: EF = p_gpu_challenger.sample_ext_element();

        let gpu_points_by_round = log_degrees_by_round
            .iter()
            .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
            .collect_vec();

        let data_and_points = gpu_data_by_round.iter().zip(gpu_points_by_round).collect();
        let (gpu_opening_by_round, proof) = gpu_pcs.open(data_and_points, &mut p_gpu_challenger);
        assert_eq!(gpu_opening_by_round.len(), num_rounds);

        // ------------------------ verify ------------------------

        let mut v_gpu_challenger = challenger.clone();
        v_gpu_challenger.observe_slice(&gpu_commits_by_round);
        let zeta: EF = v_gpu_challenger.sample_ext_element();

        let gpu_commits_and_claims_by_round =
            izip!(gpu_commits_by_round, gpu_domains_and_polys_by_round, gpu_opening_by_round)
                .map(|(commit, domains_and_polys, openings)| {
                    let claims = domains_and_polys
                        .iter()
                        .zip(openings)
                        .map(|((domain, _), mat_openings)| {
                            (*domain, vec![(zeta, mat_openings[0].clone())])
                        })
                        .collect_vec();
                    (commit, claims)
                })
                .collect_vec();
        assert_eq!(gpu_commits_and_claims_by_round.len(), num_rounds);

        cpu_pcs.verify(gpu_commits_and_claims_by_round, &proof, &mut v_gpu_challenger).unwrap();
    }

    fn create_cpu_pcs(config: &FriConfig) -> CpuPcs {
        let perm = poseidon2_init();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(mmcs.clone());

        let dft = CpuDft::default();
        let fri_config = P3FriConfig {
            log_blowup: config.log_blowup,
            num_queries: config.num_queries,
            proof_of_work_bits: config.proof_of_work_bits,
            mmcs: challenge_mmcs,
        };
        CpuPcs::new(dft, mmcs, fri_config)
    }

    fn p3_commit(
        dft: CpuDft,
        log_blowup: usize,
        evaluations: Vec<(TwoAdicMultiplicativeCoset<F>, RowMajorMatrix<F>)>,
    ) -> (DigestHash, MerkleTree<F, F, RowMajorMatrix<F>, 8>) {
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let shift = F::GENERATOR / domain.shift;
                // Commit to the bit-reversed LDE.
                dft.coset_lde_batch(evals, log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        let perm = poseidon2_init();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = ValMmcs::new(hash, compress);
        mmcs.commit(ldes)
    }
}
