extern crate alloc;
use alloc::vec::Vec;

use crate::{dft::GpuLde, merkle_tree::GpuMerkleTree, types::DigestHash};

pub struct GpuMerkleTreeMmcs;

impl GpuMerkleTreeMmcs {
    pub fn commit(leaves: Vec<GpuLde>) -> (DigestHash, GpuMerkleTree) {
        let tree = GpuMerkleTree::new(leaves).expect("Failed to create a merkle tree.");
        let root = tree.root();
        (root, tree)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use p3_commit::Mmcs;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::Field;
    use p3_matrix::{Matrix, bitrev::BitReversableMatrix, dense::RowMajorMatrix};
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
    use zkm_primitives::poseidon2_init;

    use crate::{
        data_transporter::transport_matrix_to_device,
        dft::GpuLde,
        merkle_tree::GpuMerkleTreeMmcs,
        types::{F, MyCompress, MyHash, ValMmcs},
    };

    #[test]
    fn test_commit_matrices() {
        const LOG_BLOWUP: usize = 1;
        const COLS_1: usize = 16;
        const COLS_2: usize = 8;
        const ROWS_1: usize = 2;
        const ROWS_2: usize = 4;
        let shift = F::GENERATOR;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix_1 = RowMajorMatrix::<F>::rand_nonzero(&mut rng, ROWS_1, COLS_1);
        let h_matrix_2 = RowMajorMatrix::<F>::rand_nonzero(&mut rng, ROWS_2, COLS_2);
        let d_matrix_1 = transport_matrix_to_device(&h_matrix_1);
        let d_matrix_2 = transport_matrix_to_device(&h_matrix_2);

        let gpu_root = {
            let lde_1 = GpuLde::new(d_matrix_1, LOG_BLOWUP, shift);
            let lde_2 = GpuLde::new(d_matrix_2, LOG_BLOWUP, shift);
            let (gpu_root, _) = GpuMerkleTreeMmcs::commit(vec![lde_1, lde_2]);
            gpu_root
        };

        let cpu_root = {
            let perm = poseidon2_init();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm);
            let mmcs = ValMmcs::new(hash, compress);

            let dft = Radix2DitParallel::<F>::default();
            let lde_1 = dft
                .coset_lde_batch(h_matrix_1, LOG_BLOWUP, shift)
                .bit_reverse_rows()
                .to_row_major_matrix();
            let lde_2 = dft
                .coset_lde_batch(h_matrix_2, LOG_BLOWUP, shift)
                .bit_reverse_rows()
                .to_row_major_matrix();

            let (cpu_root, _) = mmcs.commit(vec![lde_1, lde_2]);
            cpu_root
        };

        assert_eq!(gpu_root, cpu_root);
    }
}
