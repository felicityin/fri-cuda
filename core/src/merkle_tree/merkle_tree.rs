extern crate alloc;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use hashbrown::HashMap;

use cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};
use itertools::Itertools;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::debug_span;

use crate::{
    base::{DeviceMatrix, MatrixDimensions},
    cuda::kernels::merkle_tree::*,
    dft::GpuLde,
    types::{Digest, DigestHash, F},
};

#[derive(Clone)]
pub struct GpuMerkleTree {
    pub leaves: Vec<GpuLde>,
    pub digest_layers: Vec<Arc<DeviceBuffer<Digest>>>,
}

impl GpuMerkleTree {
    pub fn new(leaves: Vec<GpuLde>) -> Result<Self, CudaError> {
        assert!(!leaves.is_empty(), "No matrices given?");

        let mut matrices_largest_first =
            leaves.iter().sorted_by_key(|m| Reverse(m.height())).peekable();

        let max_height = matrices_largest_first.peek().unwrap().height();

        let digests = Arc::new(DeviceBuffer::<Digest>::with_capacity(max_height));
        {
            let tallest_matrices = matrices_largest_first
                .peeking_take_while(|m| m.height() == max_height)
                .map(|m| m.get_lde(m.height()))
                .collect_vec();

            Self::hash_matrices(&digests, &tallest_matrices).unwrap();
        }

        let mut digest_layers: Vec<Arc<DeviceBuffer<Digest>>> = vec![digests];

        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;
            let next_layer = Arc::new(DeviceBuffer::<Digest>::with_capacity(next_layer_len));
            let is_inject = {
                let matrices_to_inject = matrices_largest_first
                    .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                    .map(|m| m.get_lde(m.height()))
                    .collect_vec();

                let has_matrices = !matrices_to_inject.is_empty();
                if has_matrices {
                    Self::hash_matrices(&next_layer, &matrices_to_inject).unwrap();
                }
                has_matrices
            };

            Self::hash_compress(&next_layer, prev_layer, is_inject).unwrap();

            digest_layers.push(next_layer);
        }

        Ok(Self { leaves, digest_layers })
    }

    pub fn root(&self) -> DigestHash {
        let root = self.digest_layers.last().unwrap();
        assert_eq!(root.len(), 1, "Only one root is supported");
        root.to_host().unwrap()[0].into()
    }

    fn hash_matrices(
        out: &DeviceBuffer<Digest>,
        matrices: &[DeviceMatrix<F>],
    ) -> Result<(), CudaError> {
        // For poseidon2_rows_p3_multi we need:
        // matrices_ptr - array of pointers to matrices
        // matrices_col - array of column sizes
        // matrices_row - array of row sizes
        let matrices_ptr: Vec<u64> = matrices.iter().map(|m| m.buffer().as_ptr() as u64).collect();
        let matrices_col: Vec<u64> = matrices.iter().map(|m| m.width() as u64).collect();
        let matrices_row: Vec<u64> = matrices.iter().map(|m| m.height() as u64).collect();

        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();

        unsafe {
            poseidon2_rows_p3_multi(
                out,
                &d_matrices_ptr,
                &d_matrices_col,
                &d_matrices_row,
                matrices_row[0],
                matrices.len() as u64,
            )
        }
    }

    fn hash_compress(
        out: &DeviceBuffer<Digest>,
        prev_layer: &DeviceBuffer<Digest>,
        is_inject: bool,
    ) -> Result<(), CudaError> {
        unsafe { poseidon2_compress(out, prev_layer, out.len() as u32, is_inject) }
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::result_unit_err)]
    pub fn open_batch_at_multiple_indices(
        &self,
        indices: &[usize],
    ) -> Result<Vec<(Vec<Vec<F>>, Vec<Digest>)>, ()> {
        let max_height = self.leaves.iter().map(|m| m.height()).max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        // Open all indices of one leaf at once to reduce gpu peak memory
        // the structure of openings: [leaf_index][point_index][column_index]
        let mut openings_indexed_by_leaf_idx = self
            .leaves
            .iter()
            .map(|matrix| {
                let openings_per_matrix = debug_span!("read rows").in_scope(|| {
                    let log_matrix_height = log2_ceil_usize(matrix.height());
                    let bits_reduced = log_max_height - log_matrix_height;

                    let mut unique_map = HashMap::new();
                    let mut unique_indices = Vec::new();

                    let reduced_indices = indices
                        .iter()
                        .map(|&index| {
                            let reduced_idx = index >> bits_reduced;
                            if !unique_map.contains_key(&reduced_idx) {
                                unique_map.insert(reduced_idx, unique_map.len());
                                unique_indices.push(reduced_idx);
                            }
                            reduced_idx
                        })
                        .collect_vec();

                    let unique_openings_vec =
                        matrix.get_lde_rows(&unique_indices).to_host().unwrap();
                    let unique_openings_rows =
                        unique_openings_vec.chunks(matrix.width()).collect_vec();

                    reduced_indices
                        .iter()
                        .map(|idx| Ok(unique_openings_rows[unique_map[idx]].to_vec()))
                        .collect::<Result<Vec<_>, ()>>()
                })?;

                Ok(openings_per_matrix)
            })
            .collect::<Result<Vec<_>, ()>>()?;

        // Convert openings' structure to [point_index][leaf_index][column_index]
        let openings_indexed_by_point_idx = indices
            .iter()
            .rev()
            .map(|_| {
                // For each point (in reverse)
                openings_indexed_by_leaf_idx
                    .iter_mut()
                    .map(|openings_per_matrix| openings_per_matrix.pop().unwrap())
                    .collect_vec()
            })
            .collect_vec()
            .into_iter()
            .rev()
            .collect_vec();

        let proofs: Vec<Vec<Digest>> = debug_span!("read digests").in_scope(|| {
            let query_indices = indices
                .iter()
                .flat_map(|index| {
                    (0..log_max_height)
                        .map(|i| {
                            ((index >> i) ^ 1) as u64 //sibling_index
                        })
                        .collect::<Vec<u64>>()
                })
                .collect::<Vec<_>>();

            let num_query = indices.len();
            let num_layer = log_max_height;

            let all_digests = Self::query_digest_layers(
                &self.digest_layers[..log_max_height],
                &query_indices,
                num_query,
                num_layer,
            );
            assert_eq!(num_layer + 1, self.digest_layers.len());
            assert_eq!(all_digests.len(), query_indices.len());

            all_digests
                .chunks(num_layer)
                .map(|layers_digest| Ok(layers_digest.to_vec()))
                .collect::<Result<Vec<_>, ()>>()
        })?;

        Ok(openings_indexed_by_point_idx.into_iter().zip(proofs).collect::<Vec<(_, _)>>())
    }

    fn query_digest_layers(
        digest_layers: &[Arc<DeviceBuffer<Digest>>],
        indices: &[u64],
        num_query: usize,
        num_layer: usize,
    ) -> Vec<Digest> {
        assert_eq!(num_layer, digest_layers.len());
        assert_eq!(num_layer * num_query, indices.len());

        let digest_layers_ptr =
            digest_layers.iter().map(|layer| layer.as_ptr() as u64).collect_vec();
        let digest_layers_ptr_buf = digest_layers_ptr.to_device().unwrap();

        let d_indices = indices.to_device().unwrap();
        let digest_buffer = DeviceBuffer::<Digest>::with_capacity(indices.len());

        unsafe {
            query_digest_layers_kernel(
                &digest_buffer,
                &digest_layers_ptr_buf,
                &d_indices,
                num_query.try_into().unwrap(),
                num_layer.try_into().unwrap(),
            )
            .unwrap();
        }
        digest_buffer.to_host().unwrap()
    }

    pub fn get_max_height(&self) -> usize {
        self.leaves.iter().map(|m| m.height()).max().unwrap()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::array;

    use cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer};
    use p3_field::PackedValue;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use p3_maybe_rayon::prelude::ParallelSliceMut;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
    use zkm_primitives::{POSEIDON2_HASHER, poseidon2_hash, poseidon2_init};

    use crate::{
        data_transporter::transport_matrix_to_device,
        merkle_tree::GpuMerkleTree,
        types::{Digest, F, MyCompress},
    };

    #[test]
    fn test_poseidon2() {
        const COLS: usize = 64;
        let rows = 1;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, COLS);
        let d_matrix = transport_matrix_to_device(&h_matrix);

        let digests = DeviceBuffer::<Digest>::with_capacity(1);
        GpuMerkleTree::hash_matrices(&digests, &[d_matrix]).unwrap();
        let gpu_hash = digests.to_host().unwrap();

        let cpu_hash = poseidon2_hash(h_matrix.values);

        assert_eq!(gpu_hash[0], cpu_hash);
    }

    #[test]
    fn test_hash_matrices() {
        const COLS: usize = 32;
        const ROWS: usize = 4;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix_1 = RowMajorMatrix::<F>::rand_nonzero(&mut rng, ROWS, COLS);
        let h_matrix_2 = RowMajorMatrix::<F>::rand_nonzero(&mut rng, ROWS, COLS);
        let d_matrix_1 = transport_matrix_to_device(&h_matrix_1);
        let d_matrix_2 = transport_matrix_to_device(&h_matrix_2);

        let digests = DeviceBuffer::<Digest>::with_capacity(ROWS);
        GpuMerkleTree::hash_matrices(&digests, &[d_matrix_1, d_matrix_2]).unwrap();
        let mut gpu_hashs = digests.to_host().unwrap();

        let mut cpu_hashs = p3_hash_matrices(vec![h_matrix_1, h_matrix_2]);

        assert_eq!(gpu_hashs.sort(), cpu_hashs.sort());
    }

    #[test]
    fn test_hash_compress() {
        const COLS: usize = 32;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, 2, COLS);
        let d_matrix = transport_matrix_to_device(&h_matrix);

        let pre_layer = DeviceBuffer::<Digest>::with_capacity(2);
        GpuMerkleTree::hash_matrices(&pre_layer, &[d_matrix]).unwrap();
        let next_layer = Arc::new(DeviceBuffer::<Digest>::with_capacity(1));
        GpuMerkleTree::hash_compress(&next_layer, &pre_layer, false).unwrap();
        let gpu_next_layer = next_layer.to_host().unwrap();

        let pre_layer = p3_hash_matrices(vec![h_matrix]);
        let cpu_next_layer = p3_compress(&pre_layer);

        assert_eq!(gpu_next_layer, cpu_next_layer);
    }

    fn p3_hash_matrices(matrices: Vec<RowMajorMatrix<F>>) -> Vec<Digest> {
        let width = 8;
        let max_height = matrices[0].height();
        // we always want to return an even number of digests, except when it's the root.
        let max_height_padded = if max_height == 1 { 1 } else { max_height + max_height % 2 };

        let default_digest = Digest::default();
        let mut digests = vec![default_digest; max_height_padded];

        digests[0..max_height].par_chunks_exact_mut(width).enumerate().for_each(
            |(i, digests_chunk)| {
                let first_row = i * width;
                let packed_digest = POSEIDON2_HASHER
                    .hash_iter(matrices.iter().flat_map(|m| m.vertically_packed_row(first_row)));
                for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                    *dst = src;
                }
            },
        );

        // If our packing width did not divide max_height, fall back to single-threaded scalar code
        // for the last bit.
        #[allow(clippy::needless_range_loop)]
        for i in (max_height / width * width)..max_height {
            digests[i] = POSEIDON2_HASHER.hash_iter(matrices.iter().flat_map(|m| m.row(i)));
        }

        // Everything has been initialized so we can safely cast.
        digests
    }

    fn p3_compress(prev_layer: &[Digest]) -> Vec<Digest> {
        let perm = poseidon2_init();
        let c = MyCompress::new(perm);

        let width = 8;
        // Always return an even number of digests, except when it's the root.
        let next_len_padded =
            if prev_layer.len() == 2 { 1 } else { (prev_layer.len() / 2 + 1) & !1 };
        let next_len = prev_layer.len() / 2;

        let default_digest = Digest::default();
        let mut next_digests = vec![default_digest; next_len_padded];

        next_digests[0..next_len].par_chunks_exact_mut(width).enumerate().for_each(
            |(i, digests_chunk)| {
                let first_row = i * width;
                let left = array::from_fn(|j| F::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
                let right =
                    array::from_fn(|j| F::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
                let packed_digest = c.compress([left, right]);
                for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                    *dst = src;
                }
            },
        );

        // If our packing width did not divide next_len, fall back to single-threaded scalar code
        // for the last bit.
        for i in (next_len / width * width)..next_len {
            let left = prev_layer[2 * i];
            let right = prev_layer[2 * i + 1];
            next_digests[i] = c.compress([left, right]);
        }

        // Everything has been initialized so we can safely cast.
        next_digests
    }

    #[inline]
    fn unpack_array(packed_digest: Digest) -> impl Iterator<Item = Digest> {
        (0..8).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
    }
}
