#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, KernelError},
};

pub const FP_SIZE: usize = 4; // sizeof(uint32_t)
pub const FP_EXT_SIZE: usize = 16; // 4 * sizeof(uint32_t)

// Relate to matrix.cu
pub mod matrix {
    use super::*;
    unsafe extern "C" {
        fn matrix_transpose_fp(
            output: *mut core::ffi::c_void,
            input: *const core::ffi::c_void,
            col_size: usize,
            row_size: usize,
        ) -> i32;

        fn matrix_transpose_fpext(
            output: *mut core::ffi::c_void,
            input: *const core::ffi::c_void,
            col_size: usize,
            row_size: usize,
        ) -> i32;

        fn _matrix_get_rows_fp(
            output: *mut core::ffi::c_void,
            input: *const core::ffi::c_void,
            row_indices: *const u32,
            matrix_width: u64,
            matrix_height: u64,
            row_indices_len: u32,
        ) -> i32;
    }

    pub unsafe fn matrix_transpose<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        width: usize,
        height: usize,
    ) -> Result<(), KernelError> {
        let size = core::mem::size_of::<T>();
        let result = match size {
            FP_SIZE => unsafe {
                matrix_transpose_fp(output.as_mut_raw_ptr(), input.as_raw_ptr(), width, height)
            },
            FP_EXT_SIZE => unsafe {
                matrix_transpose_fpext(output.as_mut_raw_ptr(), input.as_raw_ptr(), width, height)
            },
            _ => return Err(KernelError::UnsupportedTypeSize { size }),
        };

        CudaError::from_result(result).map_err(KernelError::from)
    }

    pub unsafe fn matrix_get_rows_fp_kernel<F>(
        output: &DeviceBuffer<F>,
        input: &DeviceBuffer<F>,
        row_indices: &DeviceBuffer<u32>,
        matrix_width: u64,
        matrix_height: u64,
        row_indices_len: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _matrix_get_rows_fp(
                output.as_mut_raw_ptr(),
                input.as_raw_ptr(),
                row_indices.as_ptr(),
                matrix_width,
                matrix_height,
                row_indices_len,
            )
        })
    }
}

// Relate to dft.cu
pub mod dft {
    use super::*;

    unsafe extern "C" {
        fn _multi_bit_reverse(io: *mut core::ffi::c_void, n_bits: u32, count: u32) -> i32;

        fn _zk_shift(
            io: *mut core::ffi::c_void,
            io_size: u32,
            log_n: u32,
            shift: crate::types::F,
        ) -> i32;

        fn _batch_expand_pad(
            output: *mut core::ffi::c_void,
            input: *const core::ffi::c_void,
            poly_count: u32,
            out_size: u32,
            in_size: u32,
        ) -> i32;
    }

    pub unsafe fn batch_bit_reverse<T>(
        io: &DeviceBuffer<T>,
        n_bits: u32,
        count: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe { _multi_bit_reverse(io.as_mut_raw_ptr(), n_bits, count) })
    }

    pub unsafe fn zk_shift<T>(
        io: &DeviceBuffer<T>,
        io_size: u32,
        log_n: u32,
        shift: crate::types::F,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe { _zk_shift(io.as_mut_raw_ptr(), io_size, log_n, shift) })
    }

    pub unsafe fn batch_expand_pad<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        poly_count: u32,
        out_size: u32,
        in_size: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _batch_expand_pad(
                output.as_mut_raw_ptr(),
                input.as_raw_ptr(),
                poly_count,
                out_size,
                in_size,
            )
        })
    }
}

// Relate to merkle_tree.cu
pub mod merkle_tree {
    use super::*;

    unsafe extern "C" {
        fn _poseidon2_rows_p3_multi(
            out: *mut core::ffi::c_void,
            ptrs: *const u64,
            cols: *const u64,
            rows: *const u64,
            row_size: u64,
            matrix_num: u64,
        ) -> i32;

        fn _poseidon2_compress(
            output: *mut core::ffi::c_void,
            input: *const core::ffi::c_void,
            output_size: u32,
            is_inject: bool,
        ) -> i32;

        fn _query_digest_layers(
            d_digest_matrix: *mut core::ffi::c_void,
            d_layers_ptr: *const u64,
            d_indices: *const u64,
            num_query: u64,
            num_layer: u64,
        ) -> i32;
    }

    pub unsafe fn poseidon2_rows_p3_multi<T>(
        out: &DeviceBuffer<T>,
        ptrs: &DeviceBuffer<u64>,
        cols: &DeviceBuffer<u64>,
        rows: &DeviceBuffer<u64>,
        row_size: u64,
        matrix_num: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _poseidon2_rows_p3_multi(
                out.as_mut_raw_ptr(),
                ptrs.as_ptr(),
                cols.as_ptr(),
                rows.as_ptr(),
                row_size,
                matrix_num,
            )
        })
    }

    pub unsafe fn poseidon2_compress<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        output_size: u32,
        is_inject: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _poseidon2_compress(output.as_mut_raw_ptr(), input.as_raw_ptr(), output_size, is_inject)
        })
    }

    pub unsafe fn query_digest_layers_kernel<T>(
        d_digest_matrix: &DeviceBuffer<T>,
        d_layers_ptr: &DeviceBuffer<u64>,
        d_indices: &DeviceBuffer<u64>,
        num_query: u64,
        num_layer: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _query_digest_layers(
                d_digest_matrix.as_mut_raw_ptr(),
                d_layers_ptr.as_ptr(),
                d_indices.as_ptr(),
                num_query,
                num_layer,
            )
        })
    }
}

// Relate to fri.cu
pub mod fri {
    use super::*;

    unsafe extern "C" {
        fn _compute_diffs(
            d_diffs: *mut core::ffi::c_void,
            d_z: *mut core::ffi::c_void,
            d_domain: *mut core::ffi::c_void,
            log_max_height: u32,
        ) -> i32;

        fn _fpext_bit_reverse(d_diffs: *mut core::ffi::c_void, log_max_height: u32) -> i32;

        fn _batch_invert(
            d_diffs: *mut core::ffi::c_void,
            log_max_height: u32,
            invert_task_num: u32,
        ) -> i32;

        fn _powers(d_data: *mut core::ffi::c_void, d_g: *const core::ffi::c_void, N: u32) -> i32;
        fn _powers_ext(
            d_data: *mut core::ffi::c_void,
            d_g: *const core::ffi::c_void,
            N: u32,
        ) -> i32;

        fn _reduce_matrix_quotient_acc(
            d_quotient_acc: *mut core::ffi::c_void,
            d_matrix: *const core::ffi::c_void,
            d_z_diff_invs: *const core::ffi::c_void,
            d_matrix_eval: *const core::ffi::c_void,
            d_alphas: *const core::ffi::c_void,
            d_alphas_offset: *const core::ffi::c_void,
            width: u32,
            height: u32,
            is_first: bool,
        ) -> i32;

        fn _cukernel_split_ext_poly_to_base_col_major_matrix(
            d_matrix: *mut core::ffi::c_void,
            d_poly: *const core::ffi::c_void,
            poly_len: u64,
            matrix_height: u32,
        ) -> i32;

        fn _cukernel_fri_fold(
            d_result: *mut core::ffi::c_void,
            d_poly: *const core::ffi::c_void,
            fri_input: *const core::ffi::c_void,
            d_constants: *const core::ffi::c_void,
            g_invs: *const core::ffi::c_void,
            N: u64,
        ) -> i32;

        fn _matrix_evaluate_chunked(
            partial_sums: *mut core::ffi::c_void,
            matrix: *const core::ffi::c_void,
            inv_denoms: *const core::ffi::c_void,
            g: crate::types::F,
            height: u32,
            width: u32,
            chunk_size: u32,
            num_chunks: u32,
            matrix_height: u32,
            inv_denoms_bitrev: bool,
        ) -> i32;

        fn _matrix_evaluate_finalize(
            output: *mut core::ffi::c_void,
            partial_sums: *const core::ffi::c_void,
            scale_factor: crate::types::EF,
            num_chunks: u32,
            width: u32,
        ) -> i32;
    }

    pub unsafe fn diffs_kernel<F, EF>(
        d_diffs: &DeviceBuffer<EF>,
        d_z: &DeviceBuffer<EF>,
        d_domain: &DeviceBuffer<F>,
        log_max_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _compute_diffs(
                d_diffs.as_mut_raw_ptr(),
                d_z.as_mut_raw_ptr(),
                d_domain.as_mut_raw_ptr(),
                log_max_height,
            )
        })
    }

    pub unsafe fn fpext_bit_rev_kernel<EF>(
        d_diffs: &DeviceBuffer<EF>,
        log_max_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _fpext_bit_reverse(d_diffs.as_mut_raw_ptr(), log_max_height)
        })
    }

    pub unsafe fn batch_invert_kernel<EF>(
        d_diffs: &DeviceBuffer<EF>,
        log_max_height: u32,
        invert_task_num: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _batch_invert(d_diffs.as_mut_raw_ptr(), log_max_height, invert_task_num)
        })
    }

    pub unsafe fn powers<F>(
        d_data: &DeviceBuffer<F>,
        d_g: &DeviceBuffer<F>,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe { _powers(d_data.as_mut_raw_ptr(), d_g.as_raw_ptr(), n) })
    }

    pub unsafe fn powers_ext<EF>(
        d_data: &DeviceBuffer<EF>,
        d_g: &DeviceBuffer<EF>,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe { _powers_ext(d_data.as_mut_raw_ptr(), d_g.as_raw_ptr(), n) })
    }

    pub unsafe fn reduce_matrix_quotient_kernel<F, EF>(
        d_quotient_acc: &DeviceBuffer<EF>,
        d_matrix: &DeviceBuffer<F>,
        d_z_diff_invs: &DeviceBuffer<EF>,
        d_matrix_eval: &DeviceBuffer<EF>,
        d_alphas: &DeviceBuffer<EF>,
        d_alphas_offset: &DeviceBuffer<EF>,
        width: u32,
        height: u32,
        is_first: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _reduce_matrix_quotient_acc(
                d_quotient_acc.as_mut_raw_ptr(),
                d_matrix.as_raw_ptr(),
                d_z_diff_invs.as_raw_ptr(),
                d_matrix_eval.as_raw_ptr(),
                d_alphas.as_raw_ptr(),
                d_alphas_offset.as_raw_ptr(),
                width,
                height,
                is_first,
            )
        })
    }

    pub unsafe fn split_ext_poly_to_base_col_major_matrix<F, EF>(
        d_matrix: &DeviceBuffer<F>,
        d_poly: &DeviceBuffer<EF>,
        poly_len: u64,
        matrix_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _cukernel_split_ext_poly_to_base_col_major_matrix(
                d_matrix.as_mut_raw_ptr(),
                d_poly.as_raw_ptr(),
                poly_len,
                matrix_height,
            )
        })
    }

    pub unsafe fn fri_fold_kernel<F, EF>(
        d_result: &DeviceBuffer<EF>,
        d_poly: &DeviceBuffer<EF>,
        fri_input: &DeviceBuffer<EF>,
        d_constants: &DeviceBuffer<EF>,
        g_invs: &DeviceBuffer<F>,
        half_folded_len: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _cukernel_fri_fold(
                d_result.as_mut_raw_ptr(),
                d_poly.as_raw_ptr(),
                fri_input.as_raw_ptr(),
                d_constants.as_raw_ptr(),
                g_invs.as_raw_ptr(),
                half_folded_len,
            )
        })
    }

    pub unsafe fn matrix_evaluate_chunked_kernel<F, EF>(
        partial_sums: &DeviceBuffer<EF>,
        matrix: &DeviceBuffer<F>,
        inv_denoms: &DeviceBuffer<EF>,
        g: crate::types::F,
        height: u32,
        width: u32,
        chunk_size: u32,
        num_chunks: u32,
        matrix_height: u32,
        inv_denoms_bitrev: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _matrix_evaluate_chunked(
                partial_sums.as_mut_raw_ptr(),
                matrix.as_raw_ptr(),
                inv_denoms.as_raw_ptr(),
                g,
                height,
                width,
                chunk_size,
                num_chunks,
                matrix_height,
                inv_denoms_bitrev,
            )
        })
    }

    pub unsafe fn matrix_evaluate_finalize_kernel<EF>(
        output: &DeviceBuffer<EF>,
        partial_sums: &DeviceBuffer<EF>,
        scale_factor: crate::types::EF,
        num_chunks: u32,
        width: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(unsafe {
            _matrix_evaluate_finalize(
                output.as_mut_raw_ptr(),
                partial_sums.as_raw_ptr(),
                scale_factor,
                num_chunks,
                width,
            )
        })
    }
}
