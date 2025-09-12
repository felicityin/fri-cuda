use cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    base::{DeviceMatrix, MatrixDimensions},
    cuda::kernels::matrix::matrix_transpose,
    types::F,
};

pub fn transport_matrix_to_device(matrix: &RowMajorMatrix<F>) -> DeviceMatrix<F> {
    let data = matrix.values.as_slice();
    let input_buffer = data.to_device().unwrap();
    let output = DeviceMatrix::<F>::with_capacity(matrix.height(), matrix.width());
    unsafe {
        matrix_transpose::<F>(output.buffer(), &input_buffer, matrix.width(), matrix.height())
            .unwrap();
    }
    assert_eq!(output.strong_count(), 1);
    output
}

pub fn transport_device_matrix_to_host<T: Clone + Send + Sync>(
    matrix: &DeviceMatrix<T>,
) -> RowMajorMatrix<T> {
    let matrix_buffer = DeviceBuffer::<T>::with_capacity(matrix.height() * matrix.width());
    unsafe {
        matrix_transpose::<T>(&matrix_buffer, matrix.buffer(), matrix.height(), matrix.width())
            .unwrap();
    }
    RowMajorMatrix::<T>::new(matrix_buffer.to_host().unwrap(), matrix.width())
}

#[cfg(test)]
mod tests {
    use p3_matrix::dense::RowMajorMatrix;
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

    use crate::{
        data_transporter::{transport_device_matrix_to_host, transport_matrix_to_device},
        types::F,
    };

    #[test]
    fn test_transport_matrix_to_device() {
        const COLS: usize = 8;

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, 8, COLS);

        let d_matrix = transport_matrix_to_device(&h_matrix);
        let tmp = transport_device_matrix_to_host(&d_matrix);
        let d2h_matrix = RowMajorMatrix::<F>::new(tmp.values, COLS);

        assert_eq!(h_matrix, d2h_matrix);
    }
}
