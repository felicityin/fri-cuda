use crate::{
    base::{DeviceMatrix, MatrixDimensions},
    types::F,
};

pub mod ops;
use ops::*;
mod ntt;

#[derive(Clone)]
pub struct GpuLde {
    pub lde: DeviceMatrix<F>,
    log_blowup: usize,
    shift: F,
}

impl MatrixDimensions for GpuLde {
    fn height(&self) -> usize {
        self.lde.height()
    }

    fn width(&self) -> usize {
        self.lde.width()
    }
}

impl GpuLde {
    pub fn new(matrix: DeviceMatrix<F>, log_blowup: usize, shift: F) -> Self {
        if log_blowup == 0 {
            return Self { lde: matrix, log_blowup, shift };
        }
        let trace_height = matrix.height();
        let lde_height = trace_height << log_blowup;
        let lde = compute_lde_matrix(&matrix, lde_height, shift);
        Self { lde, log_blowup, shift }
    }

    pub fn shift(&self) -> F {
        self.shift
    }

    pub fn trace_height(&self) -> usize {
        self.lde.height() >> self.log_blowup
    }

    pub fn get_lde(&self, domain_size: usize) -> DeviceMatrix<F> {
        assert!(self.height() >= domain_size);
        self.lde.clone()
    }

    pub fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F> {
        assert!(!row_indices.is_empty());
        get_rows_from_matrix(&self.lde, row_indices)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use p3_dft::{Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::Field;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::{Matrix, bitrev::BitReversableMatrix, dense::RowMajorMatrix};
    use rand::{Rng, thread_rng};
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

    use crate::{
        data_transporter::{transport_device_matrix_to_host, transport_matrix_to_device},
        dft::GpuLde,
    };

    #[test]
    fn test_lde() {
        type F = KoalaBear;
        const LOG_BLOWUP: usize = 1;
        let shift = F::GENERATOR;

        const COLS: usize = 4;
        const ROWS: usize = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, ROWS, COLS);

        let dft = Radix2Dit::<F>::default();
        let mut cpu_lde = dft
            .coset_lde_batch(h_matrix.clone(), LOG_BLOWUP, shift)
            .bit_reverse_rows()
            .to_row_major_matrix();

        let d_matrix = transport_matrix_to_device(&h_matrix);
        let lde = GpuLde::new(d_matrix, LOG_BLOWUP, shift);
        let h_matrix = transport_device_matrix_to_host(&lde.lde);
        let mut gpu_lde = RowMajorMatrix::<F>::new(h_matrix.values, COLS);

        cpu_lde.values.sort();
        gpu_lde.values.sort();
        assert_eq!(cpu_lde, gpu_lde);
    }

    #[test]
    fn test_batch_coset_lde() {
        let mut rng = thread_rng();
        let log_degrees = 16..18;
        let log_blowup = 1;
        let batch_size = 100;

        let p3_dft = Radix2DitParallel::<KoalaBear>::default();

        for log_d in log_degrees {
            let d = 1 << log_d;
            let width = batch_size;
            let shift = rng.r#gen::<KoalaBear>();

            let mat_h = RowMajorMatrix::rand(&mut rng, d, batch_size);

            let time = Instant::now();
            let mat_d = transport_matrix_to_device(&mat_h);
            let lde = GpuLde::new(mat_d, log_blowup, shift);
            let gpu_time = time.elapsed();
            println!("Gpu dft time log degree {log_d}: {gpu_time:?}");

            let time = Instant::now();
            let mut cpu_lde = p3_dft
                .coset_lde_batch(mat_h, log_blowup, shift)
                .bit_reverse_rows()
                .to_row_major_matrix();
            let cpu_time = time.elapsed();
            println!("Cpu dft time log degree {log_d}: {cpu_time:?}");

            let h_matrix = transport_device_matrix_to_host(&lde.lde);
            let mut gpu_lde = RowMajorMatrix::<KoalaBear>::new(h_matrix.values, width);
            cpu_lde.values.sort();
            gpu_lde.values.sort();
            assert_eq!(cpu_lde, gpu_lde);
        }
    }
}
