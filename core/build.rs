use std::process::exit;

use cuda_builder::{CudaBuilder, cuda_available};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    let builder = CudaBuilder::new();

    builder.emit_link_directives();

    builder
        .clone()
        .library_name("fri_cuda")
        .include("kernels/include")
        .files_from_glob("kernels/src/*.cu")
        .build();

    builder
        .clone()
        .library_name("supra_ntt")
        .include("kernels/include")
        .include("kernels/supra/include")
        .files_from_glob("kernels/supra/*.cu")
        .build();
}
