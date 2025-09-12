# FRI CUDA

This project implements the FRI low-degree test (LDT) on GPU via CUDA, following the algorithmic framework established in [Plonky3](https://github.com/Plonky3/plonky3).

Currently, only KoalaBear is supported.

## Usage

Reference: src/fri/test.rs

```rust
const LOG_BLOWUP: usize = 1;
const COLS: usize = 8;
let log_degree = 12;
let rows = 1 << log_degree

let mut rng = ChaCha20Rng::seed_from_u64(0);
let h_matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, COLS);
let d_matrix = transport_matrix_to_device(&h_matrix);

let config = FriConfig { log_blowup: LOG_BLOWUP, num_queries: 10, proof_of_work_bits: 8 };
let pcs = GpuTwoAdicFriPcs::new(config);

// ------------------------ commit ------------------------

let domain = pcs.natural_domain_for_degree(h_matrix.height());
let (root, data) = pcs.commit(vec![(domain, d_matrix)]);

// ------------------------ open --------------------------

let mut challenger = Challenger::new(poseidon2_init());
challenger.observe(root);
let zeta: EF = challenger.sample_ext_element();

let (open, proof) =
    pcs.open(vec![(&data, vec![vec![zeta]])], &mut challenger);
```

## Acknowledgements

We studied and built upon the work of other teams in our quest to implement this repository.
We would like to thank these teams for sharing their code for open source development:

- [OpenVM Stark Backend](https://github.com/openvm-org/stark-backend): We used [OpenVM]((https://github.com/openvm-org/stark-backend))'s open source CUDA kernels, builder and manager as the starting point for our CUDA backend.
- [Risc0](https://github.com/risc0/risc0): We adapted [Risc0](https://github.com/risc0/risc0)'s open-source BabyBear Poseidon2 CUDA kernels for our own CUDA kernels.
- [Supranational](https://github.com/supranational/sppark): We ported and modified [sppark](https://github.com/supranational/sppark)'s open source BabyBear field arithmetic and BabyBear NTT CUDA kernels for use in our CUDA backend.
- [Scroll](https://github.com/scroll-tech/): Members of the Scroll team made foundational contributions to the the CUDA backend.
