/*
 * Source: https://github.com/openvm-org/stark-backend
 * Status: MODIFIED from stark-backend/crates/cuda-backend/cuda/supra/include/ntt/parameters.cuh
 * Imported: 2025-09-07 by @felicityin
 * 
 * LOCAL CHANGES (high level):
 * - 2025-09-07: Change BabyBear to KoalaBear
 */

/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/parameters/baby_bear.h
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: BABY_BEAR_CANONICAL constants copy from sppark/ntt/parameters/baby_bear.h
 * - 2025-08-13: #if !defined FEATURE_BABY_BEAR -> delete
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "kb31_t.cuh"

#ifndef __SPPARK_NTT_PARAMETERS_CUH__
#define __SPPARK_NTT_PARAMETERS_CUH__

#define WARP_SIZE 32
#define MAX_LG_DOMAIN_SIZE 24
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

// Values in Montgomery form
const fr_t forward_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    0x01fffffeU, 0x7d000003U, 0x7b020407U, 0x60f5ef4dU, 0x6d249c01U, 0x788529f3U,
    0x07f7373eU, 0x6fe91d3cU, 0x3fd49211U, 0x1e056392U, 0x6d969babU, 0x439600ccU,
    0x150276fcU, 0x68cacc36U, 0x42336c40U, 0x019b1972U, 0x34e52f6dU, 0x1c2eb437U,
    0x7cb65829U, 0x29306faeU, 0x351c7fa7U, 0x6e3e9a00U, 0x47c2bdf7U, 0x0c895820U, 0x13c85195U
};

// Values in Montgomery form
const fr_t inverse_roots_of_unity[MAX_LG_DOMAIN_SIZE + 1] = {
    0x01fffffeU, 0x7d000003U, 0x03fdfbfaU, 0x4bfa6163U, 0x52605cfeU, 0x19b8de8dU,
    0x29a9eda0U, 0x7c319486U, 0x6be0a64fU, 0x119f6035U, 0x78c55038U, 0x5c627d99U,
    0x498aeddeU, 0x27052f97U, 0x7bf75488U, 0x2f8a590cU, 0x1dac17b7U, 0x4678e204U,
    0x157bdbf0U, 0x74ca2cd0U, 0x06ee8434U, 0x16c4aa06U, 0x4aee72abU, 0x77640e35U, 0x452f7763U
};

// Values in Montgomery form
const fr_t domain_size_inverse[MAX_LG_DOMAIN_SIZE + 1] = {
    0x01fffffeU, 0x00ffffffU, 0x40000000U, 0x20000000U, 0x10000000U, 0x08000000U,
    0x04000000U, 0x02000000U, 0x01000000U, 0x00800000U, 0x00400000U, 0x00200000U,
    0x00100000U, 0x00080000U, 0x00040000U, 0x00020000U, 0x00010000U, 0x00008000U,
    0x00004000U, 0x00002000U, 0x00001000U, 0x00000800U, 0x00000400U, 0x00000200U, 0x00000100U
};

typedef unsigned int index_t;

#endif /* __SPPARK_NTT_PARAMETERS_CUH__ */
