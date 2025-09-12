/*
 * Poseidon2 KoalaBear
 *
 * Source: https://github.com/risc0/risc0
 * Status: MODIFIED from risc0/sys/kernels/zkp/cuda/supra/poseidon2.cuh
 * Imported: 2025-09-04 by @felicityin
 *
 * LOCAL CHANGES (high level):
 * - 2025-09-03: Change BabyBear to KoalaBear.
*/

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHCELLS_OUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stdio.h"
#include "poseidon2_consts.cuh"
#include "kb31_t.cuh"

#define CELLS 16
#define CELLS_RATE 8
#define CELLS_OUT 8

#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 13
#define ROW_SIZE (CELLS + ROUNDS_PARTIAL)

struct __align__(CELLS_OUT % 4 == 0 ? 16 : 4) poseidon_out_t {
  fr_t data[CELLS_OUT];
};

struct __align__(CELLS_RATE % 4 == 0 ? 16 : 4) poseidon_in_t {
  fr_t data[CELLS_RATE];
};

namespace poseidon2 {

__device__ __forceinline__ void pow3_states_full(fr_t states[CELLS]) {
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    states[i] ^= 3;
  }
}

__device__ __forceinline__ void multiply_by_m_int(fr_t states[CELLS]) {
  fr_t sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    sum += states[i];
  }

#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    states[i] = sum + M_INT_DIAG_HZN[i] * states[i];
  }
}

__device__ __forceinline__ void multiply_by_4x4_circulant(fr_t x[4]) {
  // See appendix B of Poseidon2 paper.
  fr_t t01 = x[0] + x[1];
  fr_t t23 = x[2] + x[3];
  fr_t t0123 = t01 + t23;
  fr_t t01123 = t0123 + x[1];
  fr_t t01233 = t0123 + x[3];
  x[3] = t01233 + fr_t(2) * x[0];
  x[1] = t01123 + fr_t(2) * x[2];
  x[0] = t01123 + t01;
  x[2] = t01233 + t23;
}

__device__ __forceinline__ void multiply_by_m_ext(fr_t states[CELLS]) {
  // Optimized method for multiplication by M_EXT.
  // See appendix B of Poseidon2 paper for additional details.
  fr_t tmp_sums[4] = {0};

  for (uint32_t i = 0; i < CELLS / 4; i++) {
    fr_t* tmp = states + i * 4;
    multiply_by_4x4_circulant(tmp);

    for (uint32_t j = 0; j < 4; j++) {
      tmp_sums[j] += tmp[j];
    }
  }

  for (uint32_t i = 0; i < CELLS; i++) {
    states[i] += tmp_sums[i % 4];
  }
}

__device__ __forceinline__ void full_round(fr_t states[CELLS], uint32_t round_constants_off) {
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    states[i] += ROUND_CONSTANTS[round_constants_off][i];
  }
  pow3_states_full(states);
  multiply_by_m_ext(states);
}

__device__ __forceinline__ void partial_round(fr_t states[CELLS], uint32_t round_constants_off) {
  states[0] += ROUND_CONSTANTS[round_constants_off][0];
  states[0] ^= 3;
  multiply_by_m_int(states);
}

__device__ __forceinline__ void poseidon2_permut(fr_t states[CELLS]) {
  // First linear layer.
  multiply_by_m_ext(states);

  #pragma unroll 1
  for (uint32_t i = 0; i < ROUNDS_HALF_FULL * 2 + ROUNDS_PARTIAL; i++) {
    if (i < ROUNDS_HALF_FULL || i >= ROUNDS_HALF_FULL + ROUNDS_PARTIAL) {
      full_round(states, i);
    } else {
      partial_round(states, i);
    }
  }
}

} // namespace poseidon2
