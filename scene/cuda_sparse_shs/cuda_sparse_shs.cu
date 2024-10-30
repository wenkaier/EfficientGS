#include "cuda_runtime.h"
#include "cuda_sparse_shs.h"
#include <iostream>

__global__ void copy_shs_cuda(const int P, const int *old_idx,
                              const float *old_shs, const int *old_shs_degree,
                              const int *old_shs_ptr, float *new_shs,
                              const int *new_shs_degree,
                              const int *new_shs_ptr) {
  int new_id = blockIdx.x * NUM_THREAD + threadIdx.x;
  if (new_id >= P)
    return;
  int old_id = old_idx[new_id];
  int degree = old_shs_degree[old_id];
  for (int i = 0; i < degree * 3; i++) {
    new_shs[new_shs_ptr[new_id] + i] = old_shs[old_shs_ptr[old_id] + i];
  }
}

void copy_shs(const int P, const int *old_idx, const float *old_shs,
              const int *old_shs_degree, const int *old_shs_ptr, float *new_shs,
              const int *new_shs_degree, const int *new_shs_ptr) {
  copy_shs_cuda<<<(P + NUM_THREAD - 1) / NUM_THREAD, NUM_THREAD>>>(
      P, old_idx, old_shs, old_shs_degree, old_shs_ptr, new_shs, new_shs_degree,
      new_shs_ptr);
}
