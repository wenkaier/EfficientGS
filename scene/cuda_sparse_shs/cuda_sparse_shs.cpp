#include <cassert>
#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "cuda_sparse_shs.h"


void copy_shs_wrapper(
    torch::Tensor& old_idx,
    torch::Tensor& old_shs,
    torch::Tensor& old_shs_degree,
    torch::Tensor& old_shs_ptr,
    torch::Tensor& new_shs,
    torch::Tensor& new_shs_degree,
    torch::Tensor& new_shs_ptr)
{
    int P = new_shs.size(0);
    copy_shs(
        P,
        old_idx.contiguous().data_ptr<int>(),
        old_shs.contiguous().data_ptr<float>(),
        old_shs_degree.contiguous().data_ptr<int>(),
        old_shs_ptr.contiguous().data_ptr<int>(),
        new_shs.contiguous().data_ptr<float>(),
        new_shs_degree.contiguous().data_ptr<int>(),
        new_shs_ptr.contiguous().data_ptr<int>()
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_shs", &copy_shs_wrapper, "copy_shs kernel warpper");
}
