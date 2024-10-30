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
#include "cuda_ssim.h"


void ssim_forward_wrapper(
    torch::Tensor& img1,
    torch::Tensor& img2,
    const int window_size
){
    int C = img1.size(0);
    int H = img1.size(1);
    int W = img1.size(2);
    ssim_forward(
        C,H, W,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>()
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ssim_forward", &ssim_forward_wrapper, "ssim_forward kernel warpper");
}
