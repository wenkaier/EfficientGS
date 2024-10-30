import torch
import os
import os.path as osp
from torch.utils import cpp_extension
from torch import IntTensor, FloatTensor


def build_cuda_module():
    curr_path = os.path.abspath(os.path.dirname(__file__))

    build_directory = osp.join(curr_path, "build")
    os.makedirs(build_directory, exist_ok=True)

    cuda_module = cpp_extension.load(
        name="cuda_ssim",  # same name with *.cpp
        sources=[
            osp.join(curr_path, "cuda_ssim.cpp"),
            osp.join(curr_path, "cuda_ssim.cu"),
        ],
        build_directory=build_directory,
        verbose=True,
    )
    return cuda_module


_C = build_cuda_module()

def cuda_ssim(img1: FloatTensor, img2: FloatTensor, window_size: int = 11):
    