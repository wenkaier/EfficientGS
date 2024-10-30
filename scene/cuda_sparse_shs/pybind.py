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
        name="cuda_sparse_shs",  # same name with *.cpp
        sources=[
            osp.join(curr_path, "cuda_sparse_shs.cpp"),
            osp.join(curr_path, "cuda_sparse_shs.cu"),
        ],
        build_directory=build_directory,
        verbose=True,
    )
    return cuda_module


_C = build_cuda_module()


class SparseSHs:
    def __init__(self) -> None:
        self.shs = None
        self.shs_degree = None
        self.shs_ptr = None

    def add_shs(self, shs_degree: IntTensor):
        with torch.no_grad():
            if self.shs_degree is None:
                mem = ((shs_degree + 1) ** 2 - 1) * 3
                self.shs = torch.zeros(
                    [
                        torch.sum(mem),
                    ],
                    dtype=torch.float32,
                ).cuda()
                self.shs_degree = shs_degree
                self.shs_ptr = torch.cumsum(
                    torch.cat([torch.IntTensor([0]).cuda(), mem[:-1]], dim=0), dim=0
                ).int()
            else:
                new_shs_degree = self.shs_degree + shs_degree
                P = self.shs_degree.shape[0]
                old_idx = torch.arange(P).int().cuda()
                self.shs, self.shs_degree, self.shs_ptr = update_shs(
                    old_idx, self.shs, self.shs_degree, self.shs_ptr, new_shs_degree
                )

    def prune_points(self, mask):
        valid_points_mask = ~mask
        P = self.shs_degree.shape[0]
        old_idx = torch.arange(P)[valid_points_mask].int().cuda()
        new_shs_degree = self.shs_degree[valid_points_mask]
        self.shs, self.shs_degree, self.shs_ptr = update_shs(
            old_idx, self.shs, self.shs_degree, self.shs_ptr, new_shs_degree
        )


def update_shs(
    old_idx: IntTensor,
    old_shs: FloatTensor,
    old_shs_degree: IntTensor,
    old_shs_ptr: IntTensor,
    new_shs_degree: IntTensor,
):
    mem = ((new_shs_degree + 1) ** 2 - 1) * 3
    new_shs_ptr = torch.cumsum(
        torch.cat([torch.IntTensor([0]).cuda(), mem[:-1]], dim=0), dim=0
    ).int()
    new_shs = torch.zeros(
        [
            torch.sum(mem),
        ],
        dtype=torch.float32,
    ).cuda()
    print("old_idx", old_idx.dtype)
    print("old_shs", old_shs.dtype)
    print("old_shs_degree", old_shs_degree.dtype)
    print("old_shs_ptr", old_shs_ptr.dtype)
    print("new_shs", new_shs.dtype)
    print("new_shs_degree", new_shs_degree.dtype)
    print("new_shs_ptr", new_shs_ptr.dtype)
    _C.copy_shs(
        old_idx,
        old_shs,
        old_shs_degree,
        old_shs_ptr,
        new_shs,
        new_shs_degree,
        new_shs_ptr,
    )
    return new_shs, new_shs_degree, new_shs_ptr
