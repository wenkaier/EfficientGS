import torch
from torch import FloatTensor


# copy from
# https://github.com/KeKsBoTer/c3dgs/blob/master/scene/gaussian_model.py
def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: FloatTensor) -> FloatTensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer


def get_morton_order(xyz: FloatTensor):
    xyz_q = (
        (2**21 - 1)
        * (xyz - xyz.min(0).values)
        / (xyz.max(0).values - xyz.min(0).values)
    ).long()
    order = mortonEncode(xyz_q).sort().indices
    return order
