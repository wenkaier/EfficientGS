import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
import numpy as np

# copy from https://github.com/nkolot/SPIN/blob/2476c436013055be5cb3905e4e4ecfa86966fac3/utils/geometry.py#L47


# https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rot6d_to_quat(x) -> np.ndarray:
    rotmat = rot6d_to_rotmat(x)
    quat = Rotation.from_matrix(rotmat.cpu().numpy()).as_quat().astype(np.float32)
    return quat
