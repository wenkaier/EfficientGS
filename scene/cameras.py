#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import h5py
import os
import os.path as osp
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,getProjectionMatrix_refine
from utils.general_utils import PILtoTorch,RGBtoTorch
from PIL import Image
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_full_name, resolution,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",K=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # try:
        #     self.data_device = torch.device(data_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #     self.data_device = torch.device("cuda")

        self.image_full_name = image_full_name
        self.resolution = resolution
        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        if K is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), self.image_height, self.image_width, self.znear, self.zfar).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.original_image = None
        if data_device != "disk":
            self.original_image = self.get_original_image().to(data_device)

    def get_original_image(self, load_from_h5: bool = False):
        if self.original_image is not None:
            return self.original_image
        if load_from_h5:
            h5_fn = osp.dirname(self.image_full_name) + ".h5"
            key = osp.basename(self.image_full_name)
            h5 = h5py.File(h5_fn, "r")
            img_rgb = h5[key][:]
            h5.close()
            resized_image_rgb = RGBtoTorch(img_rgb)
            return resized_image_rgb
        img = Image.open(self.image_full_name)
        resized_image_rgb = PILtoTorch(img, self.resolution)
        image = resized_image_rgb[:3, ...]
        img.close()
        original_image = image.clamp(0.0, 1.0)
        return original_image
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

