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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from time import time
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map
        # return ssim_map.mean(1).mean(1).mean(1)
def get_ignored_tiles(loss, ignored_rate: float, block=16):
    assert ignored_rate > 0
    h, w = loss.shape
    h_blocks = (h + block - 1) // block
    w_blocks = (w + block - 1) // block
    new_loss = torch.zeros([h_blocks * block, w_blocks * block]).float().cuda()
    new_loss[:h, :w] = loss
    new_loss = new_loss.reshape([h_blocks, block, w_blocks, block]).permute(0, 2, 1, 3).reshape([h_blocks * w_blocks, -1])
    new_loss = torch.mean(new_loss, dim=-1)
    idxs = torch.argsort(new_loss)  # 从小到大排序
    n = int(h_blocks * w_blocks * ignored_rate)
    n = min(n, h_blocks * w_blocks - 1)
    limit_loss = new_loss[idxs[n]]
    ignored_tiles = torch.zeros_like(new_loss).int().cuda()
    ignored_tiles[new_loss < limit_loss] = 1
    return ignored_tiles

def cal_loss(image, gt_image, lambda_dssim):
    Ll1 = torch.abs(image - gt_image)
    ssim_loss = ssim(image, gt_image, size_average=False)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
        1.0 - ssim_loss
    )
    return loss.mean(0)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    top_k,
    enable_loss_weight,
    pix_loss,
    sparse_shs_degree,
    gt_image,
    extract_gs_info,
    reg_loss_k,
    in_gs_info,extract_gs_rays_num,
    max_gs_num,gs_radius
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        top_k,
        enable_loss_weight,
        pix_loss,
        sparse_shs_degree,
        gt_image,
        extract_gs_info,
        reg_loss_k,
        in_gs_info,extract_gs_rays_num,
        max_gs_num,gs_radius
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        top_k,
        enable_loss_weight,
        pix_loss,
        sparse_shs_degree,
        gt_image,
        extract_gs_info,
        reg_loss_k,
        in_gs_info,extract_gs_rays_num,
        max_gs_num,gs_radius
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            top_k,
            enable_loss_weight,
            pix_loss,
            sparse_shs_degree,
            raster_settings.use_hwc,
            extract_gs_info,
            reg_loss_k,
            in_gs_info,
            extract_gs_rays_num,
            max_gs_num,
            gs_radius,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, max_weight_pids, out_gs_info,out_reg_loss,out_loss_weight,out_rays_num, out_gs_weight, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, max_weight_pids, out_gs_info,out_reg_loss,out_loss_weight,out_rays_num, out_gs_weight,radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        
        # ignored_tiles
        ignored_tiles = torch.zeros([1]).cuda().int()
        if raster_settings.use_train_speedup:
            # torch.cuda.synchronize()
            # t0 = time()
            ignored_tiles = get_ignored_tiles(cal_loss(color, gt_image, raster_settings.lambda_dssim), ignored_rate=raster_settings.ignored_rate, block=16)
            # torch.cuda.synchronize()
            # t1 = time()
            # print(t1 - t0)
        ctx.ignored_tiles = ignored_tiles
        ctx.sparse_shs_degree = sparse_shs_degree
        ctx.reg_loss_k = reg_loss_k
        ctx.in_gs_info = in_gs_info
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        
        return color, radii, max_weight_pids, out_gs_info,out_reg_loss,out_loss_weight,out_rays_num, out_gs_weight

    @staticmethod
    def backward(ctx, grad_out_color, _, __, ___,grad_out_reg_loss,xxxx,xxxxx,xxxxxx):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        ignored_tiles = ctx.ignored_tiles
        sparse_shs_degree = ctx.sparse_shs_degree
        reg_loss_k = ctx.reg_loss_k
        in_gs_info = ctx.in_gs_info
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.use_train_speedup,
                ignored_tiles,
                sparse_shs_degree,
                reg_loss_k,
                in_gs_info,
                grad_out_reg_loss,
                raster_settings.use_hwc,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D,grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    use_train_speedup: bool
    ignored_rate: float
    lambda_dssim: float
    use_hwc: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, gt_image,
            shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, top_k=1, pix_loss=None,sparse_shs_degree=None,extract_gs_info: bool = False,
            reg_loss_k=-1,
            in_gs_info=None,
            extract_gs_rays_num: bool = False,
            max_gs_num: int = -1, gs_radius:float = 3.0):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        
        enable_loss_weight = pix_loss is not None
        if pix_loss is None:
            pix_loss = torch.Tensor([])
        if sparse_shs_degree is None:
            sparse_shs_degree = torch.Tensor([]).int()
        if in_gs_info is None:
            in_gs_info = torch.Tensor([]).int()
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            top_k,
            enable_loss_weight,
            pix_loss,
            sparse_shs_degree,
            gt_image,
            extract_gs_info,
            reg_loss_k,
            in_gs_info,
            extract_gs_rays_num,
            max_gs_num,gs_radius
        )

