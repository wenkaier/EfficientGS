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
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from torch import IntTensor

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    top_k=-1,
    pix_loss:torch.Tensor=None,
    use_train_speedup:bool=False,
    gt_image:torch.Tensor=None,
    lambda_dssim:float=-1,
    ignored_rate:float=0,
    no_grad: bool = False,
    use_hwc: bool = False,
    extract_gs_info: bool = False,  # 输出每个像素对应的(高斯点数量, 最大权重所在位置，最大权重的高斯编号)
    reg_loss_k: int = -1,
    in_gs_info: IntTensor=None,
    extract_gs_rays_num:bool = False,
    max_gs_num : int = -1,  # 提取权重
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if not no_grad:
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

    else:
        screenspace_points = torch.zeros([0, 1], requires_grad=True).cuda()
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        use_train_speedup=use_train_speedup,
        ignored_rate=ignored_rate,
        lambda_dssim=lambda_dssim,
        use_hwc=use_hwc
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, max_weight_pids,out_gs_info,out_reg_loss, out_loss_weight, out_rays_num, out_gs_weight = rasterizer(
        means3D=means3D,
        means2D=means2D,
        gt_image=gt_image,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        top_k=top_k,
        pix_loss=pix_loss,
        sparse_shs_degree=pc.sparse_shs_degree,
        extract_gs_info=extract_gs_info,
        reg_loss_k=reg_loss_k,
        in_gs_info=in_gs_info,
        extract_gs_rays_num=extract_gs_rays_num,
        max_gs_num=max_gs_num,
        gs_radius=pipe.gs_radius
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if use_hwc:
        rendered_image = rendered_image.permute(2, 0, 1)

    if len(out_gs_info) > 0:
        _, h, w = rendered_image.shape
        out_gs_info = out_gs_info.reshape([h, w, 3])
    return {
        "render": rendered_image,
        "max_weight_pids": max_weight_pids,
        "gs_weight": out_gs_weight,
        "gs_info": out_gs_info,
        "reg_loss": out_reg_loss,
        "loss_weight": out_loss_weight,
        "rays_num": out_rays_num,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
