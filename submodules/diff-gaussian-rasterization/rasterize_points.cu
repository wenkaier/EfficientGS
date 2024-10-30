/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int top_k,
	bool enable_loss_weight,
	const torch::Tensor& pix_loss,
	const torch::Tensor& sparse_shs_degree,
	const bool use_hwc,
	const bool extract_gs_info,
	const int reg_loss_k,
	const torch::Tensor& in_gs_info,
	const bool extract_gs_rays_num,
	const int max_gs_num,
	float gs_radius,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color;
  if(use_hwc){
	out_color = torch::full({H, W, NUM_CHANNELS}, 0.0, float_opts);
  }else{
	out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  }
  torch::Tensor out_gs_info;
  if(extract_gs_info){
	out_gs_info = torch::full({H, W, 3}, 0, int_opts);
  }else{
	out_gs_info = torch::full({0}, 0, int_opts);
  }

  torch::Tensor out_reg_loss;
  if(reg_loss_k>=0){
	out_reg_loss = torch::full({H, W, NUM_CHANNELS}, 0.0, float_opts);
  }else{
	out_reg_loss = torch::full({0}, 0.0, float_opts);
  }

  torch::Tensor out_rays_num;
  if(extract_gs_rays_num){
	out_rays_num = torch::full({P*2}, 0, float_opts);
  }else{
	out_rays_num = torch::full({0}, 0, float_opts);
  }

  torch::Tensor out_gs_weight;
  if(max_gs_num>0){
	out_gs_weight = torch::full({H, W, max_gs_num}, 0.0, float_opts);
  }else{
	out_gs_weight = torch::full({0}, 0.0, float_opts);
  }


  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Tensor out_max_weight_pids = torch::full({top_k > 0 ? P : 0}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_loss_weight = torch::full({enable_loss_weight ? P : 0}, 0, means3D.options().dtype(torch::kFloat32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  bool use_rotmat = rotations.size(1) == 9;
	  bool use_sparse_shs = sparse_shs_degree.size(0) != 0;

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		use_rotmat,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_max_weight_pids.contiguous().data<int>(),
		top_k,
		enable_loss_weight,
		pix_loss.contiguous().data<float>(),
		out_loss_weight.contiguous().data<float>(),
		use_sparse_shs,
		sparse_shs_degree.contiguous().data<int>(),
		extract_gs_info,
		out_gs_info.contiguous().data<int>(),
		reg_loss_k,
		in_gs_info.contiguous().data<int>(),
		out_reg_loss.contiguous().data<float>(),
		extract_gs_rays_num,
		out_rays_num.contiguous().data<float>(),
		max_gs_num,
		out_gs_weight.contiguous().data<float>(),
		gs_radius,
		radii.contiguous().data<int>(),
		use_hwc,
		debug);
  }
  return std::make_tuple(rendered, out_color, out_max_weight_pids, out_gs_info, out_reg_loss,out_loss_weight,out_rays_num, out_gs_weight,radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool use_train_speedup,
	const torch::Tensor& ignored_tiles,
	const torch::Tensor& sparse_shs_degree,
	const int reg_loss_k,
    const torch::Tensor& in_gs_info,
	const torch::Tensor& dL_reg_loss,
	const bool use_hwc,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(use_hwc ? 0 : 1) ;
  const int W = dL_dout_color.size(use_hwc ? 1 : 2) ;
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

	bool use_rotmat = rotations.size(1) == 9;
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());

  torch::Tensor dL_drotations = torch::zeros({P, use_rotmat ? 9: 4}, means3D.options());
  
  if(P != 0)
  {  
	  bool use_sparse_shs = sparse_shs_degree.size(0) != 0;
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  use_rotmat,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  use_train_speedup,
	  ignored_tiles.contiguous().data<int>(),
	  use_sparse_shs,
	  sparse_shs_degree.contiguous().data<int>(),
	  reg_loss_k,
	  in_gs_info.contiguous().data<int>(),
	  dL_reg_loss.contiguous().data<float>(),
	  use_hwc,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}