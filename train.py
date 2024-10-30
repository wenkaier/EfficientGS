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

import os
import os.path as osp
import torch

from random import randint
from scene.renderer import render
from utils.loss_utils import l1_loss, ssim

# from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel, CamerasDataset
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.json_file import save_json
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from time import time
import copy
from simple_knn._C import distCUDA2
from torch import FloatTensor


def tic():
    torch.cuda.synchronize()
    return time()


def collect_fn(x):
    return x


def cal_iso_loss(gaussians: GaussianModel):
    s = gaussians.get_scaling
    s_mean = torch.mean(s, dim=-1, keepdim=True)
    iso_loss = torch.mean(torch.abs(s - s_mean))
    return iso_loss


def cal_dis_loss(gaussians: GaussianModel):
    xyz: FloatTensor = gaussians.get_xyz
    dist = torch.clamp_min(
        torch.sqrt(distCUDA2(xyz.detach())),
        0.0000001,
    )
    s = gaussians.get_scaling
    return torch.mean(torch.clamp_min(s - dist.unsqueeze(-1), 0))


def get_sparse_shs_mask(
    scene: Scene, gaussians: GaussianModel, pipe, bg, opt, load_from_h5: bool
):
    if load_from_h5:
        dataset = CamerasDataset(
            scene.getTrainCameras(), load_from_h5=load_from_h5, to_cuda=False
        )
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=opt.num_workers,
            prefetch_factor=opt.prefetch_factor,
            collate_fn=collect_fn,
        )
    else:
        data_loader = scene.getTrainCameras()
    with torch.no_grad():
        loss_weight_sum = None
        for batch in tqdm(data_loader, desc="get_sparse_shs_mask"):
            if load_from_h5:
                cam = batch[0]["cam"]
                gt_image = batch[0]["image"].cuda()
            else:
                cam = batch
                gt_image = batch.get_original_image().cuda()
            pred_image = render(cam, gaussians, pipe, bg)["render"]
            pix_loss = torch.abs(pred_image - gt_image)
            loss_weight = render(cam, gaussians, pipe, bg, pix_loss=pix_loss)[
                "loss_weight"
            ]
            if loss_weight_sum is None:
                loss_weight_sum = loss_weight
            else:
                loss_weight_sum += loss_weight
        n = loss_weight_sum.shape[0]
        k = int(n * opt.sparse_rate)
        idx = torch.argsort(-loss_weight_sum)[k]
        threshold = loss_weight_sum[idx]
        mask = loss_weight_sum >= threshold
    return mask


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    no_gui,
    load_from_h5,
):
    if load_from_h5:
        assert dataset.resolution == 1
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(
        dataset.sh_degree if opt.oneupSHdegree_from_iter == 0 else 0,
        dataset.use_sphere,
        dataset.use_rot6d,
    )
    scene = Scene(dataset, gaussians, shuffle=not opt.not_shuffle_image)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    use_data_cpu = "cuda" not in dataset.data_device
    if use_data_cpu:
        scene_loader = DataLoader(
            CamerasDataset(
                scene.getTrainCameras(),
                samples_num=opt.iterations,
                load_from_h5=load_from_h5,
            ),
            batch_size=1,
            num_workers=opt.num_workers,
            prefetch_factor=opt.prefetch_factor,
            collate_fn=collect_fn,
            shuffle=True,
        )
        scene_loader = iter(scene_loader)

    train_start = time()
    report_time = 0
    sparse_cnt = opt.sparse_cnt
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if opt.oneupSHdegree_from_iter > 0 and iteration == opt.oneupSHdegree_from_iter:
            gaussians.set_shs_memory(dataset.sh_degree)
            if opt.use_sparse_shs:
                n = gaussians._xyz.shape[0]
                gaussians.sparse_shs_degree = torch.zeros(
                    [
                        n,
                    ],
                    dtype=torch.int32,
                ).cuda()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (
            iteration % opt.oneupSHdegree_interval == 0
            and iteration >= opt.oneupSHdegree_from_iter
        ):
            if opt.use_sparse_shs and sparse_cnt > 0:
                sparse_cnt -= 1
                if opt.sparse_way == "pix_loss":
                    mask = get_sparse_shs_mask(
                        scene, gaussians, pipe, bg, opt, load_from_h5=load_from_h5
                    )
                elif opt.sparse_way == "random":
                    n = gaussians.get_xyz.shape[0]
                    mask = (
                        torch.zeros(
                            [
                                n,
                            ]
                        )
                        .bool()
                        .cuda()
                    )
                    idx = torch.randperm(n)[: int(n * opt.sparse_rate)]
                    mask[idx] = True
                else:
                    assert False, "bad sparse_way"
                gaussians.sparse_shs_degree[mask] += 1
                mask = gaussians.sparse_shs_degree > gaussians.max_sh_degree
                gaussians.sparse_shs_degree[mask] = gaussians.max_sh_degree
            gaussians.oneupSHdegree()
        if iteration == opt.sphere_to_ellipsoid_iter:
            gaussians.sphere_to_ellipsoid()
        # Pick a random Camera
        if use_data_cpu:
            batch = next(scene_loader)
            assert len(batch) == 1
            viewpoint_cam = batch[0]["cam"]
            gt_image = batch[0]["image"].cuda()
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            gt_image = viewpoint_cam.get_original_image().cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        use_reg_loss = (
            opt.reg_loss_from_iter > 0 and iteration >= opt.reg_loss_from_iter
        )

        gs_info = None
        if use_reg_loss:
            with torch.no_grad():
                gs_info = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    use_hwc=opt.use_hwc,
                    extract_gs_info=True,
                )["gs_info"]

        if opt.use_train_speedup:
            ignored_rate = -1
            if opt.train_speedup_way == "t1":
                if iteration >= 0:
                    ignored_rate = 0.2
                if iteration >= 5000:
                    ignored_rate = 0.5
            elif opt.train_speedup_way == "t2":
                if iteration >= 10000:
                    ignored_rate = 0.5
            else:
                assert False, "bad train_speedup_way"
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_train_speedup=ignored_rate > 0,
                lambda_dssim=opt.lambda_dssim,
                ignored_rate=ignored_rate,
                gt_image=gt_image,
                use_hwc=opt.use_hwc,
            )
        else:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_hwc=opt.use_hwc,
                reg_loss_k=opt.reg_loss_k if use_reg_loss else -1,
                in_gs_info=gs_info,
            )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image, window_size=opt.window_size)
        )
        reg_loss = torch.zeros([1]).float().cuda()
        if use_reg_loss:
            reg_loss = torch.mean(render_pkg["reg_loss"])
            loss += reg_loss * opt.reg_loss_weight

        ios_loss = torch.zeros([1]).float().cuda()
        if opt.ios_loss_weight > 0:
            ios_loss = cal_iso_loss(gaussians)
            loss += ios_loss * opt.ios_loss_weight

        dis_loss = torch.zeros([1]).float().cuda()
        if opt.dis_loss_weight > 0:
            dis_loss = cal_dis_loss(gaussians)
            loss += dis_loss * opt.dis_loss_weight
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            report_start = time()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                reg_loss,
                ios_loss,
                dis_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                pipe,
                background,
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            report_end = time()
            report_time += report_end - report_start
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    use_grad_norm=opt.use_grad_norm,
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        opt.size_threshold
                        if iteration > opt.opacity_reset_interval
                        else None
                    )

                    threshold = opt.densify_grad_threshold
                    if opt.use_grad_norm:
                        if opt.grad_norm_way == "s1":
                            pass
                        elif opt.grad_norm_way == "s2":
                            if iteration >= 5000:
                                threshold *= 0.8
                            if iteration >= 10000:
                                threshold *= 0.8
                        else:
                            assert False, "bad grad_norm_way"

                    points_mask = None
                    gaussians.densify_and_prune(
                        # opt.densify_grad_threshold,
                        threshold,
                        opt.min_opacity,  # 0.005,
                        scene.cameras_extent,
                        size_threshold,
                        points_mask=points_mask,
                    )

            # prune_by_max_weight
            if opt.prune_by_max_weight:
                top_k = -1
                view_num = 1
                if opt.prune_way == "p1":
                    if iteration >= 15000:
                        k = iteration - 15000
                        if k % 2000 == 0:
                            top_k = opt.top_k
                            view_num = 1
                elif opt.prune_way == "p2":
                    if iteration % 3000 == 0:
                        top_k = opt.top_k
                        view_num = 1
                elif opt.prune_way == "p3":
                    if iteration in [10000, 15000, 18000]:
                        top_k = opt.top_k
                        view_num = 1
                elif opt.prune_way == "p4":
                    if iteration in [15000, 18000, 21000]:
                        top_k = opt.top_k
                        view_num = 1
                elif opt.prune_way == "p5":
                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        top_k = opt.top_k
                        view_num = 1
                elif opt.prune_way == "p6":
                    if iteration in [
                        opt.densify_until_iter,
                        opt.densify_until_iter + 3000,
                        opt.densify_until_iter + 6000,
                    ]:
                        top_k = opt.top_k
                        view_num = 1
                else:
                    assert False, "bad prune_way"

                if top_k >= 1:
                    assert top_k <= 10  # forward 存储上限
                    max_weight_pids_sum = None
                    for cam in scene.getTrainCameras():
                        max_weight_pids = render(cam, gaussians, pipe, bg, top_k=top_k)[
                            "max_weight_pids"
                        ]
                        if max_weight_pids_sum is None:
                            max_weight_pids_sum = max_weight_pids
                        else:
                            max_weight_pids_sum += max_weight_pids
                    points_mask = max_weight_pids_sum < view_num
                    gaussians.prune_points(points_mask)

            if iteration < opt.densify_until_iter and (not opt.not_reset_opacity):
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    train_time = time() - train_start - report_time
    json_fn = osp.join(dataset.model_path, "time.json")
    train_time = train_time / 60.0
    report_time = report_time / 60.0
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    save_json(
        json_fn,
        {
            "train_time": f"{train_time:.2f}",
            "report_time": f"{report_time:.2f}",
            "max_memory_allocated": max_memory_allocated,
            "max_memory_reserved": max_memory_reserved,
        },
    )
    print(f"train_time: {train_time:.2f} mins")
    print(f"report_time: {report_time:.2f} mins")
    print(f"max_memory_allocated: {max_memory_allocated}")
    print(f"max_memory_reserved: {max_memory_reserved}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    if not args.ignore_check_output:
        if os.path.exists(args.model_path):
            print("exists:", args.model_path)
            assert False, "os.path.exists(args.model_path)"
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    reg_loss,
    ios_loss,
    dis_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    pipe,
    background,
):
    if iteration % 100 == 0:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/reg_loss", reg_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ios_loss", ios_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/dis_loss", dis_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("scalar/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "scalar/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
        tb_writer.add_scalar(
            "scalar/memory_allocated", torch.cuda.memory_allocated(), iteration
        )
        tb_writer.add_scalar(
            "scalar/max_memory_allocated", torch.cuda.max_memory_allocated(), iteration
        )
        tb_writer.add_scalar(
            "scalar/max_memory_reserved", torch.cuda.max_memory_reserved(), iteration
        )
        tb_writer.add_scalar("scalar/xyz_lr", scene.gaussians.get_xyz_lr(), iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        render(viewpoint, scene.gaussians, pipe, background)["render"],
                        0.0,
                        1.0,
                    )
                    # gt_image = torch.clamp(
                    #     viewpoint.original_image.to("cuda"), 0.0, 1.0
                    # )
                    gt_image = viewpoint.get_original_image().cuda()
                    if False and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            5000,
            7_000,
            10000,
            15000,
            20000,
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
        ],
    )
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(0, 30001, 500))[1:])
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 90000],
    )
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.no_gui:
    #     # network_gui.init(args.ip, args.port)
    #     print("use gui")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.no_gui,
        args.load_from_h5,
    )

    # All done
    print("\nTraining complete.")
