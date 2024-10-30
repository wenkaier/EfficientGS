import torch
from scene import Scene
import os
import os.path as osp
from tqdm import tqdm
from os import makedirs
from scene.renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.renderer import GaussianModel
from tqdm import tqdm
from time import time


# https://github.com/graphdeco-inria/gaussian-splatting/issues/422
# https://github.com/graphdeco-inria/gaussian-splatting/issues/349


def test_fps(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    txt_fn = osp.join(dataset.model_path, f"fps_{iteration}.txt")
    if osp.exists(txt_fn):
        print("osp.exists:", txt_fn)
        return

    ply_fn = osp.join(
        dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"
    )
    if not osp.exists(ply_fn):
        return

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cameras = scene.getTrainCameras() + scene.getTestCameras()

        iters = 10
        torch.cuda.synchronize()
        t0 = time()
        for _ in tqdm(range(iters), desc="test_fps"):
            for view in cameras:
                rendering = render(view, gaussians, pipeline, background, no_grad=True)[
                    "render"
                ]
        torch.cuda.synchronize()
        t1 = time()
        t = t1 - t0
        fps = int(iters * len(cameras) / t)
        print("fps", fps)
        with open(txt_fn, "w") as fw:
            fw.write(str(fps))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Run fps " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    test_fps(model.extract(args), args.iteration, pipeline.extract(args))
