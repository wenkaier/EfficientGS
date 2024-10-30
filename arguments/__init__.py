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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.use_sphere = False  # add by lwk
        self.use_rot6d = False  # add by lwk
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"  # disk, cpu, cuda
        self.eval = False
        self._no_gui = False
        self.mill19_llffhold = -1
        self.ignore_check_output = False  # 是否忽略检查输出文件夹是否已经存在, 如果不忽略检查并且文件夹已经存在，则不重新训练
        self.use_K = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.gs_radius = 3.0
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.shs_lr_rate = 20
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        
        # by lwk
        # 删点策略
        self.prune_by_max_weight = False
        self.prune_way = ""  # "p1",
        self.top_k = 1
        
        # 分裂策略
        self.use_grad_norm = False
        self.grad_norm_way = ""  # "s1"
        self.oneupSHdegree_from_iter = 0
        self.oneupSHdegree_interval = 1000
        self.not_reset_opacity = False
        
        # 加速训练策略
        self.use_train_speedup = False
        self.train_speedup_way = ""  # "t1"
        
        # 多进程加载图片
        self.prefetch_factor = 3
        self.num_workers = 6
        self.load_from_h5 = False
        
        self.sphere_to_ellipsoid_iter = -1

        # reg loss
        self.reg_loss_from_iter = -1
        self.reg_loss_weight = 100.0
        self.reg_loss_k = 3

        self.ios_loss_weight = -1.0
        self.dis_loss_weight = -1.0
        
        # Sparse SHs
        self.use_sparse_shs = False
        self.sparse_rate = 0.1
        self.sparse_cnt = 3
        self.sparse_way = "pix_loss"  # random
        
        self.use_hwc = False
        
        self.not_shuffle_image = False
        self.window_size = 11
        
        self.size_threshold = 20
        self.min_opacity = 0.005
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
