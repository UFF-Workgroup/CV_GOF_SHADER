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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, brdf_args=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")\

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, brdf_args=brdf_args)["render"]
        rendering = rendering[:3, :, :]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}_{1}'.format(idx, view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}_{1}'.format(idx, view.image_name) + ".png"))

def select_views(views, selector):
    # selector: indices e/ou image_names separados por virgula, ou None para todas as vistas.
    if selector is None:
        return views
    tokens = [t.strip() for t in selector.split(",") if t.strip()]
    by_name = {v.image_name: v for v in views}
    selected = []
    for t in tokens:
        if t.isdigit() and int(t) < len(views):
            selected.append(views[int(t)])
        elif t in by_name:
            selected.append(by_name[t])
        else:
            raise ValueError(f"Vista '{t}' nao encontrada (nem indice, nem image_name)")
    return selected

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, views : str = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, select_views(scene.getTrainCameras(), views), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, brdf_args=dataset)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, select_views(scene.getTestCameras(), views), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, brdf_args=dataset)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--views", default=None, type=str,
                         help="indices e/ou image_names separados por virgula (ex.: '3,7,21'); default renderiza todas as vistas do split")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.views)