"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from mesh_utils import MeshIntersection, MeshFinetune
import cv2
import json

import argparse
import math
import pathlib
import time
import sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceFieldSGNew
from field import Field
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_bake_texture_images_with_occgrid,
    render_image_field_with_occgrid,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
from matplotlib import pyplot as plt
import trimesh
import os
from einops import rearrange
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/arc/burst/pr-kmyi-1/gopalshr/nerfacc/",
    help="the root",
)
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="finetune",
    help="experiment name",
)
parser.add_argument(
    "--mesh_path",
    type=str,
    default="",
    help="mesh_path",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    # choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--scaling",
    type=float,
    default=3 / 128,
    help="scaling for deformation field",
)
parser.add_argument(
    "--up_sample",
    type=float,
    default=1.0,
    help="image upsampling factor",
)
parser.add_argument(
    "--optix",
    type=str2bool,
    default="False",
    help="whether to use nvidia optix for ray tracing",
)
parser.add_argument(
    "--voxel_size",
    type=int,
    default=512,
    help="Voxel size used to downsample",
)
parser.add_argument(
    "--texture_size",
    type=int,
    default=4096,
    help="Voxel size used to downsample",
)
parser.add_argument(
    "--max_hits",
    type=int,
    default=10,
    help="Maximum number of hits per ray",
)
# parser for number of lobes
parser.add_argument(
    "--num_lobes",
    type=int,
    default=0,
    help="number of lobes",
)
parser.add_argument(
    "--o_lambda",
    type=float,
    default=1e-3,
    help="scale factor for the opacity loss",
)
# parser for number of layers
parser.add_argument(
    "--num_layers",
    type=int,
    default=1,
    help="number of layers",
)
parser.add_argument(
    "--out_activation",
    type=str,
    default="sigmoid",
    help="output activation function",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="",
    help="checkpoint path",
)
parser.add_argument(
    "--discretize",
    type=str2bool,
    default=False,
    help="whether to discretize the features",
)
# log2_hashmap_size
parser.add_argument(
    "--log2_hashmap_size",
    type=int,
    default=19,
    help="log2_hashmap_size",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.5,
    help="scale of the scene",
)
# lambda_thres
parser.add_argument(
    "--lambda_thres",
    type=float,
    default=7.5,
    help="lambda_thres",
)

# compression_type
parser.add_argument(
    "--compression_type",
    type=str,
    default="linear",
    help="compression_type",
)

args = parser.parse_args()

if args.scene in ["horse", "woolly"]:
    args.scale = 2.0

device = "cuda:0"
set_random_seed(42)
if args.optix:
    from build.lib import intersector
mesh_intersect = MeshIntersection(mesh_path=args.mesh_path,
                                  scale=1.,
                                  simplify_mesh=False,
                                  optix=args.optix,
                                  num_intersections=args.max_hits,
                                  render_step_size=5e-3, )

from torch.utils.tensorboard import SummaryWriter

# setup tensorboard logger to log values of the loss function
# logger = SummaryWriter("logs/{}/{}".format(args.scene, args.exp_name), flush_secs=200)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 50000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 16
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 50000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 16
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device) * args.scale
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

from torch.utils.data import DataLoader


def collate_function(batch):
    return batch[0]


def initialize_dataloaders():
    test_dataset_ = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=torch.device("cpu"),
        **test_dataset_kwargs,
        mesh_intersect=mesh_intersect,
        fine_tune_vertices=False,
        upsample=args.up_sample,
    )

    test_dataset = iter(DataLoader(
        test_dataset_,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_function,
    ))
    return test_dataset


test_dataset = initialize_dataloaders()

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)
if args.num_lobes > 0:
    radiance_field = NGPRadianceFieldSGNew(
        aabb=estimator.aabbs[-1],
        use_viewdirs=False,
        num_g_lobes=args.num_lobes,
        num_layers=args.num_layers,
        discretize=args.discretize,
        log2_hashmap_size=args.log2_hashmap_size,
    ).to(device)
else:
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64,
                                      log2_hashmap_size=args.log2_hashmap_size
                                      ).to(
        device
    )
os.makedirs(args.root + "results/{}/{}/".format(args.scene, args.exp_name), exist_ok=True)
args.exp_name = args.ckpt_path.split("/")[-2]

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
test_ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
save_path = "/".join(args.mesh_path.split("/")[:-1]) + "/texture_{}.npy".format(args.texture_size)

SIZE = args.texture_size

mesh = trimesh.load(args.mesh_path, process=False)

uv = mesh.visual.uv - 1e-7
uv = np.array(uv).astype(np.float32) * SIZE
uv = np.clip(uv, 0, SIZE - 1)
uv = torch.from_numpy(uv).cuda()


texture_path = "/".join(args.mesh_path.split("/")[:-1]) + "/texture_{}/".format(SIZE)
from texture_utils import FeatureCompression
compressor = FeatureCompression(args.num_lobes, initialize=False, 
                                texture_size=args.texture_size, 
                                path=texture_path,
                                compression_type=args.compression_type,
                                lambda_thres=args.lambda_thres)
mask = compressor.alpha > 0

texture=None
def test(new_scaling, prefix=""):
    torch.cuda.empty_cache()
    radiance_field.eval()
    estimator.eval()

    psnrs = []
    lpips = []
    ssims = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_dataset._dataset.images))):
            data = next(test_dataset)
            render_bkgd = data["color_bkgd"].cuda()
            rays = data["rays"]
            pixels = data["pixels"].cuda()
            rgb, _, depth, n_rendering_samples, _, _, rays, _ = render_image_bake_texture_images_with_occgrid(
                radiance_field,
                rays,
                data["data"],
                texture=texture,
                uv=uv,
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                mesh_intersect=mesh_intersect,
                mesh_finetune=None,
                scaling=new_scaling,
                discretize=args.discretize,
                compressor=compressor,
            )
            depth = depth - depth.min()
            depth = depth / (depth.max() + 1e-6)
            torch.cuda.empty_cache()

            rgb = rgb.reshape(int(test_dataset._dataset.HEIGHT), int(test_dataset._dataset.WIDTH), 3)
            pixels = pixels.reshape(int(test_dataset._dataset.HEIGHT // args.up_sample),
                                    int(test_dataset._dataset.WIDTH // args.up_sample), 3)
            depth = depth.reshape(int(test_dataset._dataset.HEIGHT), int(test_dataset._dataset.WIDTH))

            rgb = cv2.resize(rgb.cpu().numpy(), (
            int(test_dataset._dataset.WIDTH // args.up_sample), int(test_dataset._dataset.HEIGHT // args.up_sample)),
                             interpolation=cv2.INTER_AREA)
            depth = cv2.resize(
                depth.cpu().numpy(), (int(test_dataset._dataset.WIDTH // args.up_sample),
                                      int(test_dataset._dataset.HEIGHT // args.up_sample)), interpolation=cv2.INTER_AREA
            )
            rgb = torch.from_numpy(rgb).cuda()
            depth = torch.from_numpy(depth).cuda()

            test_ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0))
            ssim = test_ssim.compute()
            test_ssim.reset()
            ssims.append(ssim.item())

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            lpips.append(lpips_fn(rgb, pixels).item())
            rgb = torch.clamp(rgb, 0, 1)
            imageio.imwrite(
                args.root + "results/{}/{}/rgb_test_baking_new_{}_{}.png".format(args.scene, args.exp_name, prefix, i),
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                args.root + "results/{}/{}/depth_baking_new_{}_{}.png".format(args.scene, args.exp_name, prefix, i),
                (depth.cpu().numpy() * 255).astype(np.uint8),
            )
            print("psnr: ", psnr.item())
            print("lpips: ", lpips_fn(rgb, pixels).item())
            print("ssim: ", ssim.item())
    psnr_avg = sum(psnrs) / len(psnrs)
    lpips_avg = sum(lpips) / len(lpips)
    ssim_avg = sum(ssims) / len(ssims)
    return psnr_avg, lpips_avg, ssim_avg


psnr_avg, lpips_avg, ssim_avg = test(0, prefix=str(args.texture_size) + "_" + str(args.discretize))
print("before psnr: ", psnr_avg)
print("before lpips: ", lpips_avg)
print("before ssim: ", ssim_avg)
# write the results in json file

exp_name = args.ckpt_path.split("/")[-2]
scene = args.scene

with open(args.root + "results/{}/{}/results_baking_textureimage_{}_{}_{}.json".format(args.scene, exp_name, str(args.texture_size),
                                                                           args.discretize, args.up_sample), "w") as f:
    json.dump({"psnr": psnr_avg, "lpips": lpips_avg, "ssim": ssim_avg}, f)
