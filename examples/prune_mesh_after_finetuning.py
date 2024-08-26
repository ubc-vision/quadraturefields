"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import glob
from mesh_utils import MeshIntersection, MeshFinetune
import cv2
import argparse
import math
import json
import pathlib
import time
import sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField, NGPRadianceFieldSGNew
from field import Field
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_finetune_with_occgrid,
    render_image_field_with_occgrid,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
from matplotlib import pyplot as plt
import trimesh
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def collate_function(batch):
    return batch[0]


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
    type=str,
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
# log2_hashmap_size
parser.add_argument(
    "--log2_hashmap_size",
    type=int,
    default=19,
    help="log2_hashmap_size",
)
# log2_hashmap_size
parser.add_argument(
    "--scale",
    type=float,
    default=1.5,
    help="scale of the scene",
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
args = parser.parse_args()
args.optix = str2bool(args.optix)

args.exp_name = (
    args.exp_name
    + "_voxel_size_{}_max_hits_{}_num_lobes_{}_num_layers_{}_o_lambda_{}_up_{}_sc_{}".format(
        args.voxel_size,
        args.max_hits,
        args.num_lobes,
        args.num_layers,
        args.o_lambda,
        args.up_sample,
        args.scaling,
    )
)
if args.optix:
    from build.lib import intersector

device = "cuda:0"
set_random_seed(42)

device = "cuda:0"
set_random_seed(42)

mesh_intersect = MeshIntersection(mesh_path=args.mesh_path,
                                  scale=1.,
                                  simplify_mesh=False,
                                  optix=args.optix,
                                  num_intersections=args.max_hits,)

from torch.utils.tensorboard import SummaryWriter

# setup tensorboard logger to log values of the loss function

if args.scene in ["horse", "woolly"]:
    args.scale = 2.0

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
    train_dataset_ = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="train",
        num_rays=None,
        device=torch.device("cpu"),
        **test_dataset_kwargs,
        mesh_intersect=mesh_intersect,
        fine_tune_vertices=False,
        upsample=args.up_sample,
    )

    train_dataset = iter(DataLoader(
        train_dataset_,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_function,
    ))
    return train_dataset

train_dataset = initialize_dataloaders()

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
if args.num_lobes > 0:
    radiance_field = NGPRadianceFieldSGNew(
        aabb=estimator.aabbs[-1],
        use_viewdirs=False,
        num_g_lobes=args.num_lobes,
        num_layers=args.num_layers,
        log2_hashmap_size=args.log2_hashmap_size,
    ).to(device)
else:
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64, log2_hashmap_size=args.log2_hashmap_size).to(
        device
    )

# Dummy network to keep the code consistent
field_net = Field(scale=1.5,
                precision=16,
                log2_T=9,
                L=16,
                max_res=32,
                min_res=16,
                output_dim=1,
                hidden_size=2,
                num_features=2,
                back_prop=False,
                nl="relu").cuda()
mesh_finetune = MeshFinetune(mesh_intersect.mesh.vertices, mesh_intersect.mesh.faces, args.scaling)

ckpt = torch.load(args.ckpt_path)
radiance_field.load_state_dict(ckpt["radiance_field"])
estimator.load_state_dict(ckpt["estimator"])


lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
test_ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()

def test(new_scaling, prefix=""):
    torch.cuda.empty_cache()
    radiance_field.eval()
    estimator.eval()
    triangles_weights = torch.zeros(mesh_intersect.mesh.faces.shape[0]).cuda()
    from torch_scatter import scatter_max
    num_samples = []
    num_valid_samples = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(train_dataset._dataset.images))):
            data = next(train_dataset) #[i]
            render_bkgd = data["color_bkgd"].cuda()
            rays = data["rays"]
            rgb, _, depth, n_rendering_samples, weights, _, rays, _, index_tri = render_image_finetune_with_occgrid(
                        radiance_field,
                        field_net,
                        estimator,
                        rays,
                        data["data"],
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                        mesh_intersect=mesh_intersect,
                        mesh_finetune=mesh_finetune,
                        scaling=new_scaling
                    )
            num_samples.append(len(weights))
            num_valid_samples.append(torch.sum(weights > 0.001).data.cpu().numpy())
            torch.cuda.empty_cache()
            triangles_weights_i = torch.zeros_like(triangles_weights)
            scatter_max(weights[:, 0], index_tri, out=triangles_weights_i)
            triangles_weights = torch.maximum(triangles_weights, triangles_weights_i)
    return triangles_weights, num_samples, num_valid_samples

triangle_weights, num_samples, num_valid_samples = test(0.0)


mask = triangle_weights > 0.001
mask = mask.data.cpu().numpy().flatten()
mesh = mesh_intersect.mesh

print ("Number of faces before pruning: ", mesh.faces.shape[0])
mesh.update_faces(mask)
print("Number of faces after pruning: ", mesh.faces.shape[0])
# Save triangle weights
source_path = "/".join(args.mesh_path.split("/")[0:-1])
np.save("{}/triangle_weights.npy".format(source_path), triangle_weights.data.cpu().numpy())
mesh.export("{}/mesh_updated.ply".format(source_path))
# save number of samples, number of valid samples
np.save("{}/num_samples.npy".format(source_path), np.array(num_samples))
np.save("{}/num_valid_samples.npy".format(source_path), np.array(num_valid_samples))