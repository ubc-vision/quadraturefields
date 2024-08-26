"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import glob
from mesh_utils import MeshIntersection, MeshFinetune
import cv2
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
from radiance_fields.ngp import NGPRadianceField, NGPRadianceFieldSGNew
from field import Field
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
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
from parameterization_utils import sample_points_on_triangle
from parameterization_utils import fill_triangles
from texture_utils import FeatureCompression


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
    "--texture_size",
    type=int,
    default=4096,
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
    "--ckpt_path_sg",
    type=str,
    default="",
    help="checkpoint path",
)
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

device = "cuda:0"
set_random_seed(42)
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
    max_steps = 24000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 17
    weight_decay = 1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
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


estimator = OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

radiance_field_sg = NGPRadianceFieldSGNew(
    aabb=estimator.aabbs[-1],
    use_viewdirs=False,
    num_g_lobes=args.num_lobes,
    num_layers=args.num_layers,
    log2_hashmap_size=args.log2_hashmap_size,
).to(device)
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64,log2_hashmap_size=args.log2_hashmap_size).to(
        device
    )

ckpt = torch.load(args.ckpt_path)
radiance_field.load_state_dict(ckpt["radiance_field"])

ckpt = torch.load(args.ckpt_path_sg)
radiance_field_sg.load_state_dict(ckpt["radiance_field"])

radiance_field.eval()
radiance_field_sg.eval()

root_path = "/".join(args.mesh_path.split("/")[0:-1])


batch_size = 100000
mesh = trimesh.load(args.mesh_path, process=False)
mesh_name = args.mesh_path.split("/")[-1].split(".")[0]

V = np.load(root_path + "/V_{}.npy".format(args.texture_size)).astype(np.float32)

compressor = FeatureCompression(args.num_lobes, initialize=True, 
                                texture_size=args.texture_size, 
                                path=None,
                                compression_type=args.compression_type,
                                lambda_thres=args.lambda_thres)

mask = V.sum(-1) == 0
mask = ~mask
ind = np.argwhere(mask)
plt.imsave("{}/mask_V_{}.png".format(root_path, args.texture_size), mask)

for b in range(0, ind.shape[0], batch_size):
    with torch.no_grad():
        indices = np.arange(b, min(b + batch_size, ind.shape[0]))
        features = radiance_field_sg.features(torch.from_numpy(V[ind[indices][:, 0], ind[indices][:, 1]]).cuda())
        density = radiance_field.query_density(torch.from_numpy(V[ind[indices][:, 0], ind[indices][:, 1]]).cuda())
        features[..., -1] = density.flatten()
        compressor.load_features_into_maps(features, torch.from_numpy(ind[indices]).cuda())
    print ("Baking features: ", b, ind.shape[0])

os.makedirs(name=root_path + "/texture_{}".format(args.texture_size), exist_ok=True)
compressor.save_to_file(root_path + "/texture_{}/".format(args.texture_size))
