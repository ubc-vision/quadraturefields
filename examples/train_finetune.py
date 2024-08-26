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
from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss
import json
from shutil import ignore_patterns
import shutil


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
parser.add_argument(
    "--max_iterations",
    type=int,
    default=24000,
    help="Maximum number of iterations used for training",
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
    default=1e-4,
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
    "--c_lambda",
    type=float,
    default=1e-5,
    help="scale factor for the opacity loss",
)
parser.add_argument(
    "--agg",
    type=float,
    default=2,
    help="scale factor for the opacity loss",
)
parser.add_argument(
    "--reg_type",
    type=str,
    default="none",
    help="which train split to use",
)

parser.add_argument(
    "--continuous_loss",
    type=str2bool,
    default="True",
    help="whether to use soft rgb loss",
)
# log2_hashmap_size
parser.add_argument(
    "--log2_hashmap_size",
    type=int,
    default=19,
    help="log2 of the hashmap size",
)
# batch size
parser.add_argument(
    "--batch_size",
    type=int,
    default=18,
    help="batch size",
)
parser.add_argument(
    "--occ_thres",
    type=float,
    default=0.01,
    help="threshold for pruning grid",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.5,
    help="scale of the scene in which to perform volumetric rendering",
)
parser.add_argument(
    "--grad_accum",
    type=int,
    default=1,
    help="number of gradient accumulation steps",
)

args = parser.parse_args()
args.optix = str2bool(args.optix)

if args.optix:
    from build.lib import intersector

device = "cuda:0"
set_random_seed(42)
os.makedirs(args.root + "ckpts/{}/{}/".format(args.scene, args.exp_name), exist_ok=True)
os.makedirs(args.root + "results/{}/{}/".format(args.scene, args.exp_name), exist_ok=True)
os.makedirs(args.root + "logs/{}/{}/".format(args.scene, args.exp_name), exist_ok=True)

# setup tensorboard logger to log values of the loss function
logger = SummaryWriter(args.root + "logs/{}/{}".format(args.scene, args.exp_name), flush_secs=200)
with open(args.root + "results/{}/{}/log.txt".format(args.scene, args.exp_name), "w") as f:
    pass

with open(args.root + "results/{}/{}/args.json".format(args.scene, args.exp_name), "w") as f:
    json.dump(args.__dict__, f, indent=2)

shutil.copytree(
    os.getcwd() + "/examples/",
    args.root + "results/{}/{}/code".format(args.scene, args.exp_name),
    ignore=ignore_patterns("*.pyc", "tmp*", "__pycache__", "build", "pycolmap"),
    dirs_exist_ok=True,
)

mesh_intersect = MeshIntersection(
    mesh_path=args.mesh_path,
    scale=args.scale,
    simplify_mesh=False,
    optix=args.optix,
    voxel_size=args.voxel_size,
    num_intersections=args.max_hits,
)

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
    max_steps = args.max_iterations
    init_batch_size = 1024
    target_sample_batch_size = 1 << args.batch_size
    weight_decay = 1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    # scene parameters
    aabb = torch.tensor([-1., -1., -1., 1., 1., 1.], device=device) * args.scale
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


def initialize_dataloaders():
    train_dataset_ = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=target_sample_batch_size,
        device=torch.device("cpu"),
        **train_dataset_kwargs,
        mesh_intersect=mesh_intersect,
        upsample=2,
    )

    train_dataset = iter(
        DataLoader(
            train_dataset_,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1 if args.optix else 1,
            pin_memory=True,
            collate_fn=collate_function,
        )
    )
    train_whole_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="whole",
        num_rays=target_sample_batch_size,
        device=torch.device("cpu"),
        **train_dataset_kwargs,
        mesh_intersect=mesh_intersect,
        upsample=2,
    )

    train_whole_dataset = iter(
        DataLoader(
            train_whole_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1 if args.optix else 1,
            pin_memory=False,
            collate_fn=collate_function,
        )
    )

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

    test_dataset = iter(
        DataLoader(
            test_dataset_,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0 if args.optix else 0,
            pin_memory=False,
            collate_fn=collate_function,
        )
    )
    return train_dataset, test_dataset, train_whole_dataset


train_dataset, test_dataset, train_whole_dataset = initialize_dataloaders()

estimator = OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

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
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64,log2_hashmap_size=args.log2_hashmap_size).to(
        device
    )

# Defines the deformation field
field_net = Field(
    scale=args.scale,
    precision=16,
    log2_T=24,
    L=16,
    max_res=512,
    min_res=16,
    output_dim=1,
    hidden_size=32,
    num_features=2,
    back_prop=False,
    nl="relu",
).cuda()
mesh_finetune = MeshFinetune(mesh_intersect.mesh.vertices, mesh_intersect.mesh.faces, args.scaling)

# define the parameter groups for the optimizer including Field NGPRadianceField
param_groups = [
    {"params": field_net.parameters(), "lr": 2e-2},
    {"params": radiance_field.parameters(), "lr": 2e-2},
]
ckpt = torch.load(args.ckpt_path)
radiance_field.load_state_dict(ckpt["model"])
estimator.load_state_dict(ckpt["estimator"])


binaries = estimator.binaries
np.save(
    args.root + "results/{}/{}/binaries.npy".format(args.scene, args.exp_name),
    binaries.cpu().numpy(),
)


def generate_splits(data, num_rays):
    xyzs, dirs, index_ray, ts, index_tri, origins = data
    chunks = []
    chunk_size = 160000
    for i in range(0, num_rays, chunk_size):
        mask1 = index_ray < i + chunk_size
        mask2 = index_ray >= i
        mask = mask1 * mask2
        if mask.sum() == 0:
            continue
        chunks.append(
            (
                xyzs[mask].contiguous(),
                dirs[mask].contiguous(),
                index_ray[mask].contiguous(),
                ts[mask].contiguous(),
                index_tri[mask].contiguous(),
                origins[mask].contiguous(),
            )
        )
    return chunks


optimizer = torch.optim.Adam(param_groups, lr=1e-2, eps=1e-15)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=1000),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 4,
                max_steps * 2,
                max_steps * 6 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
test_ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
results = {}

# training
tic = time.time()
for step in range(max_steps + 1):
    torch.cuda.empty_cache()
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = next(train_dataset)  # [i]

    render_bkgd = data["color_bkgd"].cuda()
    rays = data["rays"]
    pixels = data["pixels"].cuda()

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=args.occ_thres,
    )
    if step < 300:
        for p in radiance_field.parameters():
            p.requires_grad = False
    else:
        for p in radiance_field.parameters():
            p.requires_grad = True

    rgb, acc, depth, n_rendering_samples, weights, positions, index_ray, loss_reg, _ = (
        render_image_finetune_with_occgrid(
            radiance_field,
            field_net,
            estimator,
            rays,
            data["data"],
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            mesh_intersect=mesh_intersect,
            scaling=args.scaling,
        )
    )
    acc = acc.squeeze()

    num_above_thres = (weights > 1e-3).sum().item()
    rgb_full, _, _, _, _ = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )

    rgb_discrete_loss = F.smooth_l1_loss(rgb.squeeze(), pixels)
    rgb_smooth_loss = F.smooth_l1_loss(rgb_full.squeeze(), pixels)
    rgb_loss = (rgb_discrete_loss + rgb_smooth_loss) / 2
    loss = rgb_loss + loss_reg
    optimizer.zero_grad()

    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        elapsed_time = time.time() - tic
        rgb_loss = F.mse_loss(rgb.squeeze(), pixels)
        psnr = -10.0 * torch.log(rgb_loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss.item():.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | ",
            "avghits={:.3f}".format(weights.shape[0] / len(pixels)),
            "avghits_thres={:.3f}".format(num_above_thres / weights.shape[0])
                                
        )
        logger.add_scalar("train/avghits", weights.shape[0] / len(pixels), step)
        logger.add_scalar("train/avghits_thres", num_above_thres / weights.shape[0], step)
        logger.add_scalar("train/loss", loss.item(), step)
        logger.add_scalar("train/rgb_smooth_loss", rgb_smooth_loss.item(), step)
        logger.add_scalar("train/rgb_discrete_loss", rgb_discrete_loss.item(), step)
        logger.add_scalar("train/loss_reg", loss_reg.mean().item(), step)
    del loss, loss_reg, rgb, acc, depth, n_rendering_samples, weights, positions, rays, data
    if step % 1000 == 0 and step > 0:
        binaries = estimator.binaries
        np.save(
            args.root + "results/{}/{}/binaries.npy".format(args.scene, args.exp_name),
            binaries.cpu().numpy(),
        )

        path = args.root + "ckpts/{}/{}/model.pth".format(args.scene, args.exp_name)

        save_dict = {
            "estimator": estimator.state_dict(),
            "field_model": field_net.state_dict(),
            "radiance_field": radiance_field.state_dict(),
        }

        torch.save(save_dict, path)
        print("Saved checkpoints at", path)

    if step > 0 and step % 2000 == 0:
        # evaluation
        @torch.no_grad()
        def test(new_scaling, evaluation_dataset, prefix=""):
            torch.cuda.empty_cache()
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            lpips = []
            ssims = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(evaluation_dataset._dataset.images))):
                    torch.cuda.empty_cache()
                    data = next(evaluation_dataset)  # [i]
                    render_bkgd = data["color_bkgd"].cuda()
                    rays = data["rays"]
                    pixels = data["pixels"].cuda()
                    splits = generate_splits(data["data"], rays.origins.shape[0])

                    # Assuming white background
                    rgb = torch.ones((rays.origins.shape[0], 3), device=device)
                    depth = torch.zeros((rays.origins.shape[0]), device=device)
                    n_rendering_samples = 0
                    for split in splits:
                        color, _, d, n_r_s, _, _, _, _, _ = (
                            render_image_finetune_with_occgrid(
                                radiance_field,
                                field_net,
                                estimator,
                                rays,
                                split,
                                # rendering options
                                near_plane=near_plane,
                                render_step_size=render_step_size,
                                render_bkgd=render_bkgd,
                                cone_angle=cone_angle,
                                alpha_thre=alpha_thre,
                                mesh_intersect=mesh_intersect,
                                mesh_finetune=mesh_finetune,
                                scaling=new_scaling,
                            )
                        )
                        rgb[split[2]] = color[split[2]]
                        depth[split[2]] = d.squeeze()[split[2]]
                        n_rendering_samples += n_r_s
                    
                    torch.cuda.empty_cache()
                    rgb = rgb.reshape(int(evaluation_dataset._dataset.HEIGHT), int(evaluation_dataset._dataset.WIDTH), 3)
                    pixels = pixels.reshape(int(evaluation_dataset._dataset.HEIGHT // args.up_sample) , int(evaluation_dataset._dataset.WIDTH // args.up_sample), 3)
                    depth = depth.reshape(int(evaluation_dataset._dataset.HEIGHT), int(evaluation_dataset._dataset.WIDTH))
                    
                    rgb = cv2.resize(rgb.cpu().numpy(), (int(evaluation_dataset._dataset.WIDTH // args.up_sample), int(evaluation_dataset._dataset.HEIGHT // args.up_sample)), interpolation=cv2.INTER_AREA)
                    depth = cv2.resize(
                        depth.cpu().numpy(), (int(evaluation_dataset._dataset.WIDTH // args.up_sample), int(evaluation_dataset._dataset.HEIGHT // args.up_sample)), interpolation=cv2.INTER_AREA
                    )
                    rgb = torch.from_numpy(rgb).cuda()
                    depth = torch.from_numpy(depth).cuda()

                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    test_ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0))
                    ssim = test_ssim.compute()
                    test_ssim.reset()
                    ssims.append(ssim.item())
                    psnrs.append(psnr.item())
                    lpips.append(lpips_fn(rgb, pixels).item())
                    rgb = torch.clamp(rgb, 0, 1)
                    error = torch.abs(rgb - pixels)
                    error = torch.clamp(error, 0, 1)
                    depth = depth / depth.max()
                    # Log images using tensorboard logger
                    error = (error.cpu().numpy() * 255).astype(np.uint8)
                    rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
                    depth = (depth.cpu().numpy() * 255).astype(np.uint8)
                    imageio.imwrite(
                        args.root
                        + "results/{}/{}/rgb_test_{}_{}.png".format(
                            args.scene, args.exp_name, prefix, i
                        ),
                        rgb,
                    )
                    imageio.imwrite(
                        args.root
                        + "results/{}/{}/rgb_error_{}_{}.png".format(
                            args.scene, args.exp_name, prefix, i
                        ),
                        error,
                    )
                    imageio.imwrite(
                        args.root
                        + "results/{}/{}/depth_{}_{}.png".format(
                            args.scene, args.exp_name, prefix, i
                        ),
                        depth,
                    )
                    if i == 0:
                        imageio.imwrite(
                            args.root
                            + "results/{}/{}/step_rgb_{}_{}_{}.png".format(
                                args.scene, args.exp_name, prefix, i, step
                            ),
                            rgb,
                        )
                        imageio.imwrite(
                            args.root
                            + "results/{}/{}/step_error_{}_{}_{}.png".format(
                                args.scene, args.exp_name, prefix, i, step
                            ),
                            error,
                        )
                        imageio.imwrite(
                            args.root
                            + "results/{}/{}/step_depth_{}_{}_{}.png".format(
                                args.scene, args.exp_name, prefix, i, step
                            ),
                            depth,
                        )
            psnr_avg = sum(psnrs) / len(psnrs)
            lpips_avg = sum(lpips) / len(lpips)
            ssim_avg = sum(ssims) / len(ssims)
            print(f"PSNR: {psnr_avg}, LPIPS: {lpips_avg}, SSIM: {ssim_avg}")
            return psnr_avg, lpips_avg, ssim_avg

        psnr_avg, lpips_avg, ssim_avg = test(args.scaling, train_whole_dataset, prefix="before")

        logger.add_scalar("before_psnr", psnr_avg, step)
        logger.add_scalar("before_lpips", lpips_avg, step)
        logger.add_scalar("before_ssim", ssim_avg, step)
        # Save the results in a log file
        with open(args.root + "results/{}/{}/log.txt".format(args.scene, args.exp_name), "a") as f:
            f.write(
                "step: {}, before_psnr: {}, before_lpips: {}, before_ssim: {}\n".format(
                    step, psnr_avg, lpips_avg, ssim_avg
                )
            )
        with torch.no_grad():
            mesh_finetune.update_faces()
            mesh_finetune.reset_d()
            mesh_intersect.mesh.vertices = mesh_finetune.vertices
            # This was a bug before. We are now updating it to the new vertices
            mesh_intersect.vertices = torch.from_numpy(mesh_finetune.vertices).cuda()
        if args.optix:
            print("Update mesh vertices in the optix intersector")
            mesh_intersect.rayintersector.inter = intersector.Intersector(
                mesh_finetune.vertices[mesh_intersect.mesh.faces].flatten(), args.max_hits, 0
            )
        else:
            print("Update mesh vertices in the intel intersector")
            train_dataset, test_dataset, train_whole_dataset = initialize_dataloaders()
            mesh_intersect.mesh.export(
            args.root + "results/{}/{}/mesh.ply".format(args.scene, args.exp_name),
        )

        psnr_avg_after, lpips_avg_after, ssim_avg_after = test(0, test_dataset, prefix="after")

        mesh_finetune.reset_d()
        torch.cuda.empty_cache()
        logger.add_scalar("after_psnr", psnr_avg_after, step)
        logger.add_scalar("after_lpips", lpips_avg_after, step)
        logger.add_scalar("after_ssim", ssim_avg_after, step)
        # Save the results in a log file
        results[step] = {
            "before_psnr": psnr_avg,
            "before_lpips": lpips_avg,
            "before_ssim": ssim_avg,
            "after_psnr": psnr_avg_after,
            "after_lpips": lpips_avg_after,
            "after_ssim": ssim_avg_after,
        }
        with open(args.root + "results/{}/{}/results.json".format(args.scene, args.exp_name), "w") as f:
            json.dump(results, f, indent=2)

        imgs = sorted(
            glob.glob(
                args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb_test*.png"
            )
        )
        imgs_depth = sorted(
            glob.glob(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "depth*.png")
        )
        imgs_error = sorted(
            glob.glob(
                args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb_error*.png"
            )
        )

        imageio.mimsave(
            args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb.mp4",
            [imageio.imread(img) for img in imgs],
            fps=20,
            macro_block_size=1,
        )
        imageio.mimsave(
            args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "depth.mp4",
            [imageio.imread(img) for img in imgs_depth],
            fps=20,
            macro_block_size=1,
        )
        imageio.mimsave(
            args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb_error.mp4",
            [imageio.imread(img) for img in imgs_error],
            fps=20,
            macro_block_size=1,
        )
        del psnr_avg, lpips_avg, ssim_avg, psnr_avg_after, lpips_avg_after, ssim_avg_after