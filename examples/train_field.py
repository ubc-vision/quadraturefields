import argparse
import pathlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField, NGPRadianceFieldSGNew
from field import Field
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_field_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
from field_utils import plot_field, extract_grid
from torch.utils.tensorboard import SummaryWriter
import os
import json
from shutil import ignore_patterns
import shutil
from field_utils import extract_density_grid


torch.multiprocessing.set_start_method("spawn")
parser = argparse.ArgumentParser()
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
    "--root",
    type=str,
    default="/arc/burst/pr-kmyi-1/gopalshr/nerfacc/",
    help="the root",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="ngp",
    help="experiment name",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    help="which scene to use",
)
# parser for number of lobes
parser.add_argument(
    "--num_lobes",
    type=int,
    default=2,
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
    default=2,
    help="number of layers",
)
parser.add_argument(
    "--out_activation",
    type=str,
    default="sigmoid",
    help="output activation function",
)
# checkpoint path
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="path to the checkpoint",
)
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
    "--scale",
    type=float,
    default=1.5,
    help="scaling factor of the grid",
)
parser.add_argument(
    "--occ_thres",
    type=float,
    default=0.01,
    help="threshold for pruning grid",
)

parser.add_argument(
    "--max_steps", type=int, default=20000, help="max number of steps"
)
args = parser.parse_args()

os.makedirs(
    args.root + "ckpts/{}/{}/".format(args.scene, args.exp_name), exist_ok=True
)
os.makedirs(
    args.root + "results/{}/{}/".format(args.scene, args.exp_name), exist_ok=True
)
with open(
    args.root + "ckpts/{}/{}/args.json".format(args.scene, args.exp_name), "w"
) as f:
    json.dump(vars(args), f, indent=4)
with open(
    args.root + "results/{}/{}/args.json".format(args.scene, args.exp_name), "w"
) as f:
    json.dump(args.__dict__, f, indent=2)
# Back up the current directory with all its python file to the results folder

shutil.copytree(
    os.getcwd() + "/examples/",
    args.root + "results/{}/{}/code".format(args.scene, args.exp_name),
    ignore=ignore_patterns("*.pyc", "tmp*", "__pycache__", "build", "pycolmap"),
    dirs_exist_ok=True,
)

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = args.max_steps
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
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
    max_steps = args.max_steps
    init_batch_size = 1024
    target_sample_batch_size = 1 << args.batch_size
    weight_decay = 1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    # scene parameters
    aabb = (
        torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device) * args.scale
    )
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

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)
logger = SummaryWriter(args.root + "logs/{}/{}".format(args.scene, args.exp_name))

grad_scaler = torch.cuda.amp.GradScaler(2**10)
if args.num_lobes > 0:
    radiance_field = NGPRadianceFieldSGNew(
        aabb=estimator.aabbs[-1],
        use_viewdirs=False,
        num_g_lobes=args.num_lobes,
        log2_hashmap_size=args.log2_hashmap_size,
        num_layers=args.num_layers,
    ).to(device)
else:
    radiance_field = NGPRadianceField(
        aabb=estimator.aabbs[-1],
        num_layers=2,
        log2_hashmap_size=args.log2_hashmap_size,
    ).to(device)

field_net = Field(
    scale=0.5,
    precision=16,
    log2_T=30,
    L=16,
    max_res=512,
    min_res=16,
    output_dim=1,
    hidden_size=16,
    num_features=2,
    back_prop=False,
    nl="elu",
    bias=True,
    bias_last=True,
).cuda()

# define the parameter groups for the optimizer including Field NGPRadianceField
param_groups = [
    {"params": field_net.parameters(), "lr": 2e-2, "weight_decay": weight_decay}
]
ckpt = torch.load(args.ckpt_path)
radiance_field.load_state_dict(ckpt["model"])
estimator.load_state_dict(ckpt["estimator"])

binaries = estimator.binaries
np.save(
    args.root + "results/{}/{}/binaries.npy".format(args.scene, args.exp_name),
    binaries.cpu().numpy(),
)

extract_density_grid(
    radiance_field,
    scale=args.scale,
    prefix=args.root + "results/{}/{}/".format(args.scene, args.exp_name),
    grid_size=1024,
)

optimizer = torch.optim.Adam(param_groups, lr=2e-3, eps=1e-15)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=args.occ_thres,
    )

    # render
    with torch.no_grad():
        (
            rgb,
            acc,
            depth,
            n_rendering_samples,
            weights,
            weights_rev,
            positions,
            dirs,
        ) = render_image_field_with_occgrid(
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
        torch.cuda.empty_cache()
        _, positions = radiance_field.normalize(positions)
    # normalization
    positions = positions - 0.5
    positions = positions.detach()
    positions.requires_grad = True
    field, field_grad = field_net(positions)
    field_loss = field_net.compute_field_loss(
        weights, weights_rev=weights_rev, field_norm=field_grad, view_dirs=dirs
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    rgb_loss = F.smooth_l1_loss(rgb, pixels)
    loss = field_loss

    optimizer.zero_grad()
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()
    logger.add_scalar("loss", loss, step)
    logger.add_scalar("field_loss", field_loss, step)

    if step % 1000 == 0:
        torch.cuda.empty_cache()
        estimator.eval()
        binaries = estimator.binaries
        plot_field(
            field_net,
            args.root + "results/{}/{}/".format(args.scene, args.exp_name),
            binaries,
            scale=0.5,
        )

        elapsed_time = time.time() - tic
        rgb_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(rgb_loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"floss={field_loss:.5f} "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % args.max_steps == 0:
        torch.cuda.empty_cache()
        estimator.eval()
        extract_grid(
            field_net,
            args.root + "results/{}/{}/".format(args.scene, args.exp_name),
            scale=0.5,
        )
        binaries = estimator.binaries
        np.save(
            args.root
            + "results/{}/{}/binaries.npy".format(args.scene, args.exp_name),
            binaries.cpu().numpy(),
        )

        path = args.root + "ckpts/{}/{}/model.pth".format(args.scene, args.exp_name)

        save_dict = {
            "estimator": estimator.state_dict(),
            "model": field_net.state_dict(),
        }

        torch.save(save_dict, path)
        print("Saved checkpoints at", path)
