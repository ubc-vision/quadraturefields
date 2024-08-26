import argparse
import pathlib
import time
import json
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField, NGPRadianceFieldSGNew
from torch.utils.tensorboard import SummaryWriter
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
from matplotlib import pyplot as plt
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
import os
from torch_efficient_distloss import flatten_eff_distloss
import shutil
from shutil import ignore_patterns


cmap = plt.cm.jet
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
    "--reg_type",
    type=str,
    default="occ",
    help="which train split to use",
)
parser.add_argument(
    "--occ_thres",
    type=float,
    help="threshold used for pruning datastructures",
    default=0.01
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
    # choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
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
parser.add_argument(
    "--c_lambda",
    type=float,
    default=1e-5,
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
# define scale
parser.add_argument(
    "--scale",
    type=float,
    default=1.5,
    help="batch size",
)
import glob

parser.add_argument("--max_steps", type=int, default=20000, help="max number of steps")
args = parser.parse_args()

os.makedirs(args.root + "results/{}/".format(args.scene), exist_ok=True)
os.makedirs(args.root + "results/{}/{}/".format(args.scene, args.exp_name), exist_ok=True)
logger = SummaryWriter(args.root + "logs/{}/{}".format(args.scene, args.exp_name), flush_secs=200)
with open(args.root + "results/{}/{}/log.json".format(args.scene, args.exp_name), "w") as f: pass


with open(args.root + "results/{}/{}/args.json".format(args.scene, args.exp_name), "w") as f:
    json.dump(args.__dict__, f, indent=2)

shutil.copytree(os.getcwd() + "/examples/", args.root + "results/{}/{}/code".format(args.scene, args.exp_name), ignore=ignore_patterns('*.pyc', 'tmp*', "__pycache__", "build", "pycolmap"), dirs_exist_ok=True)

os.makedirs(args.root + 'ckpts/{}/'.format(args.scene), exist_ok=True)
os.makedirs(args.root + 'ckpts/{}/{}/'.format(args.scene, args.exp_name), exist_ok=True)
path = args.root + 'ckpts/{}/{}/{}'.format(args.scene, args.exp_name, 'ngp.pth')

device = "cuda:0"
set_random_seed(42)
test_ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = args.max_steps
    init_batch_size = 10
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e2
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
    unbounded = True
else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = args.max_steps
    init_batch_size = 4096
    target_sample_batch_size = 1 << args.batch_size
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1, -1, -1, 1, 1, 1.0], device=device) * args.scale
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
    unbounded = False
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

grad_scaler = torch.cuda.amp.GradScaler(2**10)
if args.num_lobes > 0:
    radiance_field = NGPRadianceFieldSGNew(aabb=estimator.aabbs[-1],
                                        use_viewdirs=False,
                                        num_g_lobes=args.num_lobes,
                                        log2_hashmap_size=args.log2_hashmap_size,
                                        num_layers=args.num_layers).to(device)
else:
    if unbounded:
        radiance_field = NGPRadianceField(aabb=estimator.aabbs[0], num_layers=2, hidden_size=64, unbounded=unbounded).to(device)
    else:
        radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64, log2_hashmap_size=args.log2_hashmap_size,
                                          unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay)

o_lambda = args.o_lambda


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

# training
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
        occ_thre=args.occ_thres
    )

    rgb, acc, depth, n_rendering_samples, extras = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0 and step > 100:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    rgb_loss = F.smooth_l1_loss(rgb, pixels)
    # import ipdb; ipdb.set_trace()
    acc = acc.squeeze()
    if args.reg_type == "occ":
        loss_reg = (o_lambda*(-acc * torch.log(acc+1e-10))).mean()
    elif args.reg_type == "entropy":
        loss_reg = (o_lambda*(-extras["weights"] * torch.log(extras["weights"]+1e-7))).mean()
    elif args.reg_type == "cauchy":
        loss_reg = args.c_lambda * (torch.log(1 + extras["sigmas"] ** 2)).mean()
    elif args.reg_type == "both":
        loss_reg = (o_lambda*(-acc * torch.log(acc+1e-10))).mean() + args.c_lambda * (torch.log(1 + extras["sigmas"] ** 2)).mean()
    elif args.reg_type == "lol":
        loss_reg = (o_lambda * (torch.log(torch.exp(-extras["weights"]) + torch.exp(-torch.abs(1 - extras["weights"]))))).mean()
    elif args.reg_type == "none":
        loss_reg = torch.zeros(1, device=device).mean()
    elif args.reg_type == "distortion":
        index_ray = extras["ray_indices"].squeeze()
        positions = extras["t_origins"][index_ray] + rays.viewdirs[index_ray].cuda() * (extras["t_starts"] + extras["t_ends"])[..., None] / 2.0
        positions = positions.squeeze()
        weights = extras["weights"].squeeze()
        loss_reg = args.o_lambda * flatten_eff_distloss(weights, torch.abs((positions.cuda() * rays.viewdirs[index_ray].cuda()).sum(1)),
                                                        torch.ones(weights.shape[0], device=weights.device) * render_step_size,
                                                        index_ray)

    loss = rgb_loss + loss_reg
    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        logger.add_scalar("train/loss", loss.item(), step)
        logger.add_scalar("train/n_rendering_samples", n_rendering_samples / len(pixels), step)
        logger.add_scalar("train/loss_reg", loss_reg.item(), step)
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        logger.add_scalar("train/psnr", psnr.item(), step)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
    if step % 1000 == 0:
        save_dict = {
            "estimator": estimator.state_dict(),
            'model': radiance_field.state_dict(),
        }

        torch.save(save_dict, path)
        print('Saved checkpoints at', path)
    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        estimator.eval()
        torch.cuda.empty_cache()
        psnrs = []
        lpips = []
        ssims = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset.images))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                torch.cuda.empty_cache()

                rgb, acc, depth, n_rendering_samples, extras = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                            
                rgb = rgb.reshape(test_dataset.HEIGHT, test_dataset.WIDTH, 3)
                pixels = pixels.reshape(test_dataset.HEIGHT, test_dataset.WIDTH, -1)
                depth = depth.reshape(test_dataset.HEIGHT, test_dataset.WIDTH)
                depth = (depth - depth.min())
                depth = depth / depth.max()
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                test_ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0))
                ssim = test_ssim.compute()
                test_ssim.reset()
                ssims.append(ssim.item())

                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                print(psnr.item())
                if True:
                    imageio.imwrite(
                            args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb_test_{}.png".format(str(i).zfill(3)),
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                    imageio.imwrite(
                        args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "rgb_error_{}.png".format(str(i).zfill(3)),
                        (
                                (rgb - pixels).cpu().numpy() * 255
                        ).astype(np.uint8),
                    )
                    # save depth map
                    imageio.imwrite(
                        args.root + "results/{}/{}/".format(args.scene, args.exp_name) + "depth_test_{}.png".format(str(i).zfill(3)),
                        (depth.cpu().numpy() * 255).astype(np.uint8),
                    )

        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        ssim_avg = sum(ssims) / len(ssims)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, ssim_avg={ssim_avg}")
        logger.add_scalar("test/psnr", psnr_avg, step)
        logger.add_scalar("test/lpips", lpips_avg, step)
        logger.add_scalar("test/ssim", sum(ssims) / len(ssims), step)
        with open(args.root + "results/{}/{}/log.json".format(args.scene, args.exp_name), "a") as f:
            # Write psnr, lpips, ssim in json format
            json.dump(
                {
                    "step": step,
                    "psnr": psnr_avg,
                    "lpips": lpips_avg,
                    "ssim": ssim_avg,
                },
                f,
            )


imgs = sorted(glob.glob(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'rgb_test*.png'))
imgs_depth = sorted(glob.glob(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'depth*.png'))
imgs_error = sorted(glob.glob(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'rgb_error*.png'))

imageio.mimsave(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'rgb.mp4',
                [imageio.imread(img) for img in imgs],
                fps=20, macro_block_size=1)
imageio.mimsave(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'depth.mp4',
                [imageio.imread(img) for img in imgs_depth],
                fps=20, macro_block_size=1)
imageio.mimsave(args.root + "results/{}/{}/".format(args.scene, args.exp_name) + 'rgb_error.mp4',
                [imageio.imread(img) for img in imgs_error],
                fps=20, macro_block_size=1)
