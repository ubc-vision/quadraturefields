"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence
import trimesh

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import default_collate as collate
from torch.utils.data._utils.collate import default_collate_fn_map
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import accumulate_along_rays_, render_weight_from_density, rendering
from field_rendering import rendering_field
import kaolin.render.spc as spc_render
from radiance_fields.ngp import inverse_contraction

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compress_sigma(sigma):
    alpha = (1 - torch.exp(-sigma * 0.005))
    alpha = torch.clip(alpha * 255, 0, 255)
    alpha = alpha.to(torch.uint8)
    return alpha

def inverse_of_compressed_sigma(alpha):
    alpha = alpha.to(torch.float32) / 255.0
    alpha = -torch.log(1 - alpha) / 0.005
    return alpha

def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
        use_eps_loss: bool = False,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = origins[ray_indices]
        t_dirs = viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = origins[ray_indices]
        t_dirs = viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        origins = chunk_rays.origins.cuda()
        viewdirs = chunk_rays.viewdirs.cuda()
        ray_indices, t_starts, t_ends = estimator.sampling(
            origins,
            viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        extras["t_starts"] = t_starts
        extras["t_ends"] = t_ends
        extras["ray_indices"] = ray_indices
        extras["t_origins"] = chunk_rays.origins
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        extras
    )


@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape
    positions_all = []
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = (
            t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        )
        positions_all.append(positions)
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:
        torch.cuda.empty_cache()
        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth #/ opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        total_samples,
        torch.cat(positions_all, dim=0),
    )


def render_image_field_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            early_stop_eps=1e-4# so that more points are supervised
        )
        rgb, opacity, depth, weights, weights_rev = rendering_field(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        dirs = chunk_rays.viewdirs[ray_indices]

        chunk_results = [rgb, opacity, depth, len(t_starts), weights, weights_rev, positions, dirs]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples, weights, weights_rev, positions, dirs = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]    
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        weights_rev,
        positions,
        dirs
    )


def render_image_finetune_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    field_net: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    data,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    mesh_intersect=None,
    mesh_finetune=None,
    scaling=1/128,
    bg_color="white",
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(positions, ray_indices):
        # t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs.cuda()[ray_indices]
        # positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if True
        else test_chunk_size
    )
    batch_size = 160000
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        xyzs, dirs, index_ray, ts, index_tri, origins = data

        xyzs = xyzs.cuda()
        vertices = mesh_intersect.vertices[torch.from_numpy(np.array(mesh_intersect.mesh.faces[index_tri.contiguous().numpy()])).long()]
        vertices = vertices[:, :, 0:3]
        weights = torch.rand((xyzs.shape[0], 3), device=xyzs.device)[..., None]
        vertices = torch.sum(vertices * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6)

        dirs = dirs.cuda()
        ts = ts.cuda()
        index_ray = index_ray.cuda().long()
        index_tri = index_tri.cuda().long()
        origins = origins.cuda()

        # batchify the above code to avoid OOM
        del_vector_v = torch.zeros_like(vertices)
        for b in range(0, vertices.shape[0], batch_size):
            del_vector_v[b:b+batch_size] = field_net(vertices[b:b+batch_size], return_grad=False)[0]
        del_vector_v = torch.tanh(del_vector_v) * scaling


        # del_vector = field_net(xyzs, return_grad=False)[0]
        # batchify the above code to avoid OOM
        del_vector = torch.zeros_like(xyzs)
        for b in range(0, xyzs.shape[0], batch_size):
            del_vector[b:b+batch_size] = field_net(xyzs[b:b+batch_size], return_grad=False)[0]
        del_vector = torch.tanh(del_vector) * scaling
        del_delta = (del_vector * dirs).sum(-1, keepdim=True)

        dh = del_delta * dirs
        xyzs = xyzs + dh
        ts = ts + del_delta.view(-1)

        # Re-sorting the quadrature points based on added dhs
        points, deltas, boundary, dirs, index_ray, depth, _, _ = mesh_intersect.sampling_indexing(xyzs, origins,
                                                                                                      dirs, index_ray,
                                                                                                      ts, index_tri)

        rgbs = torch.zeros_like(points)
        sigmas = torch.zeros(points.shape[0], device=points.device)
        for b in range(0, points.shape[0], batch_size):
            rgbs[b:b+batch_size], sigmas[b:b+batch_size] = rgb_sigma_fn(points[b:b+batch_size], index_ray[b:b+batch_size])

        loss = ((del_vector) ** 2).mean() + ((del_vector_v - del_vector.detach()) ** 2).mean()
        loss = loss.reshape(1)
        rgb, opacity, _, depth, weights = derive_properties(rgbs, sigmas, depth, deltas, boundary, index_ray, bg_color=bg_color, render_bkgd=render_bkgd, N=chunk_rays.origins.shape[0])

        if mesh_finetune is not None:
            mesh_finetune.update_d(dh, weights[:, 0], index_tri)

        chunk_results = [rgb, opacity, depth, xyzs.shape[0], weights, points, rays, loss, index_ray, index_tri]
        results.append(chunk_results)

    colors, opacities, depths, n_rendering_samples, weights, positions, rays, loss, index_ray, index_tri = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        positions,
        index_ray,
        loss,
        index_tri
    )


def render_image_fit_sg_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    radiance_field_sg: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    data,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    mesh_intersect=None,
    mesh_finetune=None,
    scaling=1/128,
    bg_color="white",
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(positions, ray_indices):
        # t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs.cuda()[ray_indices]
        # positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if True
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        xyzs, dirs, index_ray, ts, index_tri, origins = data
        
        xyzs = xyzs.cuda()
        dirs = dirs.cuda()
        ts = ts.cuda()
        index_ray = index_ray.cuda().long()
        t_dirs = chunk_rays.viewdirs.cuda()[index_ray]

        origins = origins.cuda()
        rgbs = []
        # sigmas_sg = []
        for b in range(0, xyzs.shape[0], 32768):
            rgb, _ = radiance_field_sg(xyzs[b:b+32768], t_dirs[b:b+32768])
            rgbs.append(rgb)
        #     sigmas_sg.append(sigma)
        # sigmas_sg = torch.cat(sigmas_sg, dim=0)
        
        with torch.no_grad():
            sigmas = []
            for b in range(0, xyzs.shape[0], 32768):
                _, sigma = rgb_sigma_fn(xyzs[b:b+32768], index_ray[b:b+32768])
                sigmas.append(sigma)
            sigmas = torch.cat(sigmas, dim=0)

        rgbs = torch.cat(rgbs, dim=0)
        boundary = spc_render.mark_pack_boundaries(index_ray)
        deltas = torch.ones_like(sigmas) * render_step_size
        rgb, opacity, _, depth, weights = derive_properties(rgbs, sigmas, ts, deltas, boundary, index_ray, 
        bg_color=bg_color, render_bkgd=render_bkgd, N=chunk_rays.origins.shape[0])

        chunk_results = [rgb, opacity, depth, xyzs.shape[0], weights, xyzs, rays, index_ray, index_tri]
        results.append(chunk_results)

    colors, opacities, depths, n_rendering_samples, weights, xyzs, rays, index_ray, index_tri = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        xyzs,
        index_ray,
        index_tri
    )

@torch.no_grad()
def render_image_finetune_baking_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
        data,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    mesh_intersect=None,
        mesh_finetune=None,
        scaling=1/128
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(positions, ray_indices):
        # t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs.cuda()[ray_indices]
        # positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if True
        else test_chunk_size
    )
    batch_size = 1000000
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        # xyzs, dirs, index_ray, ts, index_tri, _, origins = mesh_intersect.sampling_raytrace_numpy(
        #     chunk_rays.viewdirs.data.cpu().numpy(), \
        #     chunk_rays.origins.data.cpu().numpy(), \
        #     0)
        xyzs, dirs, index_ray, ts, index_tri, origins = data

        xyzs = xyzs.cuda()
        dirs = dirs.cuda()
        ts = ts.cuda()
        index_ray = index_ray.cuda().long()
        index_tri = index_tri.cuda().long()
        origins = origins.cuda()

        points, deltas, boundary, dirs, index_ray, depth, _, _ = mesh_intersect.sampling_indexing(xyzs, origins,
                                                                                                      dirs, index_ray,
                                                                                                      ts, index_tri)
        b_coords = trimesh.triangles.points_to_barycentric(mesh_intersect.mesh.vertices[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]], points.data.cpu().numpy())
        b_coords = torch.from_numpy(b_coords.astype(np.float32)).cuda().unsqueeze(2)
        points = mesh_intersect.vertices[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]]
        points = points.reshape(-1, 3)
        # index_ray = torch.repeat_interleave(index_ray, 3, dim=0)

        rgbs = torch.zeros(points.shape[0], device=points.device)
        features = torch.zeros((points.shape[0], radiance_field.mlp_head.output_dim + 1), device=points.device)
        
        for b in range(0, points.shape[0], batch_size):
            feat = radiance_field.features(points[b:b+batch_size])
            features[b:b+batch_size] = feat

        features = features.reshape(-1, 3, radiance_field.mlp_head.output_dim + 1)
        features = torch.sum(features * b_coords, dim=1)
        # index_ray = index_ray[0:-1:3]
        sigmas = features[:, -1]
        features = features[:, 0:-1]
        rgbs = radiance_field.features_to_rgb(features, chunk_rays.viewdirs.cuda()[index_ray])

        rgb, opacity, _, depth, weights = derive_properties(rgbs, sigmas, depth, deltas, boundary, index_ray, bg_color="white", N=chunk_rays.origins.shape[0])


        chunk_results = [rgb, opacity, depth, xyzs.shape[0], weights, points, rays]
        results.append(chunk_results)

    colors, opacities, depths, n_rendering_samples, weights, positions, rays = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        positions,
        rays,
        0,
    )

def derive_properties(color, density, depths, deltas, boundary, index_ray, render_bkgd=None, bg_color="white", N=0):
    color = color.reshape(-1, 3).contiguous()
    # boundary = boundary[..., None]
    # TODO Batchify this whole function
    tau = density * deltas
    tau = tau.reshape((-1, 1))
    ray_colors, transmittance = spc_render.exponential_integration(color,
                                                                   tau,
                                                                    boundary,
                                                                    exclusive=True)


    depths, _ = spc_render.exponential_integration(depths.reshape((-1, 1)),
                                                                tau,
                                                                boundary,
                                                                exclusive=True)
    alpha = spc_render.sum_reduce(transmittance, boundary)
    out_alpha = torch.zeros(N, 1, device=alpha.device, dtype=torch.float)
    # Populate the background, this is used to model the background.

    Depth = torch.zeros((N, 1), device=color.device, dtype=torch.float)

    if bg_color == 'white':
        rgb = torch.ones(N, 3, device=color.device, dtype=torch.float)
        color = (1.0 - alpha) + alpha * ray_colors
    elif bg_color == 'black':
        rgb = torch.zeros(N, 3, device=color.device, dtype=torch.float)
        color = alpha * ray_colors
    else: # background color is random
        rgb = torch.ones(N, 3, device=color.device, dtype=torch.float)
        color = alpha * ray_colors + (1.0 - alpha) * render_bkgd
    
    Depth[index_ray[boundary]] = depths.float()
    rgb[index_ray[boundary]] = color.float().clone()
    out_alpha[index_ray[boundary]] = alpha.float()
    return rgb, out_alpha, index_ray[boundary], Depth, transmittance

@torch.no_grad()
def render_image_bake_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    rays: Rays,
        data,
    # rendering options
    texture=None,
    uv=None,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    mesh_intersect=None,
    mesh_finetune=None,
    scaling=1/128,
    discretize=False
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if True
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        xyzs, dirs, index_ray, ts, index_tri, origins = data

        xyzs = xyzs.cuda()
        dirs = dirs.cuda()
        ts = ts.cuda()
        index_ray = index_ray.cuda().long()
        index_tri = index_tri.cuda().long()
        origins = origins.cuda()

        points, deltas, boundary, dirs, index_ray, depth, _, _ = mesh_intersect.sampling_indexing(xyzs, origins,
                                                                                                      dirs, index_ray,
                                                                                                      ts, index_tri)
        uv_ = uv[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]]

        b_coords = trimesh.triangles.points_to_barycentric(mesh_intersect.mesh.vertices[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]], points.data.cpu().numpy())
        b_coords = torch.from_numpy(b_coords.astype(np.float32)).cuda()
        b_coords = torch.clamp(b_coords, 0, 1)

        b_coords = b_coords / (b_coords.sum(-1, keepdim=True))
        uv_points = torch.sum(uv_ * b_coords[..., None], 1)
        uv_points = torch.clip(torch.floor(uv_points).long(), 0, texture.shape[0] - 1)
        # uv_points = uv_points + 1 / (2 * texture.shape[0])
        # uv_points = torch.clip(uv_points, 0, texture.shape[0] - 1)
        try:
            texture_points = texture[uv_points[:, 0].data.cpu().numpy(), uv_points[:, 1].data.cpu().numpy()]
            # texture_points = bilinear_interpolate_numpy(texture, uv_points[:, 1].data.cpu().numpy(), uv_points[:, 0].data.cpu().numpy())
        except:
            import ipdb; ipdb.set_trace()
        texture_points = torch.from_numpy(texture_points.astype(np.float32)).cuda()
        if discretize:
            sigmas = inverse_of_compressed_sigma(compress_sigma(texture_points[:, -1]))
        else:
            sigmas = texture_points[:, -1]
        features = texture_points[:, :-1]
        rgbs = radiance_field.features_to_rgb(features, dirs)
        rgb, opacity, _, depth, weights = derive_properties(rgbs, sigmas, depth, deltas, boundary, index_ray, bg_color="white", render_bkgd=None, N=chunk_rays.origins.shape[0])

        chunk_results = [rgb, opacity, depth, xyzs.shape[0], weights, points, rays]
        results.append(chunk_results)

    colors, opacities, depths, n_rendering_samples, weights, positions, rays = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        positions,
        rays,
        0,
    )

@torch.no_grad()
def render_image_bake_texture_images_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    rays: Rays,
        data,
    # rendering options
    texture=None,
    uv=None,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    mesh_intersect=None,
    mesh_finetune=None,
    scaling=1/128,
    discretize=False,
    compressor=None,
    bg_color="white"
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if True
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        xyzs, dirs, index_ray, ts, index_tri, origins = data

        xyzs = xyzs.cuda()
        dirs = dirs.cuda()
        ts = ts.cuda()
        index_ray = index_ray.cuda().long()
        index_tri = index_tri.cuda().long()
        origins = origins.cuda()

        points, deltas, boundary, dirs, index_ray, depth, _, _ = mesh_intersect.sampling_indexing(xyzs, origins,
                                                                                                      dirs, index_ray,
                                                                                                      ts, index_tri)
        uv_ = uv[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]]

        b_coords = trimesh.triangles.points_to_barycentric(mesh_intersect.mesh.vertices[mesh_intersect.mesh.faces[index_tri.data.cpu().numpy()]], points.data.cpu().numpy())
        b_coords = torch.from_numpy(b_coords.astype(np.float32)).cuda()
        b_coords = torch.clamp(b_coords, 0, 1)

        b_coords = b_coords / (b_coords.sum(-1, keepdim=True))
        uv_points = torch.sum(uv_ * b_coords[..., None], 1)
        uv_points = torch.clip(torch.floor(uv_points).long(), 0, compressor.texture_size - 1)
        texture_points = []
        for i in range(0, uv_points.shape[0], 32000):
            torch.cuda.empty_cache()
            texture_points.append(compressor.get_features_from_texture_map(uv_points[i:i+32000]))
        texture_points = torch.cat(texture_points, dim=0)
        torch.cuda.empty_cache()
        if discretize:
            sigmas = inverse_of_compressed_sigma(compress_sigma(texture_points[:, -1]))
            torch.cuda.empty_cache()
        else:
            sigmas = texture_points[:, -1]
        features = texture_points[:, :-1]
        rgbs = radiance_field.features_to_rgb(features, dirs)
        rgb, opacity, _, depth, weights = derive_properties(rgbs, sigmas, depth, deltas, boundary, index_ray, bg_color=bg_color, render_bkgd=None, N=chunk_rays.origins.shape[0])

        chunk_results = [rgb, opacity, depth, xyzs.shape[0], weights, points, rays]
        results.append(chunk_results)

    colors, opacities, depths, n_rendering_samples, weights, positions, rays = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        weights,
        positions,
        rays,
        0,
    )
