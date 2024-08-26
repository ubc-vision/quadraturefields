import torch
try:
    import open3d as o3d
except:
    print("open3d not installed")
# import measure module from scipy
from skimage import measure
import numpy as np
import trimesh
import time
import torch.nn.functional as F
import tqdm
import numpy as np
import torch
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.ngp import NGPRadianceField
from field import Field
from mc_utils import *
from radiance_fields.ngp import contract_to_unisphere
import trimesh
import numpy as np
# %%
from mesh_utils import RayTrace

import torch
from field_utils import GaussianSmoothing
from shutil import ignore_patterns
from utils import (
    render_image_with_occgrid,
render_image_with_occgrid_test
)
from skimage import measure
import torch
import numpy as np


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = trimesh.Trimesh()  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        try:
            assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        except:
            import ipdb; ipdb.set_trace()
        mesh = scene_or_mesh
    return mesh

def concatenate_meshes(paths, simplify=False, agg=0, verbose=False):
    # meshes = [trimesh.load(p) for p in paths]
    new_meshes = []
    for p in paths:
        m = trimesh.load(p)
        if isinstance(m, list): continue
        m = as_mesh(m)
        if simplify:
            import fast_simplification
            print ("Before simplification: ", m)
            points, faces = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces, target_count=m.faces.shape[0] // 10, agg=agg, verbose=0)
            m = trimesh.Trimesh(points, faces)
            print ("After simplification: ", m)
        new_meshes.append(m)
    meshes = trimesh.util.concatenate(new_meshes)
    return meshes


@torch.no_grad()
def inverse_contraction(x: torch.Tensor, aabb: torch.Tensor):
    """
    Inverse contraction function for the contract_to_unisphere.
    Expects input in [0, 1] and returns in [-inf, inf]
    """
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - 0.5) * 4
    mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    invalid = mag.squeeze(-1) > 2
    x[mask] = 1 / (2 - mag[mask]) * x[mask] / mag[mask]
    x = (x + 1) / 2
    x = x * (aabb_max - aabb_min) + aabb_min
    return x, invalid


# Find values of density, grid and grads
def generate_raw_values(coords, field_net, density_network):
    torch.cuda.empty_cache()
    # field_net takes input in [-1, 1]
    grids = []
    grads = []
    for b in range(0, coords.shape[0], 10000):
        grid, grad = field_net(coords[b: b + 10000])
        # grad.mean().backward()
        grids.append(grid.detach())
        grads.append(grad.detach())
        torch.cuda.empty_cache()
    with torch.no_grad():
        grids = torch.cat(grids, 0)
        grads = torch.cat(grads, 0)

        grids = grids.detach()
        grads = grads.detach()
        grads = torch.linalg.norm(grads, dim=-1)
        densities = []
        for b in range(0, coords.shape[0], 10000):
            density = density_network.density(coords[b: b + 10000])
            densities.append(density)
        densities = torch.cat(densities, 0)
        torch.cuda.empty_cache()
    return densities.squeeze(), grids.squeeze(), grads.squeeze()


def generate_field_values(coords, field_net, compute_grads=False):
    torch.cuda.empty_cache()
    # field_net takes input in [-1, 1]
    grids = []
    grads = []
    for b in range(0, coords.shape[0], 10000):
        grid, grad = field_net(coords[b: b + 10000], return_grad=compute_grads)
        # grad.mean().backward()
        grids.append(grid.detach())
        if compute_grads:
            grads.append(grad.detach())
        torch.cuda.empty_cache()
    with torch.no_grad():
        grids = torch.cat(grids, 0)
        if compute_grads:
            grads = torch.cat(grads, 0)

        grids = grids.detach()
        if compute_grads:
            grads = grads.detach()
            grads = torch.linalg.norm(grads, dim=-1).squeeze()
        torch.cuda.empty_cache()
    return grids.squeeze(), grads


# preprocess these values

# generate mesh
def generate_mesh(output, thres, length, m=None):
    if m is not None:
        verts, faces, _, _ = measure.marching_cubes(output, level=thres, mask=m)
    else:
        verts, faces, _, _ = measure.marching_cubes(output, level=thres)
    verts = verts / (length - 1)
    return verts, faces


def expand_binaries(binaries, mode="nearest", M = 1024, levels=4):
    # combine binaries
    Mask = torch.from_numpy(np.zeros((M, M, M), dtype=np.float16)).cuda()
    with torch.no_grad():
        for i in range(levels -1, -1, -1):
            size = M // 2 ** (levels - 1 - i)

            kernel = torch.nn.Upsample(size=M // 2 ** (levels - 1 - i), mode=mode)
            b = kernel(binaries[i][None, None])[0, 0]
            Mask[M // 2 - size // 2: M // 2 + size // 2, M // 2 - size // 2: M // 2 + size // 2,
            M // 2 - size // 2: M // 2 + size // 2] = \
                b
    if mode == "nearest":
        Mask = Mask.cpu().numpy().astype(bool)
    else:
        Mask = Mask.cpu().numpy().astype(np.float16)
    return Mask


from scipy.spatial import KDTree


def down_sample_naive(mesh, num, thres, verbose=False):
    for i in range(num):
        if verbose:
            print("Before ", mesh)
        tree = KDTree(mesh.vertices.astype(np.float32))

        d, indices = tree.query(mesh.vertices.astype(np.float32), 2)

        mask = d[:, 1] < 1 / thres
        v_merge = (mesh.vertices[mask] + mesh.vertices[indices[:, 1]][mask]) / 2
        mesh.vertices[mask] = v_merge
        mesh.vertices[indices[:, 1]][mask] = v_merge

        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

        if verbose:
            print("After ", mesh)
    return mesh


def downsample_mesh(mesh, vx=0):
    if vx > 0:
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                             o3d.utility.Vector3iVector(mesh.faces))
        mesh_smp = mesh_o3d.simplify_vertex_clustering(1 / vx, o3d.geometry.SimplificationContraction.Quadric)
        del mesh
        mesh = trimesh.Trimesh(vertices=np.array(mesh_smp.vertices),
                               faces=np.array(mesh_smp.triangles))
    return mesh


def down_sample_quadraic(mesh, agg, verbose=False):
    import fast_simplification
    points, faces = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces,
                                                 target_count=mesh.faces.shape[0] // 10, agg=agg, verbose=verbose)
    mesh = trimesh.Trimesh(points, faces)
    return mesh


def clean_mesh(mesh, agg=None, remove_duplicate_faces=False, remove_non_manifold_edges=False, verbose=False):
    print (mesh)
    if remove_duplicate_faces:
        mesh.remove_duplicate_faces()
    print ("After removing duplicate faces: ", mesh)

    if agg > 0:
        import fast_simplification
        points, faces = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces,
                                                     target_count=mesh.faces.shape[0] // 10, agg=agg, verbose=verbose)
        mesh = trimesh.Trimesh(points, faces)
        print ("After simplification: ", mesh)
    if remove_non_manifold_edges:
        o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                o3d.utility.Vector3iVector(mesh.faces))
        o3d_mesh.remove_non_manifold_edges()
        o3d_mesh.remove_duplicated_triangles()
        print ("After removing non manifold edges: ", o3d_mesh)
        mesh = trimesh.Trimesh(np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles))

    return mesh


# transform coordinates
def transform_mesh(verts, faces, mean, scale, vx=0):
    # given a mesh coordinates between 0-1, transform them to desired coordinates
    if vx > 0:
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        mesh_smp = mesh_o3d.simplify_vertex_clustering(1 / vx)

        mesh = trimesh.Trimesh(vertices=np.array(mesh_smp.vertices),
                               faces=np.array(mesh_smp.triangles))
    else:
        mesh = trimesh.Trimesh(vertices=np.array(verts),
                               faces=np.array(faces))
    mesh.vertices = mesh.vertices / scale
    mesh.vertices = mesh.vertices + mean
    return mesh

def generate_density_grid(coords, radiance_field):
    # Compute density using neural radiance field
    with torch.no_grad():
        densities = []
        for b in range(0, coords.shape[0], 10000):
            density = radiance_field.query_density(coords[b: b + 10000])
            densities.append(density)
        densities = torch.cat(densities, 0)
    return densities.squeeze()


def prunning_mesh_train_visibility_complement(mesh, train_dataset, max_hits=5):
    meshes = []
    raytrace = RayTrace(mesh, num_intersections=max_hits)
    all_tris = []
    batch_size = 100000
    for j, data in enumerate(train_dataset):
        if j == len(train_dataset.images): break
        dirs = data["rays"].viewdirs.reshape((-1, 3))
        origins = data["rays"].origins.reshape((-1, 3))
        tris = []
        for i in range(0, dirs.shape[0], batch_size):
            tri = \
            raytrace.raytrace(dirs[i:i + batch_size].data.cpu().numpy(), origins[i:i + batch_size].data.cpu().numpy())[
                0]
            if len(tri) == 0: continue
            tri = tri.flatten()
            tris.append(np.unique(tri).flatten())
        try:
            all_tris.append(np.concatenate(tris))
        except:
            print ("Error in this view?", j)
            continue
    all_tris = np.concatenate(all_tris).astype(np.int32)

    unique = np.unique(all_tris)
    mask = np.zeros((mesh.faces.shape[0]), dtype=bool)
    try:
        mask[unique] = True
    except:
        import ipdb; ipdb.set_trace()
    # mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[mask])
    mesh_c = mesh.copy()
    mesh.update_faces(mask)
    mesh_c.update_faces(~mask)
    return mesh, mesh_c, mask


def prunning_mesh_train_visibility(mesh, train_dataset, max_hits=5):
    raytrace = RayTrace(mesh, num_intersections=max_hits)
    all_tris = []
    batch_size = 100000
    for j, data in enumerate(train_dataset):
        if j == len(train_dataset.images): break
        print(j)
        dirs = data["rays"].viewdirs.reshape((-1, 3))
        origins = data["rays"].origins.reshape((-1, 3))
        tris = []
        for i in range(0, dirs.shape[0], batch_size):
            tri = \
            raytrace.raytrace(dirs[i:i + batch_size].data.cpu().numpy(), origins[i:i + batch_size].data.cpu().numpy())[
                0]
            if len(tri) == 0: continue
            tri = tri.flatten()
            tris.append(np.unique(tri).flatten())
        try:
            all_tris.append(np.concatenate(tris))
        except:
            print ("Error in this view?", j)
            continue
    all_tris = np.concatenate(all_tris).astype(np.int32)

    unique = np.unique(all_tris)
    mask = np.zeros((mesh.faces.shape[0]), dtype=bool)
    try:
        mask[unique] = True
    except:
        import ipdb; ipdb.set_trace()
    # mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[mask])
    mesh.update_faces(mask)
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    # mesh.remove_degenerate_faces()
    # mesh.remove_unreferenced_vertices()
    return mesh, mask


def grid_transmittance(scene, root, radiance_field_path, args):
    from datasets.nerf_360_v2 import SubjectLoader
    from radiance_fields.ngp import NGPRadianceFieldSGNew
    chunk_size = 512
    SIZE = 2048
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]).cuda()
    near_plane = 0.2
    far_plane = 1.0e2
    extra_categories=["green_stuffie", "labdog", "kitchen", "counter", "room", "bonsai"]

    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 2 if args.scene in extra_categories else 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 5 if args.scene in ["stump", "bicycle"] else 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-4
    cone_angle = 0.004
    unbounded = True
    density_ckpt = "{}/nerfacc/ckpts/{}/{}/ngp.pth".format(root, scene, radiance_field_path)

    estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).cuda()
    estimator.load_state_dict(torch.load(density_ckpt)["estimator"])
    device = torch.device("cuda")
    if args.num_lobes > 0:
        if unbounded:
            radiance_field = NGPRadianceFieldSGNew(aabb=estimator.aabbs[0],
                                                   use_viewdirs=False,
                                                   num_g_lobes=args.num_lobes,
                                                   num_layers=args.num_layers,
                                                   unbounded=True).to(device)
        else:
            radiance_field = NGPRadianceFieldSGNew(aabb=estimator.aabbs[-1],
                                                   use_viewdirs=False,
                                                   num_g_lobes=args.num_lobes,
                                                   num_layers=args.num_layers,
                                                   unbounded=False).to(device)
    else:
        if unbounded:
            radiance_field = NGPRadianceField(aabb=estimator.aabbs[0], num_layers=2, hidden_size=64,
                                              unbounded=unbounded).to(device)
        else:
            radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64,
                                              unbounded=unbounded).to(device)


    upsample_rays = 1
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 1}
    train_dataset = SubjectLoader(
        subject_id=scene,
        root_fp="{}/mvd/data/tandt/db/".format(root),
        split="train",
        num_rays=None,
        device=torch.device("cuda"),
        **train_dataset_kwargs,
        upsample=upsample_rays
    )

    torch.cuda.empty_cache()
    # radiance_field = NGPRadianceField(aabb=estimator.aabbs[0], num_layers=2, unbounded=True).cuda()
    radiance_field.load_state_dict(torch.load(density_ckpt)["model"])

    Mask = torch.zeros((chunk_size, chunk_size, chunk_size), dtype=torch.bool).cuda()

    radiance_field.eval()
    for j in tqdm.tqdm(range(len(train_dataset.images))):
        data = train_dataset[j]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        # pixels = data["pixels"]
        with torch.no_grad():
            try:
                rgb, acc, depth, _, positions = render_image_with_occgrid_test(
                1024,
                # scene
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                early_stop_eps=5e-3
                )
                # rgb_loss = F.smooth_l1_loss(rgb, pixels)

                # psnr = -10.0 * torch.log(rgb_loss) / np.log(10.0)

                contracted_positions = contract_to_unisphere(positions, radiance_field.aabb)

                coords = torch.round(contracted_positions * (Mask.shape[0] - 1)).to(torch.long)
                Mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                # coords = torch.ceil(contracted_positions * (Mask.shape[0] - 1)).to(torch.long)
                # Mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                del contracted_positions, coords, positions, rgb, acc, depth
                torch.cuda.empty_cache()
            except:
                print ("Error while processing image {}".format(j))

    del radiance_field, estimator
    torch.cuda.empty_cache()
    kernel_up = torch.nn.Upsample((SIZE, SIZE, SIZE), mode="trilinear", align_corners=False)
    Mask = kernel_up(Mask[None, None].float().cpu())[0, 0] > 0.5
    torch.save(Mask, "{}/nerfacc/results/{}/{}/binaries_transmittance.pth".format(root, scene,
                                                                                    radiance_field_path))
    return Mask


def grid_transmittance_synthetic(scene, root, radiance_field_path, args):
    from datasets.nerf_synthetic import SubjectLoader
    from radiance_fields.ngp import NGPRadianceFieldSGNew
    chunk_size = 256
    SIZE = 1024
    device=  torch.device("cuda")
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10

    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0
    unbounded = False
    density_ckpt = "{}/nerfacc/ckpts/{}/{}/ngp.pth".format(root, scene, radiance_field_path)

    estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=128, levels=1
        ).cuda()
    estimator.load_state_dict(torch.load(density_ckpt)["estimator"])
    estimator.eval()
    device = torch.device("cuda")
    if args.num_lobes > 0:
        if unbounded:
            radiance_field = NGPRadianceFieldSGNew(aabb=estimator.aabbs[0],
                                                   use_viewdirs=False,
                                                   num_g_lobes=args.num_lobes,
                                                   num_layers=args.num_layers,
                                                   unbounded=True).to(device)
        else:
            radiance_field = NGPRadianceFieldSGNew(aabb=estimator.aabbs[-1],
                                                   use_viewdirs=False,
                                                   num_g_lobes=args.num_lobes,
                                                   num_layers=args.num_layers,
                                                   unbounded=False).to(device)
    else:
        if unbounded:
            radiance_field = NGPRadianceField(aabb=estimator.aabbs[0], num_layers=2, hidden_size=64,
                                              unbounded=unbounded).to(device)
        else:
            radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], num_layers=2, hidden_size=64,
                                              unbounded=unbounded).to(device)


    upsample_rays = 1
    train_dataset_kwargs = {}
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp="{}/nerfacc/nerf_synthetic/".format(root),
        split="whole",
        num_rays=1 << 17,
        device=torch.device("cuda"),
        **train_dataset_kwargs,
        mesh_intersect=None,
        upsample=upsample_rays,
    )

    torch.cuda.empty_cache()
    radiance_field.load_state_dict(torch.load(density_ckpt)["model"])

    Mask = torch.zeros((chunk_size, chunk_size, chunk_size), dtype=torch.bool).cuda()

    radiance_field.eval()
    for j in tqdm.tqdm(range(len(train_dataset.images))):
        data = train_dataset[j]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        with torch.no_grad():
            try:
                rgb, acc, depth, _, positions = render_image_with_occgrid_test(
                1024,
                # scene
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                early_stop_eps=1e-4
                )
            except:
                print ("Error while processing image {}".format(j))
                continue
            rgb_loss = F.smooth_l1_loss(rgb, pixels)

            psnr = -10.0 * torch.log(rgb_loss) / np.log(10.0)
            print ("PSNR: ", psnr)
            contracted_positions = radiance_field.normalize(positions)[1]

            coords = torch.floor(contracted_positions * (Mask.shape[0] - 1)).to(torch.long)
            Mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            coords = torch.ceil(contracted_positions * (Mask.shape[0] - 1)).to(torch.long)
            Mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            del contracted_positions, coords, positions, rgb, acc, depth
            torch.cuda.empty_cache()
        print (j)
    del radiance_field, estimator
    torch.cuda.empty_cache()
    kernel_up = torch.nn.Upsample((SIZE, SIZE, SIZE), mode="trilinear", align_corners=False)
    Mask = kernel_up(Mask[None, None].float().cpu())[0, 0] > 0.5
    torch.save(Mask, "{}/nerfacc/results/{}/{}/binaries_transmittance.pth".format(root, scene,
                                                                                    radiance_field_path))
    return Mask
