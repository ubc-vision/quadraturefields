import os

import torch
import numpy as np
import torch
from matplotlib import pyplot as plt
# from open3d import *
import pyrr
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import trimesh
import kaolin.render.spc as spc_render
# import open3d as o3d
# from open3d import *
import yappi
from torch_scatter import scatter_mean, scatter_add
import time


def compute_rgb(c, density, deltas, boundary):
    pass


import numpy as np

epsilon = 1e-6


import torch


@torch.jit.script
def ray_triangle_intersection(o, r, n, v):
    d = -(n * v).sum(1)
    t = -((n * o).sum(1) + d) / (n * r).sum(1)
    sign = torch.sign(t)
    t = sign * t
    psi = o + t[..., None] * r
    return psi


def ray_plane_intersect(ray, normal, point):
    rayDirection = ray[3:]
    rayPoint = ray[0:3]
    ndotu = normal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")
    w = rayPoint - point
    si = -normal.dot(w) / ndotu
    Psi = w + si * rayDirection + point
    return Psi


def ray_plane_intersect_array(ray, normal, point):
    rayDirection = ray[:, 3:]
    rayPoint = ray[:, 0:3]
    ndotu = np.sum(normal * (rayDirection), -1, keepdims=True)
    w = rayPoint - point
    si = -np.sum(normal * w, 1, keepdims=True) / (ndotu + 1e-8)
    Psi = w + si * rayDirection + point
    return Psi

# Convert the above funcrtion in torch
def ray_plane_intersect_array_torch(ray, normal, point):
    rayDirection = ray[:, 3:]
    rayPoint = ray[:, 0:3]
    ndotu = torch.sum(normal * (rayDirection), -1, keepdims=True)
    w = rayPoint - point
    si = -torch.sum(normal * w, 1, keepdims=True) / (ndotu + 1e-8)
    Psi = w + si * rayDirection + point
    return Psi


class RayIntersector:
    def __init__(self, mesh, max_hits=10):
        from build.lib import intersector
        self.mesh = mesh
        self.vertices = mesh.vertices[mesh.faces].flatten()
        self.max_hits = max_hits
        self.inter = intersector.Intersector(self.vertices, max_hits, 0)

    def update_intersector(self, vertices):
        self.inter.update_vertices(vertices)

    @torch.no_grad()
    def intersects_id(self, origins, vectors, multiple_hits=True, return_locations=True, max_hits=10):
        rays = np.concatenate([origins, vectors], 1)
        rays = rays.flatten()

        ints = self.inter.find_intersections(rays)
        rays = np.array(rays)
        ints = np.array(ints)
        indices = np.where(np.array(ints) > -1)[0]
        triangle_indices = ints[indices]
        ray_indices = indices // self.max_hits
        rays = rays.reshape((-1, 6))
        # psi = ray_plane_intersect_array(rays[ray_indices], self.mesh.face_normals[triangle_indices],
        #                                 self.mesh.vertices[self.mesh.faces[triangle_indices][:, 0]])
        # psi = ray_plane_intersect_array_torch(torch.from_numpy(rays[ray_indices]).cuda(), torch.from_numpy(self.mesh.face_normals[triangle_indices].astyp(np.float32)).cuda(),
        #                                 torch.from_numpy(self.mesh.vertices[self.mesh.faces[triangle_indices][:, 0]]).cuda()).data.cpu().numpy()
        o = torch.from_numpy(rays[ray_indices][:, 0:3].astype(np.float32)).cuda()
        r = torch.from_numpy(rays[ray_indices][:, 3:].astype(np.float32)).cuda()
        n = torch.from_numpy(self.mesh.face_normals[triangle_indices].astype(np.float32)).cuda()
        v = torch.from_numpy(self.mesh.vertices[self.mesh.faces[triangle_indices][:, 0]].astype(np.float32)).cuda()

        psi = ray_triangle_intersection(o, r, n, v)
        psi = psi.data.cpu().numpy()
        return triangle_indices, ray_indices, psi


class MeshFinetune:
    def __init__(self, vertices, faces, scaling) -> None:
        if not isinstance(vertices, torch.Tensor):
            self.vertices = np.array(vertices).astype(np.float32)
        # self.vertices_orig = np.array(vertices).astype(np.float32)
        self.faces = torch.from_numpy(faces).cuda().long()
        self.cache_d = torch.zeros((faces.shape[0], 3), device=torch.device('cuda'))
        self.cache_w = torch.ones(faces.shape[0], device=torch.device("cuda")) * 1e-8
        self.cache_d.requires_grad = False
        self.cache_w.requires_grad = False
        self.scaling = scaling
        # self.gamma = gamma
        # self.s = self.sum_of_seq(gamma, epochs)

    @torch.no_grad()
    def update_d(self, d, w, index_tri):
        cache_d = torch.zeros_like(self.cache_d, device=torch.device('cuda'))
        cache_w = torch.zeros_like(self.cache_w, device=torch.device('cuda'))
        scatter_add(d * w[..., None], index_tri, dim=0, out=cache_d)
        scatter_add(w, index_tri, dim=0, out=cache_w)
        self.cache_d += cache_d
        self.cache_w += cache_w

    @torch.no_grad()
    def update_faces(self):
        torch.cuda.empty_cache()
        deformation = self.cache_d / (self.cache_w.unsqueeze(1))
        deformation = torch.clip(deformation, -self.scaling, self.scaling)
        df_vertices = torch.repeat_interleave(deformation, dim=0, repeats=3)
        dv1 = torch.zeros((self.vertices.shape[0], 3), device=torch.device('cuda'))
        scatter_mean(df_vertices, self.faces.flatten(), dim=0, out=dv1)
        self.vertices += (dv1.data.cpu().numpy())  # / self.s
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reset_d(self):
        self.cache_d[:] = 0
        self.cache_w[:] = 1e-8
        # self.scaling = self.scaling * self.gamma

    def sum_of_seq(self, gamma, epochs):
        s = 0
        for i in range(1, epochs):
            s += gamma ** i
        return s

class RayTrace:
    def __init__(self, mesh, num_intersections=20):
        # from build.lib import intersector
        self.num_intersections = num_intersections
        print("Using NVIDIA optix")
        try:
            print ("Using optix for ray tracing")
            from build.lib import intersector
            self.rayintersector = RayIntersector(mesh, max_hits=self.num_intersections)
        except:
            print ("Using embree for ray tracing")
            self.rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        # self.rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    def raytrace(self, vectors, origins):
        """
        Samples the points and define the new samples using barycentric coordinates
        """
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins, vectors, multiple_hits=True,
                                                                         return_locations=True, max_hits=self.num_intersections)
        return index_tri, index_ray, points

class MeshIntersection:
    def __init__(self, mesh_path, simplify_mesh=True, scale=1.0, num_repeat=16, optix=False, voxel_size=512, num_intersections=20, render_step_size=0.005):
        print("Reading mesh from: ", mesh_path)
        try:
            print ("Using open3d for reading mesh")
            import open3d as o3d
            import nonsense
            self.mesh = o3d.io.read_triangle_mesh(mesh_path)

            print("Before mesh simplification: ", np.array(self.mesh.vertices).shape)
            print("Before mesh simplification: ", np.array(self.mesh.triangles).shape)
        except:
            print ("Using trimesh for reading mesh")
            self.mesh = trimesh.load(mesh_path, force='mesh', process=False)
            print("Before mesh simplification: ", np.array(self.mesh.vertices).shape)
            print("Before mesh simplification: ", np.array(self.mesh.faces).shape)

        self.num_repeat = num_repeat
        self.num_intersections = num_intersections
        self.render_step_size = render_step_size
        if simplify_mesh:
            mesh_smp = self.mesh.simplify_vertex_clustering(1 / voxel_size)
            print("After mesh simplification: ", np.array(mesh_smp.vertices).shape, np.array(mesh_smp.triangles).shape)
            self.mesh = trimesh.Trimesh(vertices=np.array(mesh_smp.vertices),
                                        faces=np.array(mesh_smp.triangles), process=False)
        # else:
        #     try:
        #         self.mesh = trimesh.Trimesh(vertices=np.array(self.mesh.vertices).astype(np.float32),
        #                                     faces=np.array(self.mesh.faces), process=False)
        #     except:
        #         self.mesh = trimesh.Trimesh(vertices=np.array(self.mesh.vertices).astype(np.float32),
        #                                     faces=np.array(self.mesh.triangles), process=False)
        self.mesh.vertices *= scale

        self.vertices = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
        if optix:
            from build.lib import intersector
            print("Using NVIDIA optix")
            print(self.mesh.vertices.shape, self.mesh.faces.shape)
            self.rayintersector = RayIntersector(self.mesh, max_hits=self.num_intersections)
        else:
            print("Using Intel pyembree")
            print(self.mesh.vertices.shape, self.mesh.faces.shape)
            self.rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)

    def find_deltas(self, boundary, depth):
        deltas = torch.zeros(depth.shape[0], device=torch.device('cuda'))
        deltas[:] = self.render_step_size  # 0.005
        if not deltas.shape[0] == depth.shape[0]:
            import ipdb;
            ipdb.set_trace()
        return deltas

    def aabb_intersection(self, origins, dirs):
        tmin = (torch.tensor([[-1, -1, -1]]).cuda() - origins) / dirs
        tmax = (torch.tensor([[1, 1, 1]]).cuda() - origins) / dirs

        t1 = torch.minimum(tmin, tmax)
        t2 = torch.maximum(tmin, tmax)
        tnear = torch.maximum(torch.maximum(t1[:, 0], t1[:, 1]), t1[:, 2])
        tfar = torch.minimum(torch.minimum(t2[:, 0], t2[:, 1]), t2[:, 2])
        return tnear, tfar

    def sampling(self, vectors, origins, random=0):
        """
        NOTE: Assuming that all origins are same in the given array
        """
        num_rays = vectors.shape[0]
        # vectors = vectors / np.linalg.norm(vectors, axis=1, ord=2)[:, None]
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins,
                                                                         vectors,
                                                                         multiple_hits=True,
                                                                         return_locations=True,
                                                                         max_hits=self.num_intersections)
        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        origins = origins[index_ray]

        # depth = trimesh.util.diagonal_dot(points - origins, vectors)
        depth = np.linalg.norm(points - origins, axis=1, ord=2) / (np.linalg.norm(vectors, axis=1, ord=2) + 1e-8)
        if random > 0:
            noise = np.random.uniform(-random, random, depth.shape)
            depth += noise

        new_indices = np.lexsort((depth, index_ray))

        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        if random > 0:
            noise = noise[new_indices]
            noise = torch.from_numpy(noise).cuda()
        vectors = vectors[new_indices]
        origins = origins[new_indices]

        index_ray = torch.from_numpy(index_ray).cuda().long()
        boundary = spc_render.mark_pack_boundaries(index_ray)
        start_index = torch.where(boundary)[0]
        counts = torch.diff(start_index, append=torch.tensor([index_ray.shape[0]], device=start_index.device)).long()
        ray_index = index_ray[boundary]

        if points.shape[0] - counts.sum() != 0: import ipdb; ipdb.set_trace()
        rays_a = torch.zeros((num_rays, 3), device=index_ray.device, dtype=torch.long)
        rays_a[:, 0] = torch.arange(num_rays, device=index_ray.device)
        rays_a[ray_index] = torch.stack([ray_index, start_index, counts], dim=1)

        vectors = torch.from_numpy(vectors.astype(np.float32)).cuda()
        # deltas = None
        deltas = self.find_deltas(boundary.data.cpu().numpy(), depth.astype(np.float32))
        # deltas = torch.from_numpy(deltas.astype(np.float32)).cuda()

        points = torch.from_numpy(points.astype(np.float32)).cuda()
        if random > 0:
            points = points + noise.view((-1, 1)) * vectors

        depth = torch.from_numpy(depth.astype(np.float32)).cuda()
        return points, deltas, boundary, vectors, index_ray, depth, index_tri, rays_a, origins

    def sampling_raytrace(self, vectors, origins, random=0):
        """
        NOTE: Assuming that all origins are same in the given array
        """
        num_rays = vectors.shape[0]
        # vectors = vectors / np.linalg.norm(vectors, axis=1, ord=2)[:, None]
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins,
                                                                         vectors,
                                                                         multiple_hits=True,
                                                                         return_locations=True,
                                                                         max_hits=self.num_intersections)
        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        origins = origins[index_ray]
        vectors = torch.from_numpy(vectors.astype(np.float32)).cuda()
        origins = torch.from_numpy(origins.astype(np.float32)).cuda()
        points = torch.from_numpy(points.astype(np.float32)).cuda()

        norm = torch.linalg.vector_norm(vectors, axis=1) + 1e-7
        vectors = vectors / norm[:, None]
        depth = torch.linalg.vector_norm(points - origins, axis=1)  # / (norm)

        depth_npy = depth.data.cpu().numpy()
        new_indices = np.lexsort((depth_npy, index_ray))
        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        origins = origins[new_indices]
        depth = depth[new_indices]
        depth_npy = depth_npy[new_indices]
        vectors = vectors[new_indices]
        index_ray = torch.from_numpy(index_ray).cuda().long()
        # boundary = spc_render.mark_pack_boundaries(index_ray)

        # deltas = self.find_deltas(boundary, depth)

        return points, vectors, index_ray, depth, index_tri, 0, origins

    def sampling_raytrace_numpy(self, vectors, origins, random=0):
        """
        NOTE: Assuming that all origins are same in the given array
        """
        num_rays = vectors.shape[0]
        # vectors = vectors / np.linalg.norm(vectors, axis=1, ord=2)[:, None]
        try:
            index_tri, index_ray, points = self.rayintersector.intersects_id(origins,
                                                                            vectors,
                                                                            multiple_hits=True,
                                                                            return_locations=True,
                                                                            max_hits=self.num_intersections)
        except:
            import ipdb; ipdb.set_trace()
        if index_tri.shape[0] == 0 or index_ray.shape[0] == 0 or points.shape[0] == 0:
            return None
        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        origins = origins[index_ray]
        # vectors = torch.from_numpy(vectors.astype(np.float32)).cuda()
        # origins = torch.from_numpy(origins.astype(np.float32)).cuda()
        # points = torch.from_numpy(points.astype(np.float32)).cuda()

        norm = np.linalg.norm(vectors, axis=1) + 1e-7
        vectors = vectors / norm[:, None]
        depth = np.linalg.norm(points - origins, axis=1)  # / (norm)

        # depth_npy = depth.data.cpu().numpy()
        depth_npy = depth
        new_indices = np.lexsort((depth_npy, index_ray))
        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        depth_npy = depth_npy[new_indices]
        vectors = vectors[new_indices]
        # index_ray = torch.from_numpy(index_ray).cuda().long()
        # boundary = spc_render.mark_pack_boundaries(index_ray)

        # deltas = self.find_deltas(boundary, depth)

        return points, vectors, index_ray, depth, index_tri, 0, origins

    def sampling_indexing(self, points, origins, vectors, index_ray, depth, index_tri, random=0):
        """
        NOTE: Assuming that all origins are same in the given array
        """
        num_rays = vectors.shape[0]
        depth_npy = depth.data.cpu().numpy()
        new_indices = np.lexsort((depth_npy, index_ray.data.cpu().numpy()))

        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        depth_npy = depth_npy[new_indices]
        origins = origins[new_indices]
        vectors = vectors[new_indices]
        # origins = origins[new_indices]

        # index_ray = torch.from_numpy(index_ray).cuda().long()
        boundary = spc_render.mark_pack_boundaries(index_ray)

        deltas = self.find_deltas(boundary, depth)
        # deltas = torch.from_numpy(deltas.astype(np.float32)).cuda()

        return points, deltas, boundary, vectors, index_ray, depth, index_tri, origins

    def sampling_alpha(self, vectors, origins):
        """
        NOTE: Assuming that all origins are same in the given array
        """
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins,
                                                                         vectors,
                                                                         multiple_hits=True,
                                                                         return_locations=True,
                                                                         max_hits=self.num_intersections)
        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        depth = trimesh.util.diagonal_dot(points - origins[0], vectors)
        new_indices = np.lexsort((depth, index_ray))

        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        vectors = vectors[new_indices]

        boundary = spc_render.mark_pack_boundaries(torch.from_numpy(index_ray).cuda().long())
        deltas = self.find_deltas(boundary.data.cpu().numpy(), depth.astype(np.float32))
        points = torch.from_numpy(points.astype(np.float32)).cuda()
        deltas = torch.from_numpy(deltas.astype(np.float32)).cuda()
        vectors = torch.from_numpy(vectors.astype(np.float32)).cuda()
        index_ray = torch.from_numpy(index_ray).long().cuda()
        alphas = self.alpha[index_tri]
        alpha_ = self.alpha_[index_tri]
        return points, deltas, boundary, vectors, index_ray, depth, alphas, alpha_

    def sampling_for_fine_tuning_mesh(self, vectors, origins, V):
        """
        Samples the points and define the new samples using barycentric coordinates
        """
        self.mesh.vertices = V.data.cpu().numpy()
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins, vectors, multiple_hits=True,
                                                                         return_locations=True, max_hits=self.num_intersections)

        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        origins = origins[index_ray]

        norm = np.linalg.norm(vectors, axis=1) + 1e-7
        vectors = vectors / norm[:, None]
        depth = np.linalg.norm(points - origins, axis=1)  # / (norm)

        new_indices = np.lexsort((depth, index_ray))

        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        vectors = vectors[new_indices]
        origins = origins[new_indices]

        b_coords = trimesh.triangles.points_to_barycentric(self.mesh.vertices[self.mesh.faces[index_tri]], points)
        b_coords = torch.from_numpy(b_coords.astype(np.float32)).cuda().unsqueeze(2)
        points = torch.sum(V[self.mesh.faces[index_tri]] * b_coords, 1)
        boundary = spc_render.mark_pack_boundaries(torch.from_numpy(index_ray).cuda().long())
        deltas = torch.zeros(depth.shape[0]).cuda()
        deltas[:] = 0.005

        vectors = torch.from_numpy(vectors.astype(np.float32)).cuda()
        index_ray = torch.from_numpy(index_ray).long().cuda()
        depth = torch.from_numpy(depth.astype(np.float32)).cuda()
        return points, deltas, boundary, vectors, index_ray, depth

    def sampling_for_fine_tuning_mesh_ray_trace(self, vectors, origins):
        """
        Samples the points and define the new samples using barycentric coordinates
        """
        index_tri, index_ray, points = self.rayintersector.intersects_id(origins, vectors, multiple_hits=True,
                                                                         return_locations=True, max_hits=self.num_intersections)

        indices = np.argsort(index_ray)
        index_tri = index_tri[indices]
        index_ray = index_ray[indices]
        points = points[indices]
        vectors = vectors[index_ray]
        origins = origins[index_ray]

        norm = np.linalg.norm(vectors, axis=1) + 1e-7
        vectors = vectors / norm[:, None]
        depth = np.linalg.norm(points - origins, axis=1)  # / (norm)

        depth = np.repeat(depth, self.num_repeat, axis=0)
        noise = np.random.normal(0, 1 / 256, depth.shape[0])
        # make sure the original points are intact
        noise[0:-1:self.num_repeat] = 0
        depth = depth + noise
        index_ray = np.repeat(index_ray, self.num_repeat, axis=0)
        index_tri = np.repeat(index_tri, self.num_repeat, axis=0)
        vectors = np.repeat(vectors, self.num_repeat, axis=0)
        origins = np.repeat(origins, self.num_repeat, axis=0)
        points = origins + vectors * depth[:, None]

        new_indices = np.lexsort((depth, index_ray))

        index_tri = index_tri[new_indices]
        index_ray = index_ray[new_indices]
        points = points[new_indices]
        depth = depth[new_indices]
        vectors = vectors[new_indices]
        origins = origins[new_indices]

        deltas = np.zeros(depth.shape[0], dtype=np.float32)
        deltas[:] = 0.005
        return points, origins, deltas, vectors, index_ray, index_tri


def create_uniform_camera_poses(distance=2):
    mesh = geometry.TriangleMesh()
    frontvectors = np.array(mesh.create_sphere(distance, 7).vertices)
    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontvectors[i],
                                                     target=np.zeros(3),
                                                     up=np.array([0.0, 1.0, 0])).T)
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def convert(a):
    a = a - a.min()
    a = a / a.max()
    return a


def geometry_intersections(outs, grid_size, image_size,
                           num_samples=32, if_use_transmittance=False,
                           thresh=0.0, multiple_hits=True, only_geometry=False):
    print("Using the threshold ", thresh)
    grid = []
    for j in range(grid_size):
        slice = outs[j]["field"].reshape((grid_size, grid_size))
        grid.append(slice)
    grids = torch.stack(grid, 2)
    np.save("grids.npy", grids.data.cpu().numpy())
    grids = torch.sin(1000 * grids).data.cpu().numpy()

    if if_use_transmittance:
        grids = get_surface_using_transmittance(grids, grid_size=grid_size).data.cpu().numpy()

    mesh, values = create_mesh(grids, thresh=0.)
    print("Finished marching cubes")
    o_mesh = geometry.TriangleMesh(utility.Vector3dVector(mesh.vertices), utility.Vector3iVector(mesh.faces))
    print("Starting mesh simplification")
    o_mesh = o_mesh.simplify_quadric_decimation(1000000)
    mesh.vertices, mesh.faces = np.array(o_mesh.vertices), np.array(o_mesh.triangles)

    if only_geometry:
        return mesh
    mesh.fix_normals()
    camera_poses = create_uniform_camera_poses_circular(num_samples)
    triangle_images, points, vectors, pixels, indices, depths = mesh_intersections(mesh, camera_poses, SIZE=grid_size,
                                                                                   multiple_hits=multiple_hits)
    return triangle_images, points, vectors, pixels, indices, depths, mesh, values


def create_mesh(grid, thresh=0.0):
    verts, faces, normals, values = measure.marching_cubes(grid, thresh)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.vertices = mesh.vertices / (grid.shape[0] - 1)
    mesh.vertices = (mesh.vertices - 0.5) * 2

    return mesh, values


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, origins=None, vectors=None, rayintersector=None,
                        multiple_hits=True):
    # this is done to correct the mistake in way trimesh raycasting works.
    # in general this cannot be done.
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    # np.linalg.inv(create_look_at(frontVector, np.zeros(3), np.array([0, 1, 0])))
    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]
    # scene.camera_transform = camera_transform_matrix(frontVector, np.zeros(3), np.array([0, 1, 0])) @ e

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [resolution, resolution]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov, fov

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()
    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=True, return_locations=True
    )
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    # find pixel locations of actual hits
    return index_tri, points, vectors, pixels, index_ray, depth


def create_uniform_camera_poses_circular(N, r=2):
    mesh = geometry.TriangleMesh()

    def rotation_matrix_y(degrees):
        rad = degrees / 180 * np.pi
        return np.array(
            [[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]]
        )

    indices = np.linspace(0, 2 * np.pi, N)
    frontvectors = np.stack([np.sin(indices) * 1.5, np.ones(indices.shape[0]), np.cos(indices) * 1.5], 1)
    camera_poses = []
    for i in range(N):
        # frontvectors = np.array([0, 0, 1]) * r
        # frontvectors = rotation_matrix_y(360 / N * i) @ frontvectors

        camera_pose = np.array(
            pyrr.Matrix44.look_at(
                eye=frontvectors[i], target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])
            ).T
        )
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def mesh_intersections(mesh, camera_poses, SIZE=1024, multiple_hits=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    triangle_images = []
    p_images = []
    vectors = []
    pixels = []
    index_rays = []
    depths = []
    for i in range(camera_poses.shape[0]):
        index_tri, points, vector, p, index, depth = trimesh_ray_tracing(
            mesh, camera_poses[i], resolution=SIZE, rayintersector=rayintersector, multiple_hits=multiple_hits,
        )

        triangle_images.append(index_tri)
        p_images.append(points)
        vectors.append(vector)
        pixels.append(p)
        index_rays.append(index)
        depths.append(depth)
    return triangle_images, p_images, vectors, pixels, index_rays, depths


def transmittance(density, z_vals, dists=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = density.device
    raw2alpha = lambda raw, d, act_fn=torch.relu: 1. - torch.exp(-act_fn(raw) * d)
    if not isinstance(dists, torch.Tensor):
        # dists = z_vals[..., 1:] - z_vals[..., :-1]
        # # [N_rays, N_samples]
        # dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)
        dists = z_vals.diff(dim=-1, prepend=(torch.zeros(z_vals.shape[0], 1, device=z_vals.device)))

    alpha = raw2alpha(density, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    transmit = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:,
               :-1]
    # alpha = raw2alpha(density, 2 * dists)
    weights = alpha * transmit
    return weights, transmit, dists


def composite(self, density_samples, depth_samples):
    depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [HW,N-1]
    depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)],
                                   dim=2)  # [HW,N]
    dist_samples = depth_intv_samples
    sigma_delta = density_samples.squeeze(-1) * dist_samples  # [HW,N]
    alpha = 1 - (-sigma_delta).exp_()  # [HW,N]
    T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2).cumsum(
        dim=2)).exp_()  # [HW,N]
    prob = (T * alpha)[..., None]  # [HW,N,1]
    return prob, None  # [HW,K], [HW,N,1]


def get_surface_using_transmittance(grid, grid_size=1024):
    grid = torch.from_numpy(grid).cuda()

    T = torch.zeros_like(grid).cuda()
    X, Y, Z = np.meshgrid(np.arange(grid_size).astype(np.uint16),
                          np.arange(grid_size).astype(np.uint16),
                          np.arange(grid_size).astype(np.uint16))
    Z = torch.from_numpy(Z.astype(np.int32)).cuda()

    T = torch.zeros_like(grid).cuda()

    for i in range(grid_size):
        T[i] = torch.maximum(transmittance(grid[i], Z[i] / grid_size + 1 / grid_size)[0], T[i])

    for i in range(grid_size):
        T[:, i] = torch.maximum(transmittance(grid[:, i], Z[i] / grid_size + 1 / grid_size)[0], T[:, i])

    for i in range(grid_size):
        T[:, :, i] = torch.maximum(transmittance(grid[:, :, i], Z[i] / grid_size + 1 / grid_size)[0], T[:, :, i])

    T = torch.flip(T, dims=(0, 1, 2))
    grid = torch.flip(grid, dims=(0, 1, 2))
    for i in range(grid_size):
        T[i] = torch.maximum(transmittance(grid[i], Z[i] / grid_size + 1 / grid_size)[0], T[i])

    for i in range(grid_size):
        T[:, i] = torch.maximum(transmittance(grid[:, i], Z[i] / grid_size + 1 / grid_size)[0], T[:, i])

    for i in range(grid_size):
        T[:, :, i] = torch.maximum(transmittance(grid[:, :, i], Z[i] / grid_size + 1 / grid_size)[0], T[:, :, i])

    T = torch.flip(T, dims=(0, 1, 2))
    grid = torch.flip(grid, dims=(0, 1, 2))
    return T


def get_surface_nerf2nerf(ray_data, camera_data, pipeline):
    mesh = trimesh.creation.box((2, 2, 2))
    # mesh = trimesh.load("/home/gopalshr/cape_coalwood/rendering/bunny.obj")
    # mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
    # mesh.vertices = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1).max()
    camera_poses = []
    camera_centers = []
    rays = []
    origins = []
    for k in range(len(camera_data)):
        # camera_center = np.linalg.inv(camera_data[k].extrinsics.view_matrix().data.cpu().numpy()[0])[:, -1][0:3]
        # Because of the weird blender transformation!!
        origins.append(ray_data.origins[k].numpy())
        # origin = np.repeat(camera_center.reshape((1, 3)), ray_data[k].shape[0], axis=0)
        rays.append(ray_data.dirs[k].numpy())
        # camera_centers.append(camera_center)

    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    import time
    num_samples_along_ray = 256
    grid_size = 512
    num_samples = 2
    T = torch.zeros(grid_size, grid_size, grid_size).cuda()
    for j in range(len(rays)):
        t1 = time.time()
        index_tri, index_ray, points = rayintersector.intersects_id(
            origins[j], rays[j], multiple_hits=True, return_locations=True
        )
        depth = trimesh.util.diagonal_dot(points - origins[j][index_ray], rays[j][index_ray])
        t2 = time.time()

        # trimesh.PointCloud(points).scene().show()
        print("Processing {}^th image".format(j))
        sorted_indices = np.argsort(index_ray)
        index_ray = index_ray[sorted_indices]
        points = points[sorted_indices]
        depth = depth[sorted_indices]

        t3 = time.time()
        uniques, unique_indices, _, counts = np.unique(index_ray,
                                                       return_counts=True,
                                                       return_index=True,
                                                       return_inverse=True)

        # import ipdb; ipdb.set_trace()
        Depths = np.ones((len(uniques), num_samples), dtype=np.float32) * 100
        Points = np.zeros((len(uniques), num_samples, 3), dtype=np.float32)
        for u in range(len(uniques)):
            Depths[u, 0:counts[u]] = depth[unique_indices[u]:unique_indices[u] + counts[u]][0:num_samples]
            Points[u, 0:counts[u]] = points[unique_indices[u]:unique_indices[u] + counts[u]][0:num_samples]

        sort_indices = np.argsort(Depths, axis=1)

        Depths = np.take_along_axis(Depths, sort_indices, axis=1)
        Points = np.take_along_axis(Points, sort_indices[:, :, None], axis=1)
        # trimesh.PointCloud(points).scene().show()

        invalid_indices = counts == 1
        Points[invalid_indices, 1] = Points[invalid_indices, 0]
        steps = np.linspace(0, 1, num_samples_along_ray).reshape(1, -1, 1).astype(np.float32)
        s_points = Points[:, None, 0] + (Points[:, None, 1] - Points[:, None, 0]) * steps
        t4 = time.time()
        outs = []

        with torch.no_grad():
            s_points = torch.from_numpy(s_points.astype(np.float32)).cuda()
            for index_j in range(0, s_points.shape[0], 10000):
                out = pipeline.nef.density(s_points[index_j: index_j + 10000], None)
                outs.append(out["density"])
        density = torch.cat(outs)
        s_points = torch.floor((s_points + 1 + 1e-6) * (grid_size - 0.1) / 2)
        t5 = time.time()
        Points = torch.from_numpy(Points.astype(np.float32)).cuda()
        Depths = torch.from_numpy(Depths.astype(np.float32)).cuda()
        sample_rel_depth = (Points[:, None, 1] - Points[:, None, 0]) * torch.from_numpy(steps).cuda()
        sample_rel_depth = torch.norm(sample_rel_depth, dim=-1)
        sample_depth = sample_rel_depth + Depths[:, None, 0]
        t6 = time.time()
        prob, _ = transmittance(density[:, :, 0], sample_depth)
        C = s_points.reshape((-1, 3)).long()

        T[C[:, 0], C[:, 1], C[:, 2]] = torch.maximum(prob.reshape((-1)), T[C[:, 0], C[:, 1], C[:, 2]])
        t7 = time.time()
        print("Time taken: ", t7 - t6, t6 - t5, t5 - t4, t4 - t3, t3 - t2, t2 - t1)

    mesh, _ = create_mesh(T.data.cpu().numpy().astype(np.float32), 0)
    torch.save(T, "_results/logs/runs/funny-nerf-experiment/two_spheres/nerf2nerf_grid_{}_{}.pth".format(grid_size,
                                                                                                         num_samples_along_ray))


class Bake:
    def __init__(self, mesh_path, grid, nef, dataset_path, bg_color, near=0, far=6.0):
        self.grid = grid
        self.nef = nef
        self.mesh_intersect = MeshIntersection(mesh_path)
        self.mesh = self.mesh_intersect.mesh
        self.near = near
        self.far = far
        nef.eval()
        if bg_color == 'white':
            bg_clor = [1, 1, 1]
        else:
            bg_color = [0, 0, 0]

        transform = SampleRays(num_samples=4096)
        self.train_dataset = wisp.datasets.load_multiview_dataset(
            dataset_path=dataset_path,
            split='train',
            mip=0,
            bg_color=bg_color,
            dataset_num_workers=0,
            transform=transform)

    def extract_alphas(self):
        Alphas = []
        vertices = np.array(self.mesh.vertices).astype(np.float32)[self.mesh.faces].mean(1)
        vertices = torch.from_numpy(vertices).cuda()

        Origins = self.train_dataset.data["rays"].origins.cuda()
        Dirs = self.train_dataset.data["rays"].dirs.cuda()
        origins = Origins[:, 0]

        for j in range(vertices.shape[0]):
            dirs = vertices[j:j + 1].cuda() - origins
            dirs = dirs / torch.linalg.vector_norm(dirs, ord=2, axis=1, keepdims=True)
            alphas = []
            for i in range(Origins.shape[0]):
                points, _, _, vectors, _, depth = self.mesh_intersect.sampling(dirs.data.cpu().numpy()[i:i + 1],
                                                                               origins[i:i + 1].data.cpu().numpy())
                if points.shape[0] == 0:
                    continue
                rays = Rays(Origins[:, 0], dirs, self.near, self.far)

                raymarch_results = self.grid.raymarch(rays[i:i + 1],
                                                      level=self.grid.active_lods[15],
                                                      num_samples=2048,
                                                      raymarch_type="ray")

                samples = raymarch_results.samples
                ridx = raymarch_results.ridx
                deltas = raymarch_results.deltas
                boundary = raymarch_results.boundary
                depth_samples = raymarch_results.depth_samples
                indices = torch.searchsorted(depth_samples[:, 0], torch.from_numpy(depth).cuda())

                with torch.no_grad():
                    output = self.nef.rgba(samples, rays[i:i + 1].dirs.repeat(samples.shape[0], 1))
                sigmas = output["density"]
                slots = torch.zeros((samples.shape[0])).cuda()
                try:
                    min_index = np.argmin(
                        np.linalg.norm(vertices[j:j + 1].data.cpu().numpy() - points.data.cpu().numpy(), ord=2, axis=1))
                except:
                    import ipdb;
                    ipdb.set_trace()
                if min_index == 0:
                    index_start = 0
                    index_end = indices[min_index]
                elif min_index == len(indices) - 1:
                    index_start = indices[min_index]
                    index_end = len(samples)
                else:
                    index_start = indices[min_index - 1]
                    index_end = indices[min_index]

                alpha = torch.exp(-sigmas[index_start:index_end] * deltas[index_start:index_end, 0])
                alpha = torch.prod(alpha)
                alphas.append(alpha)
            alphas = torch.tensor(alphas)
            print(torch.mode(alphas), torch.std(alphas), len(alphas))
            Alphas.append(torch.mode(alphas))
        return Alphas

    @torch.no_grad()
    def extract_alphas_fast(self, args):
        import time
        vertices = np.array(self.mesh.vertices).astype(np.float32)[self.mesh.faces].mean(1)
        vertices = torch.from_numpy(vertices).cuda()

        Origins = self.train_dataset.data["rays"].origins.cuda()
        Dirs = self.train_dataset.data["rays"].dirs.cuda()
        Alphas = torch.zeros(vertices.shape[0])
        Counts = torch.zeros(vertices.shape[0])
        chunks = 10000
        print(vertices.shape, len(Alphas))

        from tqdm import tqdm
        for i in tqdm([args.view_id], desc="Average over directions"):
            dirs = vertices.cuda() - Origins[i:i + 1, 0]
            dirs = dirs / torch.linalg.vector_norm(dirs, ord=2, axis=1, keepdims=True)

            origins_ = np.repeat(Origins[i:i + 1, 0].data.cpu().numpy(), repeats=dirs.shape[0], axis=0)
            points, _, _, vectors, index_ray, depth = self.mesh_intersect.sampling(dirs.data.cpu().numpy(), origins_)
            if points.shape[0] == 0:
                continue
            depth = torch.from_numpy(depth.astype(np.float32)).cuda()
            rays = Rays(Origins[i:i + 1, 0].repeat(dirs.shape[0], 1), dirs, self.near, self.far)

            for j in tqdm(range(0, vertices.shape[0], chunks), desc="Iterating over vertices"):
                raymarch_results = self.grid.raymarch(rays[j:j + chunks],
                                                      level=self.grid.active_lods[15],
                                                      num_samples=512,
                                                      raymarch_type="ray")

                samples = raymarch_results.samples
                ridx = raymarch_results.ridx
                deltas = raymarch_results.deltas
                boundary = raymarch_results.boundary
                depth_samples = raymarch_results.depth_samples

                with torch.no_grad():
                    output = self.nef.rgba(samples, rays[j:j + chunks].dirs[ridx], 1)
                sigmas = output["density"]

                for index_j in range(chunks):
                    valid_indices = index_ray == (j + index_j)
                    sample_valid_indices = ridx == index_j
                    if valid_indices.sum() == 0 or sample_valid_indices.sum() == 0:
                        continue
                    indices = torch.searchsorted(depth_samples[:, 0][sample_valid_indices], depth[valid_indices]).cuda()
                    try:
                        min_index = torch.min(
                            torch.linalg.vector_norm(vertices[j + index_j: j + index_j + 1] - points[valid_indices],
                                                     ord=2, dim=1), 0)[1]
                    except:
                        import ipdb;
                        ipdb.set_trace()
                    if min_index == 0:
                        index_start = 0
                        index_end = indices[min_index]
                    elif min_index == len(indices) - 1:
                        index_start = indices[min_index]
                        index_end = len(samples[sample_valid_indices])
                    else:
                        index_start = indices[min_index - 1]
                        index_end = indices[min_index]

                    alpha = torch.exp(
                        -sigmas[sample_valid_indices][index_start:index_end] * deltas[sample_valid_indices][
                                                                               index_start:index_end, 0])
                    alpha = torch.prod(alpha)
                    Alphas[index_j + j] += alpha.data.cpu()
                    Counts[index_j + j] += 1
                    # alpha = inner_step(j, index_j, index_ray, ridx, depth_samples, depth, points, vertices, samples, sigmas, deltas)
                    # if alpha==None: continue
                    # Alphas[index_j + j] += alpha.data.cpu()
                    # Counts[index_j + j] += torch.tensor(1)
                    if index_j + j >= vertices.shape[0] - 1: break
        return Alphas, Counts


def rotate_perturbation_point_cloud(angle_sigma=0.06, angle_clip=0.1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

