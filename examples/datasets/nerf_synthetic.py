"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import collections
import json
import os
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import time
from .utils import Rays
import ctypes
import multiprocessing as mp
import os
import trimesh
import numpy as np
# import open3d as o3d
import pyrr


def spiral(scale=1, z=0):
    u = np.linspace(0.01, 2 * np.pi - 0.01, 100)
    x = np.sin(u)
    y = np.cos(u)
    z = np.ones_like(y) * 0.01
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], 1) * scale
    return points


def create_uniform_camera_poses(distance=2):
    frontvectors = spiral(distance, 0.1)
    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontvectors[i],
                                                     target=np.zeros(3),
                                                     up=np.array([0.0, 0.0, 1.0])).T)
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
        print (camera_pose)
    return np.stack(camera_poses, 0)


def trimesh_ray_tracing(M, resolution=800, fov=60):
    mesh = trimesh.Trimesh()
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()
    # np.linalg.inv(create_look_at(frontVector, np.zeros(3), np.array([0, 1, 0])))
    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]
    scene.camera.resolution = [resolution, resolution]
    scene.camera.fov = fov, fov
    origins, vectors, pixels = scene.camera_rays()
    # x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    origins_new = np.zeros((resolution, resolution, 3))
    vectors_new = np.zeros((resolution, resolution, 3))
    # # pixels_indices = np.stack([x, y], axis=-1)
    origins_new[pixels[:, 1], pixels[:, 0]] = origins
    vectors_new[pixels[:, 1], pixels[:, 0]] = vectors
    return origins_new.reshape((-1, 3)), vectors_new.reshape((-1, 3)), pixels


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        try:
            fname = os.path.join(data_dir, frame["file_path"] + ".png")
            rgba = imageio.imread(fname)
        except:
            fname = os.path.join(data_dir, frame["file_path"])
            rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


def _load_parameters(root_fp: str, subject_id: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)

    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        camtoworlds.append(frame["transform_matrix"])

    camtoworlds = np.stack(camtoworlds, axis=0)

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
            mesh_intersect = None,
            fine_tune_vertices=False,
            add_ray_direction_noise=False,
        upsample=1,

    ):
        super().__init__()
        self.upsample = int(upsample)
        NEAR, FAR = 2.0, 6.0
        OPENGL_CAMERA = True
        # assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.mesh_intersect = mesh_intersect
        self.fine_tune_vertices = fine_tune_vertices
        self.add_ray_direction_noise = add_ray_direction_noise
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        elif split == "train" or split == "test":
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        else:
            # this is used when we want to do evaluation using the training set
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, "train"
            )
        self.focal = self.focal * self.upsample
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        height, width = self.images.shape[1:3]
        self.WIDTH, self.HEIGHT = int(width * self.upsample), int(self.upsample * height)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)
        print (self.images.shape)
        print (self.HEIGHT, self.WIDTH)
        assert self.images.shape[1:3] == (self.HEIGHT // self.upsample, self.WIDTH // self.upsample)

    def __len__(self):
            return 10000000 #len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        index = index % len(self.images)
        data = self.fetch_data(index)
        data = self.preprocess(data)
        if self.mesh_intersect is not None:
            if self.fine_tune_vertices:
                self.mesh_intersect.rayintersector.mesh.vertices = self.shared_array.data.cpu().numpy()
                points, origins, deltas, vectors, index_ray, index_tri = self.mesh_intersect.sampling_for_fine_tuning_mesh_ray_trace(
                    data["rays"].viewdirs.data.cpu().numpy(), \
                    data["rays"].origins.data.cpu().numpy())
                data["data"] = [points.astype(np.float32), origins.astype(np.float32), deltas.astype(np.float32), vectors.astype(np.float32), index_ray.astype(np.int64), index_tri, self.mesh_intersect.mesh]
                # print(self.mesh_intersect.rayintersector.mesh.vertices.sum())
            else:
                xyzs, dirs, index_ray, ts, index_tri, _, origins = self.mesh_intersect.sampling_raytrace_numpy(
                    data["rays"].viewdirs.data.cpu().numpy(), \
                    data["rays"].origins.data.cpu().numpy(), \
                    0)
                # convert all the data to torch tensors with the right type
                data["data"] = [torch.from_numpy(xyzs.astype(np.float32)), torch.from_numpy(dirs.astype(np.float32)), torch.from_numpy(index_ray.astype(np.int64)), torch.from_numpy(ts.astype(np.float32)),
                                torch.from_numpy(index_tri.astype(np.int64)), torch.from_numpy(origins.astype(np.float32))]
                # data["data"] = [torch.from_numpyxyzs.astype(np.float32), dirs.astype(np.float32), index_ray.astype(np.int64), ts.astype(np.float32),
                #         index_tri.astype(np.int64), _, origins.astype(np.float32)]
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        if self.upsample != 1 and not self.training:
            # stratified sampling
            x_, y_ = torch.meshgrid(
            torch.arange(self.WIDTH // self.upsample, device=self.images.device),
            torch.arange(self.HEIGHT // self.upsample, device=self.images.device),
            indexing="xy",
            )
            x_ = x_.flatten()
            y_ = y_.flatten()
            # We want the image to be in original resolution for evaluation
            rgba = self.images[image_id, y_, x_] / 255.0  # (num_rays, 4)
        else:
            rgba = self.images[image_id, torch.floor(y / self.upsample).long(), torch.floor(x / self.upsample).long()] / 255.0  # (num_rays, 4)

        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        if self.add_ray_direction_noise and self.training:
            x = x.float()
            y = y.float()
            
            x = x.float() + torch.rand_like(x)
            y = y.float() + torch.rand_like(y)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

            # We want the image to be in original resolution for evaluation
            rgba = torch.reshape(rgba, (self.HEIGHT // self.upsample, self.WIDTH // self.upsample, 4))
        rgba = rgba.view(-1, 4)
        rays = Rays(origins=origins.view(-1, 3), viewdirs=viewdirs.view(-1, 3))

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }


class SubjectLoaderOwnViews(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
            self,
            resolution=2000,
            mesh_intersect=None,
            camera_scale=2.0,
    ):
        super().__init__()
        self.mesh_intersect = mesh_intersect
        # 1.8 for khady
        self.camera_poses = create_uniform_camera_poses(camera_scale)
        self.len_images = self.camera_poses.shape[0]
        self.resolution = resolution
        self.WIDTH, self.HEIGHT = resolution, resolution
        self.NEAR, self.FAR = 2.0, 6.0
        self.OPENGL_CAMERA = True

    def __len__(self):
        return self.len_images

    @torch.no_grad()
    def __getitem__(self, index):
        index = index % self.len_images
        origins_original, vectors, pixels = trimesh_ray_tracing(self.camera_poses[index], resolution=self.resolution)

        # normalize vectors to unit norm
        vectors = vectors / np.linalg.norm(vectors, axis=1, ord=2)[:, None]
        data = {}

        xyzs, dirs, index_ray, ts, index_tri, _, origins = self.mesh_intersect.sampling_raytrace_numpy(
            vectors.astype(np.float32), \
            origins_original.astype(np.float32), \
            0)
        # convert all the data to torch tensors with the right type
        origins_original = torch.from_numpy(origins_original.astype(np.float32))
        vectors = torch.from_numpy(vectors.astype(np.float32))
        data["data"] = [torch.from_numpy(xyzs.astype(np.float32)), torch.from_numpy(dirs.astype(np.float32)),
                        torch.from_numpy(index_ray.astype(np.int64)), torch.from_numpy(ts.astype(np.float32)),
                        torch.from_numpy(index_tri.astype(np.int64)),
                        torch.from_numpy(origins.astype(np.float32))]

        rays = Rays(origins=origins_original.view(-1, 3), viewdirs=vectors.view(-1, 3))
        data["rays"] = rays
        data["pixels"] = pixels
        return data
