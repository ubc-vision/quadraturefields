"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
from .utils import Rays

_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


import os
import trimesh
import numpy as np
import open3d as o3d
import pyrr

#
# def spiral_kitchen(scale=1, z=0):
#     scale=0.4
#     u = np.linspace(0, 2 * np.pi, 100)
#     x = np.sin(u) * 2
#     y = np.cos(u) * 2
#     z = -0.8 * np.ones_like(x)
#     points = np.stack([x.flatten(), z.flatten(), y.flatten()], 1) * scale
#     return points


class CameraPose:
    def __init__(self, resolution_x, resolution_y, fov, scale, target, up, ):
        self.fov = fov
        self.scale = scale
        self.target = target
        self.up = up
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.camera_poses = create_uniform_camera_poses(distance=self.scale, target=self.target, up=self.up)

    def get_all_rays(self, i):
        data = trimesh_ray_tracing(self.camera_poses[i], resolution_x=self.resolution_x, resolution_y=self.resolution_y, fov=self.fov)
        return data

def spiral(scale=1):
    u = np.linspace(np.pi, 3 * np.pi, 200)
    x = np.sin(u) * 2
    y = np.cos(u) * 2
    z = -.60 * np.ones_like(x)
    points = np.stack([x.flatten(), z.flatten(), y.flatten()], 1) * scale
    return points


def create_uniform_camera_poses(distance=2, target=np.array([0.1, -0.1, 0.1]), up=np.array([0.0, -1.0, 0])):
    frontvectors = spiral(distance)
    frontvectors = frontvectors - target[None, :]
    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontvectors[i],
                                                     target=target,
                                                     up=up).T)
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
        print (camera_pose)
    return np.stack(camera_poses, 0)

def trimesh_ray_tracing(M, resolution_x=800, resolution_y=800, fov=30):
    mesh = trimesh.Trimesh()
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    # np.linalg.inv(create_look_at(frontVector, np.zeros(3), np.array([0, 1, 0])))
    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]
    scene.camera.resolution = [resolution_y, resolution_x]
    scene.camera.fov = [fov * resolution_y / resolution_x, fov]
    origins, vectors, pixels = scene.camera_rays()
    # x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    origins_new = np.zeros((resolution_y, resolution_x, 3))
    vectors_new = np.zeros((resolution_y, resolution_x, 3))
    origins_new[pixels[:, 0], pixels[:, 1]] = origins
    vectors_new[pixels[:, 0], pixels[:, 1]] = vectors
    return origins_new.reshape((-1, 3)), vectors_new.reshape((-1, 3)), pixels


def get_camera_intrisics_for_all_images(manager, factor, up_sample=1):
    Ks = []
    for im in manager.images:
        camera_id = im.camera_id
        cam = manager.cameras[camera_id]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        # fx = fx #* up_sample
        # fy = fy #* up_sample
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= factor
        K[:2, :] *= up_sample
        Ks.append(K)
    return np.stack(Ks, axis=0)

def _load_colmap(root_fp: str, subject_id: str, factor: int = 1, up_sample=1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    # fx = fx #* up_sample
    # fy = fy #* up_sample
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor
    K[:2, :] *= up_sample

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]
    camera_ids = [imdata[k].camera_id for k in imdata]
    # # Switch from COLMAP (right, down, fwd) to Nerf (right, up, back) frame.
    # poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    # assert params is None, "Only support pinhole camera model."

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camera_ids = [int(camera_ids[i]) for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    if factor > 1:
        image_dir_suffix = f"_{factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]
    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    # resize images if the number of camera's are more than 1
    camera_ids = np.array(camera_ids)
    valid_index = np.where(camera_ids == 1)[0][0]

    if len(manager.cameras) > 1:
        # NOTE: This is assuming the two or more cameras are very similar
        print ("Resizing images")
        # Use camera intrinsics to resize images.
        # Take care of the principal point offset.
        # Use the first image as reference.
        # Resize the images such that the principal point is at the center.
        width = images[valid_index].shape[1]
        height = images[valid_index].shape[0]

        for index, image in enumerate(images):
            image = np.array(image)
            if image.shape[0] != height or image.shape[1] != width:
                print ("Before resize: ", image.shape)
                image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
                print ("After resize: ", image.shape)
                images[index] = image
    images = np.stack(images, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])
    if not "green_stuffie" in subject_id:
        print ("Splitting dataset")
        split_indices = {
            "test": all_indices[all_indices % 8 == 0],
            "train": all_indices[all_indices % 8 != 0],
        }
    else:
        print ("splitting dataset for green stuffie")
        split_indices = {
            "test": all_indices[all_indices % 8 == 0],
            "train": all_indices,
        }
    # save camtoworlds
    # np.save(os.path.join(data_dir, "camtoworlds.npy"), camtoworlds)
    # np.save(os.path.join(data_dir, "K.npy"), K)
    # np.save(os.path.join(data_dir, "split_indices.npy"), split_indices)
    # np.save(os.path.join(data_dir, "images.npy"), images)
    # np.save(os.path.join(data_dir, "image_names.npy"), image_names)
    return images, camtoworlds, K, split_indices


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
    ]

    OPENGL_CAMERA = False

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
        factor: int = 1,
        device: str = "cpu",
        mesh_intersect=None,
        add_ray_direction_noise=False,
        upsample=1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.mesh_intersect = mesh_intersect
        self.add_ray_direction_noise = add_ray_direction_noise
        self.upsample = upsample
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.K, split_indices = _load_colmap(
            root_fp, subject_id, factor, up_sample=self.upsample
        )
        # normalize the scene
        T, sscale = similarity_from_cameras(
            self.camtoworlds, strict_scaling=False
        )
        self.camtoworlds = np.einsum("nij, ki -> nkj", self.camtoworlds, T)
        self.camtoworlds[:, :3, 3] *= sscale
        # split
        # np.save(os.path.join("{}/{}/".format(root_fp, subject_id), "camtoworlds.npy"), self.camtoworlds)
        # np.save(os.path.join("{}/{}/".format(root_fp, subject_id), "K.npy"), self.K)
        # np.save(os.path.join("{}/{}/".format(root_fp, subject_id), "split_indices.npy"), split_indices)
        # np.save(os.path.join("{}/{}/".format(root_fp, subject_id), "images.npy"), self.images)
        # np.save(os.path.join("{}/{}/".format(root_fp, subject_id), "image_names.npy"), split_indices)
        indices = split_indices[split]
        self.images = self.images[indices]
        self.camtoworlds = self.camtoworlds[indices]
        # to tensor
        self.images = torch.from_numpy(self.images).to(torch.uint8).to(device)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(torch.float32).to(device)
        )
        self.K = torch.tensor(self.K).to(torch.float32).to(device)
        self.height, self.width = self.images.shape[1:3]
        self.height = int(self.height * self.upsample)
        self.width = int(self.width * self.upsample)
        print ("Image shape: ", self.images.shape)
        print ("Camera to world shape: ", self.camtoworlds.shape)
        print ("K shape: ", self.K.shape)
        print ("Height: ", self.height)
        print ("Width: ", self.width)
    def __len__(self):
        return 100000000

    @torch.no_grad()
    def __getitem__(self, index):
        index = index % len(self.images)
        data = self.fetch_data(index)
        data = self.preprocess(data)
        if self.mesh_intersect is not None:
            output = self.mesh_intersect.sampling_raytrace_numpy(
                data["rays"].viewdirs.reshape((-1, 3)).data.cpu().numpy(), \
                data["rays"].origins.reshape((-1, 3)).data.cpu().numpy(), \
                0)
            if output is None:
                return None
            xyzs, dirs, index_ray, ts, index_tri, _, origins = output
            data["data"] = [torch.from_numpy(xyzs.astype(np.float32)), torch.from_numpy(dirs.astype(np.float32)), torch.from_numpy(index_ray.astype(np.int64)), torch.from_numpy(ts.astype(np.float32)),
                            torch.from_numpy(index_tri.astype(np.int64)), torch.from_numpy(origins.astype(np.float32))]
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

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

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
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
                0, self.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        # rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        # generate rays
        if self.upsample != 1 and not self.training:
            # for testing we want the image to be in original resolution
            x_, y_ = torch.meshgrid(
            torch.arange(self.width // self.upsample, device=self.images.device),
            torch.arange(self.height // self.upsample, device=self.images.device),
            indexing="xy",
            )
            x_ = x_.flatten()
            y_ = y_.flatten()
            # We want the image to be in original resolution for evaluation
            rgb = self.images[image_id, y_, x_] / 255.0  # (num_rays, 4)
        else:
            # for training we want the image to be in upsampled resolution
            rgb = self.images[image_id, torch.floor(y / self.upsample).long(), torch.floor(x / self.upsample).long()] / 255.0  # (num_rays, 4)

        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
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

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height // self.upsample, self.width // self.upsample, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }



class SubjectLoaderOwnViews(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
            self,
            resolution_x=800,
            resolution_y=800,
            fov=30,
            scale=1,
            target=np.array([0.1, -0.1, 0.1]),
            up=np.array([0.0, -1.0, 0]),
            mesh_intersect=None,
    ):
        super().__init__()
        self.mesh_intersect = mesh_intersect

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.width = resolution_x
        self.height = resolution_y
        self.camera_poses = CameraPose(resolution_x=resolution_x, resolution_y=resolution_y, fov=fov, scale=scale, target=target, up=up)
        self.len_images = self.camera_poses.camera_poses.shape[0]

    def __len__(self):
        return self.len_images

    @torch.no_grad()
    def __getitem__(self, index):
        index = index % self.len_images
        origins_original, vectors, pixels = self.camera_poses.get_all_rays(index)

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
