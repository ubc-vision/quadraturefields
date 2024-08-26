import torch
import numpy as np
from matplotlib import pyplot as plt
import imageio
import trimesh
from skimage import measure
import numpy as np
from scipy.ndimage import zoom
import trimesh
import sys
from field_utils import GaussianSmoothing

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

root = sys.argv[1]
sigma = float(sys.argv[2])
include_grad = True if sys.argv[3] == "True" else False
omega = float(sys.argv[4])
thres = float(sys.argv[5])
axis = int(sys.argv[6])
combine = True if sys.argv[7] == "True" else False
grad_thres = float(sys.argv[8])
density_thres = float(sys.argv[9])

print("Running marching cubes!!")
print("root: {}, sigma: {}, include_grad: {}, omega: {}, thres: {}".format(root, sigma, include_grad, omega, thres, grad_thres))

kernel = GaussianSmoothing(channels=1, kernel_size=5, sigma=sigma, dim=3).cuda()
m = torch.nn.Upsample((1024, 1024, 1024), mode="trilinear", align_corners=True).cuda()

grid = np.load(root + "grids_valid.npy")
grads = np.load(root + "grads_valid.npy")

density = np.load("{}binaries.npy".format(root))[0]
grid = torch.from_numpy(grid).float().cuda()
grads = torch.from_numpy(grads).float().cuda()
grid = kernel(grid[None, None, ...])[0, 0, ...]

with torch.no_grad():
    d = torch.from_numpy(density).float().cuda()
    d = m(d[None, None, ...])[0, 0, ...]

    min = (grid * d).min()
    grid.sub_(min)
    max = (grid * d).max()
    grid.div_(max + 1e-6)
    grid.sub_(0.5).mul_(2)

    quantity = grid * d
    temp = grads > grad_thres
    temp_lower = grads > grad_thres
    del grads
    if include_grad:
        quantity = quantity * (temp)

    quantity = quantity[None, None, ...]

if combine:
    nerf_density = np.load("{}density_grids_valid.npy".format(root))
    # uncomment the following to get sparser mesh
    # nerf_density = nerf_density * temp_lower.data.cpu().numpy()

    verts, faces, normals, values = measure.marching_cubes(nerf_density, level=density_thres)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.vertices = mesh.vertices / (nerf_density.shape[0] - 1)
    mesh.vertices = (mesh.vertices - 0.5) * 2
    print("Density mesh: Faces ", mesh.faces.shape, "Vertices: ", mesh.vertices.shape)
    mesh = mesh.export("{}/mesh_nerf.ply".format(root))

quantity = quantity.data.cpu().numpy()[0, 0, ...]
print(quantity.shape, thres)

verts, faces, normals, values = measure.marching_cubes(np.sin(omega * quantity), level=thres)

mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.vertices = mesh.vertices / (quantity.shape[0] - 1)
mesh.vertices = (mesh.vertices - 0.5) * 2
print("Combined mesh: Faces ", mesh.faces.shape, "Vertices: ", mesh.vertices.shape)

mesh = trimesh.util.concatenate([mesh, trimesh.load("{}/mesh_nerf.ply".format(root))])
mesh = mesh.export("{}/mesh.ply".format(root))
