import torch
import trimesh
import xatlas
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
from parameterization_utils import fill_triangles_fill_boundary, concatenate_meshes
import sys
import os
import time
from radiance_fields.ngp import contract_to_unisphere, inverse_contraction

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

root_path = sys.argv[1]
mesh_name = sys.argv[2]
labels_name = sys.argv[3]
texture_size = int(sys.argv[4])
contraction = str2bool(sys.argv[5])

mesh = trimesh.load("{}/{}".format(root_path, mesh_name), process=False)

# triangle_weights = np.load("{}/{}".format(root_path, "/triangle_weights.npy"))

if contraction:
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device="cuda")

    vertices = mesh.vertices.astype(np.float32)
    vertices = torch.from_numpy(vertices).cuda()
    with torch.no_grad():
        vertices = contract_to_unisphere(vertices, aabb)
    vertices = vertices.data.cpu().numpy()
    mesh.vertices = vertices

print(mesh)

with open("{}/{}".format(root_path, labels_name), "r") as file:
    data = json.load(file)
    array = np.array(data["segIndices"])[mesh.faces]
    array = torch.from_numpy(array).cuda()
    array = torch.mode(array, 1, keepdim=False).values
    closest_indices = array.data.cpu().numpy()

print(np.unique(closest_indices), mesh.faces.shape)
meshes = []
atlas = xatlas.Atlas()

uniques = np.unique(closest_indices)
tic = time.time()
given_weights = []
for index, i in enumerate(uniques):
    t1 = time.time()
    valid_indices = np.where(closest_indices == i)
    new_mesh = mesh.submesh(valid_indices, append=True)
    indices = new_mesh.faces
    uvs = None
    # weight = triangle_weights[valid_indices].mean() + 0.5
    # given_weights.append(weight)
    new_mesh.vertices =  new_mesh.vertices # * weight
    atlas.add_mesh(positions=new_mesh.vertices, indices=indices, uvs=uvs)
    meshes.append(new_mesh)
    if index % 200 == 0:
        print(index, i, time.time() - tic)
        tic = time.time()

ratio = 0.6
while True:
    print("Generating atlas...")
    start = time.time()
    copt = xatlas.ChartOptions()
    copt.normal_deviation_weight = 0
    copt.max_iterations = 100
    copt.use_input_mesh_uvs = False
    copt.roundness_weight = 0.001
    copt.straightness_weight = 0.001
    copt.normal_seam_weight = 0.001
    copt.max_cost = 10000

    popt = xatlas.PackOptions()
    popt.padding = 0
    popt.create_image = True
    popt.resolution = int(texture_size * ratio)
    popt.bilinear = True
    popt.bruteForce = False
    popt.blockAlign = False
    atlas.generate(copt, popt)
    print("Done generating atlas...", time.time() - start)
    if atlas.height > texture_size:
        ratio = ratio - 0.1
    else:
        break

os.makedirs(root_path + "/{}".format(mesh_name.split(".")[0]), exist_ok=True)
print(atlas.width, atlas.height, atlas.utilization)
plt.imsave(root_path + "/{}".format(mesh_name.split(".")[0]) + "/atlas_{}.png".format(texture_size), atlas.get_chart_image(0))

# transfer the new uvs to the meshes
for i, item in enumerate(atlas):
    new_mesh = meshes[i]
    vmapping, indices, uvs = item
    new_mesh.vertices = new_mesh.vertices[vmapping] # / (given_weights[i])
    new_mesh.faces = indices
    new_mesh.visual.uv = uvs

mesh = concatenate_meshes(meshes)

if contraction:
    vertices = mesh.vertices.astype(np.float32)
    vertices = torch.from_numpy(vertices).cuda()
    with torch.no_grad():
        vertices, invalids = inverse_contraction(vertices, aabb)
    if invalids.sum() > 0:
        print("Invalids", invalids.sum())
        import ipdb; ipdb.set_trace()
    vertices = vertices.data.cpu().numpy()
    mesh.vertices = vertices

_ = xatlas.export(root_path + "/{}".format(mesh_name.split(".")[0]) + "/mesh_segmentation_{}.obj".format(texture_size), mesh.vertices,
                  mesh.faces, mesh.visual.uv)
print(atlas.width, atlas.height, atlas.utilization)
if texture_size == -1:
    SIZE = (max(atlas.width, atlas.height))
else:
    SIZE = texture_size

print(SIZE)
mesh = trimesh.load(root_path + "/{}".format(mesh_name.split(".")[0]) + "/mesh_segmentation_{}.obj".format(texture_size), process=False)
print(mesh)

V, tri_size = fill_triangles_fill_boundary(mesh, SIZE, SIZE)

if texture_size > 8192:
    np.save(root_path + "/{}".format(mesh_name.split(".")[0]) + "/V_{}.npy".format(texture_size), V.astype(np.float16))
else:
    np.save(root_path + "/{}".format(mesh_name.split(".")[0]) + "/V_{}.npy".format(texture_size), V.astype(np.float32))
