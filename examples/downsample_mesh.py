import trimesh
import numpy as np
import sys
import os
from open3d import *
import open3d as o3d

mesh_path = sys.argv[1]
voxel_size = int(sys.argv[2])
meshname=mesh_path.split("/")[-1][0:-4]
mesh = o3d.io.read_triangle_mesh(mesh_path)
print ("Before mesh simplification: ", np.array(mesh.vertices).shape, np.array(mesh.triangles).shape)
mesh_smp = mesh.simplify_vertex_clustering(1 / voxel_size, contraction=o3d.geometry.SimplificationContraction.Quadric)

print("After mesh simplification: ", np.array(mesh_smp.vertices).shape, np.array(mesh_smp.triangles).shape)
mesh = trimesh.Trimesh(vertices=np.array(mesh_smp.vertices),
                            faces=np.array(mesh_smp.triangles))

output_path = "/".join(mesh_path.split("/")[:-1]) + "/smp_{}".format(meshname) + ".ply"
mesh.export(output_path)
