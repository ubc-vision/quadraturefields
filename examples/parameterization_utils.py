import numpy as np
import torch
import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
import trimesh
import scipy
import skimage as ski


def sample_points_on_triangle(indices, mesh, n, uv):
    vertices = np.array(mesh.vertices[mesh.faces[indices]]).astype(np.float32)
    vertices = torch.from_numpy(vertices).cuda()
    uv_ = uv[mesh.faces[indices]]
    uvs = []
    vs = []
    for _ in range(n):
        weights = torch.rand((vertices.shape[0], 3), device=vertices.device)[..., None].cuda()
        v = torch.sum(vertices * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6)
        vs.append(v)
        uvs.append(torch.sum(uv_ * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6))
    return torch.cat(vs, dim=0), np.tile(indices, (n,)), torch.cat(uvs, 0)


def sample_points_on_triangle_parameterization(indices, mesh, n=3):
    vertices = np.array(mesh.vertices[mesh.faces[indices]]).astype(np.float32)
    vertices = torch.from_numpy(vertices).cuda()
    vs = []
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    z = 1 - x - y
    mask = z >= 0
    for i in range(n):
        for j in range(n):
            if z[i, j] < 0:
                continue
            weights = torch.zeros((vertices.shape[0], 3), device=vertices.device)[..., None].cuda()
            weights[:, 0] = x[i, j]
            weights[:, 1] = y[i, j]
            weights[:, 2] = z[i, j]
            v = torch.sum(vertices * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6)
            vs.append(v)
    return vs


def fill_triangles(mesh, SIZE=8192):
    uv = np.clip((mesh.visual.uv) * SIZE, 0, SIZE - 1)
    uv = uv.astype(np.int32)
    triangle_vertices = uv[mesh.faces]
    triangle_vertices = np.concatenate([triangle_vertices, np.zeros((triangle_vertices.shape[0], 3, 1))], axis=2)
    V = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for i in range(mesh.faces.shape[0]):
        rr, cc = ski.draw.polygon_perimeter(triangle_vertices[i, :, 0], triangle_vertices[i, :, 1])
        rr_all, cc_all = ski.draw.polygon(triangle_vertices[i, :, 0], triangle_vertices[i, :, 1])
        
        points = np.array([rr_all, cc_all]).T
        # add a z coordinate of 0 to the points
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        b_coords = trimesh.triangles.points_to_barycentric(np.repeat(triangle_vertices[i:i+1], points.shape[0], axis=0), points)
        verts_all = np.repeat(mesh.vertices[mesh.faces[i:i+1]], points.shape[0], axis=0)
        verts_all = b_coords[..., None] * verts_all
        verts_all = verts_all.sum(axis=1) # this is complete data
        
        points = np.array([rr, cc]).T
        # add a z coordinate of 0 to the points
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        b_coords = trimesh.triangles.points_to_barycentric(np.repeat(triangle_vertices[i:i+1], points.shape[0], axis=0), points)
        verts_boundary = np.repeat(mesh.vertices[mesh.faces[i:i+1]], points.shape[0], axis=0)
        verts_boundary = b_coords[..., None] * verts_boundary
        verts_boundary = verts_boundary.sum(axis=1) # this is complete data
        
        V[np.clip(rr - 1, 0, SIZE - 1), cc] = verts_boundary
        V[np.clip(rr + 1, 0, SIZE - 1), cc] = verts_boundary
        V[rr, np.clip(cc + 1, 0, SIZE - 1)] = verts_boundary
        V[rr, np.clip(cc - 1, 0, SIZE - 1)] = verts_boundary
        
        V[rr_all, cc_all] = verts_all
        if i % 10000 == 0:
            print("Filling the triangles: ", i)
    return V


def concatenate_meshes(meshes):
    vertices = []
    faces = []
    uvs = []
    count_vertices = 0
    for m in meshes:
        vertices.append(m.vertices)
        faces.append(m.faces + count_vertices)
        count_vertices += m.vertices.shape[0]
        uvs.append(m.visual.uv)
    mesh = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)
    mesh.visual.uv = np.concatenate(uvs)
    return mesh


def fill_triangles_fill_boundary(mesh, HEIGHT, WIDTH):
    uv = np.copy(mesh.visual.uv)
    uv[:, 0] *= (HEIGHT)
    uv[:, 1] *= (WIDTH)
    uv[:, 0] = np.clip(uv[:, 0], 0, HEIGHT - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, WIDTH - 1)
    ws = np.linspace(0, 1, 100)
    # uv = uv.astype(np.int32)
    triangle_vertices = uv[mesh.faces]
    triangle_vertices = np.concatenate([triangle_vertices, np.zeros((triangle_vertices.shape[0], 3, 1))], axis=2)
    triangle_vertices = triangle_vertices.astype(np.int32)
    V = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    tri_size = []
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    tri_id = np.ones((HEIGHT, WIDTH), dtype=np.int32) * -1
    import time
    tic = time.time()
    for i in range(mesh.faces.shape[0]):
        rr_all, cc_all = ski.draw.polygon(triangle_vertices[i, :, 0], triangle_vertices[i, :, 1])
        coords = np.stack([triangle_vertices[i, :, 0], triangle_vertices[i, :, 1]], axis=1)
        # generate points on the boundary
        vertices = np.array(mesh.vertices[mesh.faces[i]]).astype(np.float32)

        points = np.array([rr_all, cc_all]).T
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        b_coords = trimesh.triangles.points_to_barycentric(np.repeat(triangle_vertices[i:i+1], points.shape[0], axis=0), points)
        if np.isnan(b_coords).any():
            b_coords = np.random.random(b_coords.shape)
            b_coords = b_coords / b_coords.sum(axis=1)[..., None]
        if np.isnan(b_coords).any():
            import ipdb; ipdb.set_trace()
        verts_all = np.repeat(mesh.vertices[mesh.faces[i:i+1]], points.shape[0], axis=0)
        verts_all = b_coords[..., None] * verts_all
        verts_all = verts_all.sum(axis=1) # this is complete data
        
        V[rr_all, cc_all] = verts_all
        tri_size.append(rr_all.shape[0])
        if i % 20000 == 0:
            print("Filling the triangles: ", i, rr_all.shape, rr_all.shape, time.time() - tic)
            tic = time.time()
        mask[rr_all, cc_all] = True
        tri_id[rr_all, cc_all] = i
        coords = uv[mesh.faces[i]]

        # this step samples points along the triangle edge
        # and assigns the tri_id to those pixels
        l1 = (coords[1] * ws[:, None] + coords[0] * (1 - ws[:, None])).astype(np.int32)
        l2 = (coords[2] * ws[:, None] + coords[1] * (1 - ws[:, None])).astype(np.int32)
        l3 = (coords[0] * ws[:, None] + coords[2] * (1 - ws[:, None])).astype(np.int32)
        tri_id[l3[:, 0], l3[:, 1]] = i
        tri_id[l2[:, 0], l2[:, 1]] = i
        tri_id[l1[:, 0], l1[:, 1]] = i

    with torch.no_grad():
        mask = ~mask
        V[mask > 0] = mesh.vertices[mesh.faces[tri_id[mask>0]]].mean(1)
    return V, tri_size
