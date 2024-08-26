import numpy as np
import torch
import numpy as np
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

import imageio
import matplotlib.pyplot as plt
from radiance_fields.ngp import compress_polar_coordinates_torch, compress_colors, \
    compress_lambda_torch, torch_invserse_of_compressed_lambda, \
    inverse_of_azimuth_and_elevantion_torch, \
    inverse_of_compressed_colors
from utils import inverse_of_compressed_sigma


class FeatureCompression:
    def __init__(self, num_lobes, initialize=False, texture_size=None, path=None, compression_type="sigmoid", lambda_thres=7.5):
        import imageio.v2 as imageio
        # construct raw texture map
        self.num_lobes = num_lobes
        self.texture_size = texture_size
        self.compression_type = compression_type
        self.lambda_thres = lambda_thres

        if initialize:
            self.alpha = np.zeros((texture_size, texture_size), dtype=np.uint8) * 255
            # Initialized to zeros to be black
            self.diffuse = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
            self.sg_colors = {}
            self.lambdas = {}
            for i in range(self.num_lobes):
                self.sg_colors[i] = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
                self.lambdas[i] = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
            # convert all variables to torch
            self.alpha = torch.from_numpy(self.alpha).cuda()
            self.diffuse = torch.from_numpy(self.diffuse).cuda()
            self.sg_colors = {k: torch.from_numpy(v).cuda() for k, v in self.sg_colors.items()}
            self.lambdas = {k: torch.from_numpy(v).cuda() for k, v in self.lambdas.items()}
        else:
            # load from file
            self.alpha = torch.from_numpy(imageio.imread(path + "alpha.png")).cuda()
            self.diffuse = torch.from_numpy(imageio.imread(path + "diffuse.png")).cuda()
            self.num_lobes = num_lobes
            self.sg_colors = {}
            self.lambdas = {}
            for i in range(self.num_lobes):
                self.sg_colors[i] = torch.from_numpy(imageio.imread(path + "color_{}.png".format(i))).cuda()
                self.lambdas[i] = torch.from_numpy(imageio.imread(path + "lambda_axis_{}.png".format(i))).cuda()

    def compress_sigma(self, sigma):
        alpha = (1 - torch.exp(-sigma * 0.005))
        alpha = torch.clip(alpha * 255, 0, 255)
        alpha = alpha.to(torch.uint8)
        return alpha

    def sigma_to_alpha(sigma):
        alpha = (1 - np.exp(-sigma * 0.005))
        return alpha

    def inverse_of_compressed_sigma(self, alpha):
        alpha = alpha.to(torch.float32) / 255.0

        alpha = -torch.log(torch.clip(1 - alpha, 1e-6)) / 0.005
        return alpha
        
    def compress(self, features):
        """
        Compresses the features to a smaller size
        """
        N = features.shape[0]
        sigma = features[:, -1]
        alpha = self.compress_sigma(sigma)
        diffuse = compress_colors(features[..., :3], compress_type=self.compression_type)
        features = features[..., 3:-1]
        lobes = torch.reshape(features, (N, self.num_lobes, 7))
        
        axes = lobes[..., :3]
        azimuth, elevation = compress_polar_coordinates_torch(axes)
        
        lambdas = torch.abs(lobes[..., 3])
        compressed_lambda = compress_lambda_torch(lambdas, self.lambda_thres)
        c = lobes[..., 4:]
        data = {}
        data["lambdas"] = []
        data["colors"] = []
        data["alpha"] = alpha
        data["diffuse"] = diffuse

        for i in range(self.num_lobes):
            lambda_axis = torch.stack([
                compressed_lambda[..., i],
                azimuth[..., i],
                elevation[..., i]
            ], axis=-1)
            data["lambdas"].append(lambda_axis)
            data["colors"].append(compress_colors(c[..., i, :], compress_type=self.compression_type))
        return data

    def assign_values_to_texture_map(self, features, indices):
        data = self.compress(features)
        for i in range(self.num_lobes):
            self.lambdas[i][indices[:, 0], indices[:, 1]] = data["lambdas"][i]
            self.sg_colors[i][indices[:, 0], indices[:, 1]] = data["colors"][i]
        self.alpha[indices[:, 0], indices[:, 1]] = data["alpha"]
        self.diffuse[indices[:, 0], indices[:, 1]] = data["diffuse"]

    def compress_features_and_save(self, features, path):
        N = features.shape[0]
        features = features.reshape((N * N, -1))
        data = self.compress(features)

        for i in range(self.num_lobes):
            imageio.imsave(path + "color_{}.png".format(i), data["colors"][i].reshape((N, N, 3)).data.cpu().numpy())
            imageio.imsave(path + "lambda_axis_{}.png".format(i), data["lambdas"][i].reshape((N, N, 3)).data.cpu().numpy())
        imageio.imsave(path + "alpha.png", data["alpha"].data.cpu().numpy().reshape((N, N)))
        imageio.imsave(path + "diffuse.png", data["diffuse"].reshape((N, N, 3)).data.cpu().numpy())
    
    def save_to_file(self, path):
        imageio.imsave(path + "alpha.png", self.alpha.data.cpu().numpy())
        imageio.imsave(path + "diffuse.png", self.diffuse.data.cpu().numpy())
        for i in range(self.num_lobes):
            imageio.imsave(path + "color_{}.png".format(i), self.sg_colors[i].data.cpu().numpy())
            imageio.imsave(path + "lambda_axis_{}.png".format(i), self.lambdas[i].data.cpu().numpy())
    
    def spherical_gaussian(self, x, direction):
        axis = x[..., :3]
        # normalize axis
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        lambda_ = torch.abs(x[..., 3])
        c = x[..., 4:]
        return c * torch.exp(lambda_ * (torch.sum(axis * direction, -1) - 1))[..., None]

    def spherical_gaussian_mixture(self, x, direction):
        # split the x into self.num_g_lobes
        x = torch.chunk(x, self.num_lobes, dim=-1)
        rgb = torch.zeros((x[0].shape[0], 3)).cuda()
        for x_ in x:
            # compute the spherical gaussian
            rgb_ = self.spherical_gaussian(x_, direction)
            rgb = rgb + rgb_
        return rgb

    def features_to_rgb(self, features, dir):
        diffuse_color = features[:, :3]
        rgb = torch.sigmoid(diffuse_color + self.spherical_gaussian_mixture(features[:, 3:], dir))
        return rgb

    def get_features_from_texture_map(self, indices):
        sigma = self.inverse_of_compressed_sigma(self.alpha[indices[:, 0], indices[:, 1]])
        diffuse = inverse_of_compressed_colors(self.diffuse[indices[:, 0], indices[:, 1]], compress_type=self.compression_type)
        lambdas = []
        colors = []
        azimuths = []
        elevations = []
        for i in range(self.num_lobes):
            # NOTE that the lambdas and axis angles are stored in the same tensor
            shared_tensor = self.lambdas[i][indices[:, 0], indices[:, 1]]
            lam = torch_invserse_of_compressed_lambda(shared_tensor[:, 0], self.lambda_thres)
            lambdas.append(lam.squeeze())
            azimuths.append(shared_tensor[:, 1].squeeze())
            elevations.append(shared_tensor[:, 2].squeeze())
            colors.append(inverse_of_compressed_colors(self.sg_colors[i][indices[:, 0], indices[:, 1]], compress_type=self.compression_type))

        axis = []
        for i in range(self.num_lobes):
            axis.append(inverse_of_azimuth_and_elevantion_torch(azimuths[i], elevations[i]))
        features = torch.zeros((len(indices), 7 * self.num_lobes), dtype=torch.float32).cuda()
        for i in range(self.num_lobes):
            features[:, 7 * i: 7 * i + 3] = axis[i]
            features[:, 7 * i + 3] = lambdas[i]
            features[:, 7 * i + 4: 7 * (i + 1)] = colors[i]
        features = torch.cat([features, sigma.unsqueeze(1)], dim=-1)
        features = torch.cat([diffuse, features], dim=-1)
        return features

    def load_each_attribute(self, data):
        # collect all data
        lambdas = []
        colors = []
        azimuths = []
        elevations = []
        for i in range(self.num_lobes):
            lambdas.append(data["lambdas"][i][0])
            colors.append(data["colors"][i])
            azimuths.append(data["lambdas"][i][..., 1])
            elevations.append(data["lambdas"][i][..., 2])
        alpha = data["alpha"]
        diffuse = data["diffuse"]
        self.alpha = alpha
        self.diffuse = diffuse
        self.lambdas = lambdas
        self.sg_colors = colors
        self.azimuths = azimuths
        self.elevations = elevations

    def load_features_into_maps(self, features, indices):
        data = self.compress(features)
        self.alpha[indices[:, 0], indices[:, 1]] = data["alpha"]
        self.diffuse[indices[:, 0], indices[:, 1]] = data["diffuse"]
        for i in range(self.num_lobes):
            self.lambdas[i][indices[:, 0], indices[:, 1]] = data["lambdas"][i]
            self.sg_colors[i][indices[:, 0], indices[:, 1]] = data["colors"][i]
