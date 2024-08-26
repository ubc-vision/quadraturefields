"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

import torch
from torch import nn
import tinycudann as tcnn
# import vren
from einops import rearrange
import numpy as np


scaler = torch.cuda.amp.GradScaler()


class BasicDecoder(torch.nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation,
        bias,
        layer = nn.Linear,
        num_layers = 1, 
        hidden_dim = 128, 
        skip       = []
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)


    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)

    def name(self) -> str:
        """ A human readable name for the given wisp module. """
        return "BasicDecoder"


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

class _contract_to_unisphere(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, aabb):  # pylint: disable=arguments-differ
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        ctx.save_for_backward(x.clone(), mag, mask)
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        mag = ctx.saved_tensors[1]
        mask = ctx.saved_tensors[2]
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=1e-6)
        return g * dev, None

trunc_exp = _TruncExp.apply
contract_to_unisphere = _contract_to_unisphere.apply

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
    invalid = mag > 2
    x[mask] = 1 / (2 - mag[mask]) * x[mask] / mag[mask]
    x = (x + 1) / 2
    x = x * (aabb_max - aabb_min) + aabb_min
    return x, invalid


def discretize_axis(axis):
    axis = (axis + 1.0) * 255 / 2
    axis = axis.to(torch.uint8)
    return axis

def continuous_axis(axis):
    axis = axis.to(torch.float32) / 255.0 * 2 - 1
    return axis

def discretize_color(color):
    color = torch.sigmoid(color)
    color = color * 255
    color = color.to(torch.uint8)
    return color

def continuous_color(color):
    color = color.to(torch.float32) / 255.0
    color = torch.log(torch.clip(color / (1 - color), 1e-8, 1e37))
    return color

def discritize_lambda(l):
    l = l.to(torch.float16)
    return l

def continuous_lambda(l):
    l = l.to(torch.float32)
    return l


def compress_polar_coordinates_torch(vectors):
    vectors = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-6)
    azimuth = (torch.atan2(vectors[..., 1], vectors[..., 0]) * 128 / np.pi + 128).to(torch.uint8)
    elevation = (torch.acos(vectors[..., 2]) * 256 / np.pi).to(torch.uint8)
    return azimuth, elevation

def inverse_of_azimuth_and_elevantion_torch(azimuth, elevation):
    azimuth = (azimuth - 128) / 128 * np.pi
    # azimuth = torch.tan(azimuth)
    elevation = elevation / 256 * np.pi
    x = torch.cos(azimuth) * torch.sin(elevation)
    y = torch.sin(azimuth) * torch.sin(elevation)
    z = torch.cos(elevation)
    return torch.stack([x, y, z], dim=-1)

def compress_lambda_torch(lambdas, compress_threshold=7.5):
    log_lambda = torch.log(torch.clamp(lambdas, 1e-5, np.inf))
    compressed_lambda = torch.clamp((log_lambda + 2.5) / compress_threshold, 0.0, 1.0)
    compressed_lambda = (255 * compressed_lambda).to(torch.uint8)
    return compressed_lambda

def torch_invserse_of_compressed_lambda(compressed_lambda, compress_threshold=7.5):
    log_lambda = compressed_lambda * compress_threshold / 255 - 2.5
    return torch.exp(log_lambda)

def compress_colors(colors, thres=12, compress_type="sigma"):
    if compress_type == "sigma":
        colors = torch.sigmoid(colors)
    else:
        colors = torch.clip(colors, -thres, thres)
        colors = (colors + thres) / 2 / thres

    colors = colors * 255
    colors = colors.to(torch.uint8)
    return colors

def inverse_of_compressed_colors(colors, thres=12, compress_type="sigma"):
    colors = colors.to(torch.float32) / 255.0
    if compress_type == "sigma":
        colors = torch.log(torch.clip(colors / (1 - colors), 1e-8, 1e37))
    else:
        colors = colors * 2 * thres - thres
    return colors


class NGPRadianceFieldSGNew(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
            self,
            aabb: Union[torch.Tensor, List[float]],
            num_dim: int = 3,
            use_viewdirs: bool = True,
            density_activation: Callable = lambda x: trunc_exp(x - 1),
            unbounded: bool = False,
            base_resolution: int = 16,
            max_resolution: int = 4096,
            geo_feat_dim: int = 15,
            n_levels: int = 16,
            log2_hashmap_size: int = 19,
            num_g_lobes=3,
            hidden_size=64,
            num_layers=2,
            output_activation="sigmoid",
            discretize=False,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.num_g_lobes = num_g_lobes
        self.output_activation = output_activation
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()
        self.discretize = discretize
        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = BasicDecoder(
                input_dim=(self.direction_encoding.n_output_dims
                           if self.use_viewdirs
                           else 0
                           ) + self.geo_feat_dim,
                output_dim=3 + num_g_lobes * (3 + 3 + 1),
                num_layers=num_layers,
                bias=True,
                activation=torch.nn.ReLU(),
                hidden_dim=hidden_size)

    def spherical_gaussian(self, x, direction):
        axis = x[..., :3]
        # normalize axis
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        lambda_ = torch.abs(x[..., 3])
        c = x[..., 4:]
        if self.discretize:
            azimuth, elevation = compress_polar_coordinates_torch(axis)
            axis = inverse_of_azimuth_and_elevantion_torch(azimuth, elevation)
            compressed_lambda = compress_lambda_torch(lambda_)
            lambda_ = torch_invserse_of_compressed_lambda(compressed_lambda)
            c = inverse_of_compressed_colors(compress_colors(c))
        return c * torch.exp(lambda_ * (torch.sum(axis * direction, -1) - 1))[..., None]

    def spherical_gaussian_mixture(self, x, direction):
        # split the x into self.num_g_lobes
        x = torch.chunk(x, self.num_g_lobes, dim=-1)
        rgb = torch.zeros((x[0].shape[0], 3)).cuda()
        for x_ in x:
            # compute the spherical gaussian
            rgb_ = self.spherical_gaussian(x_, direction)
            rgb = rgb + rgb_
        return rgb

    def normalize(self, x):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        return selector, x

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
                self.density_activation(density_before_activation)
                * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [self.num_g_lobes * (3 + 3 + 1) + 3])
            .to(embedding)
        )
        diffuse_color = rgb[:, :3]
        rgb = torch.sigmoid(diffuse_color + self.spherical_gaussian_mixture(rgb[:, 3:], dir))
        return rgb

    def features(self, x):
        density, embedding = self.query_density(x, return_feat=True)
        h = embedding.reshape(-1, self.geo_feat_dim)
        features = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [self.num_g_lobes * (3 + 3 + 1) + 3])
            .to(embedding)
        )
        features = torch.cat([features, density], dim=-1)
        return features

    def features_to_rgb(self, features, dir):
        diffuse_color = features[:, :3]
        if self.discretize:
            diffuse_color = inverse_of_compressed_colors(compress_colors(diffuse_color))
        rgb = torch.sigmoid(diffuse_color + self.spherical_gaussian_mixture(features[:, 3:], dir))
        return rgb

    def forward(
            self,
            positions: torch.Tensor,
            directions: torch.Tensor = None,
    ):
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


class NGPRadianceFieldSG(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
            self,
            aabb: Union[torch.Tensor, List[float]],
            num_dim: int = 3,
            use_viewdirs: bool = True,
            density_activation: Callable = lambda x: trunc_exp(x - 1),
            unbounded: bool = False,
            base_resolution: int = 16,
            max_resolution: int = 4096,
            geo_feat_dim: int = 15,
            n_levels: int = 16,
            log2_hashmap_size: int = 19,
            num_g_lobes=3,
            hidden_size=64,
            num_layers=2,
            output_activation="sigmoid"
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.num_g_lobes = num_g_lobes
        self.output_activation = output_activation
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = BasicDecoder(
                            input_dim=(self.direction_encoding.n_output_dims
                            if self.use_viewdirs
                            else 0
                        ) + self.geo_feat_dim, 
                        output_dim=3 + num_g_lobes * 3 * (3 + 1 + 1),
                        num_layers=num_layers,
                        bias=True,
                        activation=torch.nn.ReLU(),
                        hidden_dim=hidden_size)

    def spherical_gaussian(self, x, direction):
        axis = x[..., :3]
        # normalize axis
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        lambda_ = torch.abs(x[..., 3])
        a = torch.abs(x[..., 4])
        return a * torch.exp(-lambda_ * (1 - torch.sum(axis * direction, -1)))

    def spherical_gaussian_mixture(self, x, direction):
        # TODO this can be made faster
        # TODO the amplitude of each gaussian should be normalized
        # split the x into self.num_g_lobes
        x = torch.chunk(x, self.num_g_lobes, dim=-1)
        rgb = torch.zeros((x[0].shape[0], 3)).cuda()
        for x_ in x:
            # split x_ into 3 parts corresponding to RGB
            x_ = torch.chunk(x_, 3, dim=-1)
            # compute the spherical gaussian
            rgb_ = torch.cat([self.spherical_gaussian(x_[i], direction).unsqueeze(1) for i in range(3)], dim=-1)
            rgb = rgb + rgb_
        return rgb

    def normalize(self, x):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        return selector, x

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
                self.density_activation(density_before_activation)
                * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [self.num_g_lobes * 3 * (3 + 1 + 1) + 3])
            .to(embedding)
        )
        diffuse_color = rgb[:, :3]
        rgb = torch.sigmoid(diffuse_color + self.spherical_gaussian_mixture(rgb[:, 3:], dir))
        return rgb
    
    def features(self, x):
        density, embedding = self.query_density(x, return_feat=True)
        h = embedding.reshape(-1, self.geo_feat_dim)
        features = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [self.num_g_lobes * 3 * (3 + 1 + 1) + 3])
            .to(embedding)
        )
        features = torch.cat([features, density], dim=-1)
        return features

    def features_to_rgb(self, features, dir):
        diffuse_color = features[:, :3]
        rgb = torch.sigmoid(diffuse_color + self.spherical_gaussian_mixture(features[:, 3:], dir))
        return rgb
    
    def forward(
            self,
            positions: torch.Tensor,
            directions: torch.Tensor = None,
    ):
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        num_layers=2,
        hidden_size=64,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_size,
                    "n_hidden_layers": 2,
                },
            )
    
    def normalize(self, x):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        return selector, x
    
    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


class NGPDensityField(torch.nn.Module):
    """Instance-NGP Density Field used for resampling"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

    def forward(self, positions: torch.Tensor):
        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        density_before_activation = (
            self.mlp_base(positions.view(-1, self.num_dim))
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        return density
    
    def density(self, positions):
        # assuming the positions are already normalized to [0, 1]
        density_before_activation = (
            self.mlp_base(positions.view(-1, self.num_dim))
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = (
            self.density_activation(density_before_activation)
        )
        return density
