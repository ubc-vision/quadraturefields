import torch
from torch import nn
import tinycudann as tcnn
# import vren
from einops import rearrange
import numpy as np
# from get_positional_embedder import get_positional_embedder
import trimesh

scaler = torch.cuda.amp.GradScaler()

@torch.jit.script
def gaussian(x):
    return 0.5 * x * (1 + torch.nn.functional.tanh((2 / 3.142) ** (0.5) * (x + 0.044715 * x ** 3)))


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
        skip       = [],
        bias_last=True
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
        self.bias_last = bias_last
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
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias_last)


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



class Field(nn.Module):
    def __init__(self, scale, back_prop=0, precision=16, log2_T=19, L=16, max_res=512, output_dim=1, min_res=16, hidden_size=32, num_features=2, nl="elu", bias=True, bias_last=True):
        super().__init__()
        # scene bounding box
        self.output_dim = output_dim
        if precision == 16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)
        self.back_prop = back_prop
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        # self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        # self.grid_size = 256
        # self.register_buffer('density_bitfield',
        # torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        F = num_features;  N_min = min_res
        # TODO Look into the max resolution
        b = np.exp(np.log(max_res*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                dtype=self.dtype
            )
        if nl == "elu":
            activation = torch.nn.ELU()
        elif nl == "relu":
            activation = torch.nn.ReLU()

        self.decoder_field = BasicDecoder(input_dim=L * num_features + 3,
                                output_dim=output_dim,
                                activation=activation,
                                bias=bias,
                                num_layers=2,
                                hidden_dim=hidden_size,
                                skip=[],
                              bias_last=bias_last)

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        if self.back_prop:
            h = self.xyz_encoder(x)
        else:
            h = self.xyz_encoder(x.detach())
        sigmas = self.decoder_field(torch.cat([x, h], 1))
        # sigmas = TruncExp.apply(sigmas)
        # if return_feat: return sigmas, h
        return sigmas


    def forward(self, x, return_grad=True):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        if not x.requires_grad:
            x.requires_grad = True
        field = self.field(x)
        if return_grad:
            field_grad = self.field_grad(x, field, create_graph=True)
        else:
            field_grad = None
        return field, field_grad

    def field(self, x):
        field = self.density(x)[:, 0:self.output_dim]
        return field

    def field_grad(self, coords, field, create_graph=True):
        field = field.flatten()
        grad_outputs = torch.ones_like(field)

        # try:
        field_grad = torch.autograd.grad(field, [coords], grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=True)[0]
        # except:
        #     print ("Error in the grad!")
        #     field_grad = [torch.zeros((field.shape[0], 3)).cuda()]
        return field_grad

    def field_double_grad(self, coords, field_grad, create_graph=True):
        coords.requires_grad = True
        # import ipdb; ipdb.set_trace()
        grad_outputs = torch.ones((field_grad.shape[0], 1)).cuda()
        second_derivative = torch.zeros((field_grad.shape[0], 3, 3)).cuda()

        for i in range(3):
            # TODO need to make sure that the axis is correct
            second_derivative[:, i] = torch.autograd.grad(field_grad[:, i:i+1], [coords], grad_outputs=grad_outputs,
                                                          create_graph=create_graph, allow_unused=False)[0]
        # print (second_derivative.shape)
        return second_derivative
    
    def compute_field_loss(self, weights, weights_rev, field_norm, view_dirs):
        view_dirs = view_dirs/torch.norm(view_dirs, dim=1, keepdim=True)
        field_norm = field_norm #+ torch.rand_like(field_norm) * 0.001
        field_loss = torch.abs(torch.maximum(weights.detach(), weights_rev.detach()) - \
                                    torch.abs(torch.sum(field_norm * view_dirs.detach(), 1)))
        field_loss = field_loss.mean()
        return field_loss

    def compute_abs_loss(self, field_norm):
        l1_norm = torch.linalg.norm(field_norm, ord=1, dim=1)
        abs_loss = l1_norm.mean()
        return abs_loss
    
    def compute_double_field_loss(self, field_double_der):

        field_loss = torch.abs(field_double_der)
        field_loss = field_loss.mean()
        return field_loss

# from siren import SirenNet

class FieldMLP(nn.Module):
    def __init__(self, scale, back_prop=0, precision=16, log2_T=19, L=16, max_res=512, output_dim=1, min_res=16,hidden_size=256, num_features=None, nl=None, bias=True, bias_last=True):
        super().__init__()
        # scene bounding box
        self.output_dim = output_dim
        # if precision == 16:
        #     self.dtype = torch.float16
        # else:
        #     self.dtype = torch.float32
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)
        self.back_prop = back_prop
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        # self.grid_size = 256
        # self.register_buffer('density_bitfield',
        # torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        F = 2;  N_min = min_res
        # TODO Look into the max resolution
        b = np.exp(np.log(max_res*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        self.pe, dim = get_positional_embedder(3)
        # self.xyz_encoder = \
        #     tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "Grid",
	    #             "type": "Hash",
        #             "n_levels": L,
        #             "n_features_per_level": F,
        #             "log2_hashmap_size": log2_T,
        #             "base_resolution": N_min,
        #             "per_level_scale": b,
        #             "interpolation": "Linear"
        #         },
        #         dtype=self.dtype
        #     )
        # self.decoder_field = BasicDecoder(input_dim=dim,
        #                         output_dim=1,
        #                         activation=torch.sin,
        #                         bias=True,
        #                         num_layers=6,
        #                         hidden_dim=hidden_size,
        #                         skip=[3])
        self.decoder_field = SirenNet(dim, dim_hidden=hidden_size, dim_out=1, num_layers=6, w0_initial=30, use_bias=True)

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        # x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        # if self.back_prop:
        #     h = self.xyz_encoder(x)
        # else:
        #     h = self.xyz_encoder(x.detach())
        sigmas = self.decoder_field(x)
        # sigmas = TruncExp.apply(sigmas)
        # if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, create_graph=True, return_grad=True):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        x.requires_grad = True
        input = self.pe(x)
        field = self.field(input)
        if return_grad:
            field_grad = self.field_grad(x, field, create_graph=create_graph)
        else:
            field_grad = None
        return field, field_grad

    def field(self, x):
        field = self.density(x)[:, 0:self.output_dim]
        return field

    def field_grad(self, coords, field, create_graph=True):
        field = field.flatten()
        grad_outputs = torch.ones_like(field)
        field = scaler.scale(field)
        # try:
        field_grad = torch.autograd.grad(field, [coords], grad_outputs=grad_outputs, create_graph=create_graph)[0]
        field_grad = field_grad / (scaler.get_scale() + 1e-7)
        # field_grad = scaler.unscale(field_grad)

        # print (scaler.get_scale())
        # import ipdb; ipdb.set_trace()
        # except:
        #     print ("Error in the grad!")
        #     field_grad = [torch.zeros((field.shape[0], 3)).cuda()]
        return field_grad

    def field_double_grad(self, coords, field_grad, create_graph=True):
        coords.requires_grad = True
        # import ipdb; ipdb.set_trace()
        grad_outputs = torch.ones((field_grad.shape[0], 1)).cuda()
        field_grad = scaler.scale(field_grad)
        second_derivative = torch.zeros((field_grad.shape[0], 3, 3)).cuda()

        for i in range(3):
            # TODO need to make sure that the axis is correct
            second_derivative[:, i] = torch.autograd.grad(field_grad[:, i:i+1], [coords], grad_outputs=grad_outputs,
                                                          create_graph=create_graph, allow_unused=False)[0]
        # print (second_derivative.shape)
        return second_derivative
    
    def compute_field_loss(self, weights, weights_rev, field_norm, view_dirs):
        view_dirs = view_dirs/torch.norm(view_dirs, dim=1, keepdim=True)
        field_norm = field_norm #+ torch.rand_like(field_norm) * 0.001
        field_loss = torch.abs(torch.maximum(weights.detach().float(), weights_rev.detach().float()) - \
                                    torch.abs(torch.sum(field_norm * view_dirs.detach(), 1)))
        field_loss = field_loss.mean()
        return field_loss

    def compute_abs_loss(self, field_norm):
        l1_norm = torch.linalg.norm(field_norm, ord=1, dim=1)
        abs_loss = l1_norm.mean()
        return abs_loss
    
    def compute_double_field_loss(self, field_double_der):
        field_loss = torch.abs(field_double_der)
        field_loss = field_loss.mean()
        return field_loss

class DeltaField(nn.Module):
    def __init__(self, scale, back_prop=0, precision=16, log2_T=19, L=16, max_res=512):
        super().__init__()
        # scene bounding box
        if precision == 16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)
        self.back_prop = back_prop
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        # self.grid_size = 256
        # self.register_buffer('density_bitfield',
        # torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        F = 2;
        N_min = 128
        # TODO Look into the max resolution
        b = np.exp(np.log(max_res * scale / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                dtype=self.dtype
            )

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)

        h = self.xyz_encoder(x)
        # sigmas = TruncExp.apply(sigmas)
        # if return_feat: return sigmas, h
        # import ipdb; ipdb.set_trace()
        return h.sum(1)

    def forward(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        x.requires_grad = True
        field = self.field(x)
        return field

    def field(self, x):
        field = self.density(x)
        return field