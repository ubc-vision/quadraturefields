import torch
import numpy as np
from matplotlib import pyplot as plt


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


def plot_field(field_grid, prefix, mask, scale=1.0):
    print ("Plotting field images!")
    grid_size = 512
    mask = mask[0]
    size = mask.shape[0]
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size))

    grid_x = grid_x * scale
    grid_y = grid_y * scale
    kernel = torch.nn.Upsample(scale_factor=grid_size // size, mode='trilinear')
    mask = mask[None, None, ...]
    mask = kernel(mask.float())[0, 0].data.cpu().numpy()

    for i in range(10):
        torch.cuda.empty_cache()
        m = (mask[:, :, int(grid_size // 2 + 10 * i)] * 255).astype(np.uint8)
        # m = np.flip(m, axis=1)
        coords = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.ones((grid_size ** 2, 1)) * 10 * i / grid_size], 1)
        coords = coords.cuda()
        field, field_grad = field_grid(coords.detach())

        field_grad = (field_grad).data.cpu().numpy()

        field = (field).data.cpu().numpy()
        field_grad = field_grad.reshape((grid_size, grid_size, -1)) * m[..., None]
        field = field.reshape((grid_size, grid_size)) * m
        field = field - field.min()
        field = field / (field.max() + 1e-6)

        field_grad = field_grad - field_grad.min()
        field_grad = field_grad / (field_grad.max() + 1e-6)
        
        plt.imsave("{}/grad_{}.png".format(prefix, i), (field_grad * 255).astype(np.uint8))
        plt.imsave("{}/grid_{}.png".format(prefix, i), (field * 255).astype(np.uint8))
        plt.imsave("{}/mask_{}.png".format(prefix, i), m)


def plot_field_mip360(field_grid, prefix, binary=None, normalize="none", scale=1.0):
    print ("Plotting field images!")
    grid_size = 1024
    z = torch.linspace(-1, 1, grid_size)
    # Since we are not normalizing the coordinates, the input to the grid should be between 0, 1
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size))
    z = z * scale
    grid_x = grid_x * scale
    grid_y = grid_y * scale
    if binary is not None:
        binary = binary[0]
        binary = binary.reshape((128, 128, 128))
        size = binary.shape[0]
        binary = binary.float()
        kernel = torch.nn.Upsample(scale_factor=grid_size // size, mode='trilinear')
        binary = binary[None, None, ...]
        binary = kernel(binary.float())[0, 0]
        size = binary.shape[0]

    for i in range(20):
        torch.cuda.empty_cache()
        # m = np.flip(m, axis=1)
        coords = torch.cat([grid_x.reshape(-1, 1),
                            torch.ones((grid_size ** 2, 1)) * z[grid_size // 2 + 16 * i], grid_y.reshape(-1, 1)], 1)
        coords = coords.cuda()
        field, field_grad = field_grid(coords.detach())

        field_grad = (field_grad).data.cpu().numpy()

        field = (field).data.cpu().numpy()
        field_grad = field_grad.reshape((grid_size, grid_size, -1))
        field = field.reshape((grid_size, grid_size))
        field = field - field.min()
        field = field / (field.max() + 1e-6)

        field_grad = field_grad - field_grad.min()
        field_grad = field_grad / (field_grad.max() + 1e-6)
        if binary is not None:
            m = (binary[:, int(size // 2 + 16 * i), :].data.cpu().numpy() * 255).astype(np.uint8)
            plt.imsave("{}/mask_{}.png".format(prefix, i), m)

        plt.imsave("{}/grad_{}.png".format(prefix, i), (field_grad * 255).astype(np.uint8))
        plt.imsave("{}/grid_{}.png".format(prefix, i), (field * 255).astype(np.uint8))


def extract_grid_mip360_chunks(field_grid, prefix, scale, grid_size=1024, chunk_size=256, average_scale=2):
    import time
    tix = time.time()
    print("Extracting geometry!")
    num_chunks = grid_size // chunk_size

    m = torch.nn.AvgPool3d(average_scale, stride=average_scale, padding=0)

    def eval(x, batch_size=1000000):
        fields = []
        field_grads = []
        for b in range(0, x.shape[0], batch_size):
            torch.cuda.empty_cache()
            f, fg = field_grid(x[b:b + batch_size].detach())
            with torch.no_grad():
                fields.append(f.detach())
                field_grads.append(fg.detach())
        fields = torch.cat(fields, 0)
        field_grads = torch.cat(field_grads, 0)
        return fields, field_grads

    def eval_chunks(start, end, chunk_id, average_scale=2):
        temp_grids = torch.zeros((grid_size * average_scale, grid_size * average_scale, average_scale))
        temp_grads = torch.zeros((grid_size * average_scale, grid_size * average_scale, average_scale))
        z = torch.linspace(-1, 1, chunk_size * average_scale)
        grids = np.zeros((grid_size, grid_size, chunk_size), dtype=np.float16)
        grads = np.zeros((grid_size, grid_size, chunk_size), dtype=np.float16)
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, average_scale * grid_size), torch.linspace(-1, 1, average_scale * grid_size))
        for j in range(chunk_size * average_scale):
            coords = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.ones(((average_scale * grid_size) ** 2, 1)) * z[j]], 1)
            coords = coords.cuda() * scale
            coords = coords.detach()
            coords.requires_grad = True
            field, field_grad = eval(coords.detach())
            with torch.no_grad():
                field_grad = torch.linalg.norm(field_grad, dim=-1)

            torch.cuda.empty_cache()
            field = field.reshape((grid_size * average_scale, grid_size * average_scale))
            field_grad = field_grad.reshape((grid_size * average_scale, grid_size * average_scale))
            field_grad = torch.clip(field_grad, 0, 65504)
            temp_grids[:, :, j % average_scale] = field
            temp_grads[:, :, j % average_scale] = field_grad

            if average_scale == 1:
                grids[:, :, j] = field.data.cpu().numpy()
                grads[:, :, j] = field_grad.data.cpu().numpy()
            elif j % average_scale == 1:
                grids[:, :, j // average_scale] = m(temp_grids[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
                grads[:, :, j // average_scale] = m(temp_grads[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()

        # save using h5py format
        tic = time.time()
        import h5py
        with h5py.File("{}/grids_valid_{}.h5".format(prefix, chunk_id), 'w') as hf:
            hf.create_dataset("grids", data=grids)
        with h5py.File("{}/grads_valid_{}.h5".format(prefix, chunk_id), 'w') as hf:
            hf.create_dataset("grads", data=grads)
        print("Time taken to save the chunk: ", time.time() - tic)

    for i in range(num_chunks):
        print("Extracting chunk: ", i)
        tic = time.time()
        eval_chunks(i * chunk_size, (i + 1) * chunk_size, i, average_scale=average_scale)
        print("Time taken to extract the chunk: ", time.time() - tic)
    print ("Time taken to extract the geometry: ", time.time() - tix)

def extract_grid_mip360(field_grid, prefix, scale):
    print("Extracting geometry!")
    grid_size = 1024
    z = torch.linspace(-1, 1, grid_size * 2)
    grids = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    grads = np.zeros((grid_size, grid_size, grid_size), dtype=np.float16)
    m = torch.nn.AvgPool3d(2, stride=2, padding=0)

    def eval(x, batch_size=1000000):
        fields = []
        field_grads = []
        for b in range(0, x.shape[0], batch_size):
            torch.cuda.empty_cache()
            f, fg = field_grid(x[b:b + batch_size].detach())
            with torch.no_grad():
                fields.append(f.detach())
                field_grads.append(fg.detach())
        fields = torch.cat(fields, 0)
        field_grads = torch.cat(field_grads, 0)
        return fields, field_grads
    temp_grids = torch.zeros((grid_size * 2, grid_size * 2, 2))
    temp_grads = torch.zeros((grid_size * 2, grid_size * 2, 2))

    for j in range(grid_size * 2):
        # NOTE First compute the field at higher resolution then average it
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, 2 * grid_size), torch.linspace(-1, 1, 2 * grid_size))
        coords = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.ones(((2 * grid_size) ** 2, 1)) * z[j]], 1)
        coords = coords.cuda() * scale
        coords = coords.detach()
        coords.requires_grad = True
        # field, field_grad = field_grid(coords.detach())
        field, field_grad = eval(coords.detach())
        with torch.no_grad():
            field_grad = torch.linalg.norm(field_grad, dim=-1)

        torch.cuda.empty_cache()
        field = field.reshape((grid_size * 2, grid_size * 2))
        field_grad = field_grad.reshape((grid_size * 2, grid_size * 2))

        temp_grids[:, :, j % 2] = field
        temp_grads[:, :, j % 2] = torch.clip(field_grad, 0, 65504)
        if j % 2 == 1:
            grids[:, :, j // 2] = m(temp_grids[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
            grads[:, :, j // 2] = m(temp_grads[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
    np.save("{}/grids_valid.npy".format(prefix), grids)
    # trying to save disk memory
    np.save("{}/grads_valid.npy".format(prefix), grads.astype(np.float16))


def extract_grid(field_grid, prefix, scale):
    print("Extracting density!")
    grid_size = 1024
    z = torch.linspace(-1, 1, grid_size * 2)
    grids = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    grads = np.zeros((grid_size, grid_size, grid_size), dtype=np.float16)
    m = torch.nn.AvgPool3d(2, stride=2, padding=0)

    # TODO First compute the field at higher resolution then average it
    temp_grids = torch.zeros((grid_size * 2, grid_size * 2, 2))
    temp_grads = torch.zeros((grid_size * 2, grid_size * 2, 2))

    for j in range(grid_size * 2):
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, 2 * grid_size), torch.linspace(-1, 1, 2 * grid_size))
        coords = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.ones(((2 * grid_size) ** 2, 1)) * z[j]], 1)
        coords = coords.cuda() * scale
        coords = coords.detach()
        coords.requires_grad = True
        fields = []
        field_grads = []
        for b in range(0, coords.shape[0], 1000000):
            f, fg = field_grid(coords[b:b + 1000000].detach())
            with torch.no_grad():
                fields.append(f.detach())
                field_grads.append(fg.detach())
        field = torch.cat(fields, 0)
        field_grad = torch.cat(field_grads, 0)
        # field, field_grad = field_grid(coords.detach())
        with torch.no_grad():
            field_grad = torch.linalg.norm(field_grad, dim=-1)

        torch.cuda.empty_cache()
        field = field.reshape((grid_size * 2, grid_size * 2))[None, None, ...]
        field_grad = field_grad.reshape((grid_size * 2, grid_size * 2))[None, None, ...]

        temp_grids[:, :, j % 2] = field
        # clip to the max value of torch.float16
        temp_grads[:, :, j % 2] = torch.clip(field_grad, 0, 65504)
        if j % 2 == 1:
            grids[:, :, j // 2] = m(temp_grids[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
            grads[:, :, j // 2] = m(temp_grads[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
    np.save("{}/grids_valid.npy".format(prefix), grids)
    np.save("{}/grads_valid.npy".format(prefix), grads.astype(np.float16))


def extract_density_grid(model, scale, prefix, grid_size=512):
    print("Extracting geometry!")
    grid_size = grid_size
    z = torch.linspace(-1, 1, grid_size * 2)
    grids = np.zeros((grid_size, grid_size, grid_size), dtype=np.float16)
    m = torch.nn.AvgPool3d((2, 2, 2), stride=2, padding=(0, 0, 0))
    # TODO First compute the field at higher resolution then average it
    temp_grids = torch.zeros((grid_size * 2, grid_size * 2, 2))

    for j in range(grid_size * 2):
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, grid_size * 2), torch.linspace(-1, 1, grid_size * 2))
        coords = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.ones(((grid_size * 2) ** 2, 1)) * z[j]], 1)
        coords = coords.cuda() * scale
        with torch.no_grad():
            sigma = model.query_density(coords)
        torch.cuda.empty_cache()
        field = sigma.reshape((grid_size * 2, grid_size * 2))
        temp_grids[:, :, j % 2] = torch.clip(field, 0, 65504)
        if j % 2 == 1:
            grids[:, :, j // 2] = m(temp_grids[None, None, ...])[0, 0, ..., 0].data.cpu().numpy()
    np.save(prefix + "density_grids_valid.npy", grids)


def batchify(field_net, coords, batch_size=100000):
    fields = []
    field_grads = []

    for i in range(0, coords.shape[0], batch_size):
        x = coords[i:i + batch_size].detach()
        field, field_grad = field_net(x, create_graph=False)
        field_grad = torch.linalg.norm(field_grad, dim=-1)
        fields.append(field)
        field_grads.append(field_grad)
    fields = torch.cat(fields, 0)
    field_grads = torch.cat(field_grads, 0)
    torch.cuda.empty_cache()
    return fields, field_grads

def plot_field_(model, thres):
    print ("Plotting field images!")
    indices, c_coords = model.get_all_cells()[0]

    mask = model.density_grid > thres
    # mask = mask.reshape((1, -1))
    v_index = c_coords[:, 0] == 64  
    m = mask[0, v_index].data.cpu().numpy()

    plt.imsave("grad_{}.png".format(0), m.reshape((128, 128)))
