import torch
import torch.nn.functional as F
from torch.autograd import Variable # deprecated - use Tensor

import numpy as np

from .vgg import VGG


def get_cuda_device() -> str:
    cuda_device = 'cuda:%s' % torch.cuda.current_device()
    if not cuda_device:
        raise Exception('no cuda device found')
    return cuda_device


def get_vgg(engine, model_path):
    vgg = VGG()

    for param in vgg.parameters():
        param.requires_grad = False

    if engine == 'gpu':
        vgg.cuda()

        vgg.load_state_dict(torch.load(model_path))

        if torch.cuda.is_available():
            do_cuda = True
        else:
            msg = "gpu mode was requested, but cuda is not available"
            raise RuntimeError(msg)

    elif engine == 'cpu':
        vgg.load_state_dict(torch.load(model_path))
        do_cuda = False
    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg)

    return vgg, do_cuda


def rescale_tensor(tensor, scale_factor):
    b, c, w, h = tensor.size()
    tensor = torch.nn.functional.interpolate(tensor, size=[int(w*scale_factor), int(h*scale_factor)], mode='bilinear')
    return tensor


def make_gaussian_pyramid(img, mips=5, cuda=True, pyramid_scale_factor=0.63):
    kernel = _build_gauss_kernel(cuda)
    gaus_pyramid = _gaussian_pyramid(img, kernel, mips, pyramid_scale_factor)
    return gaus_pyramid

# todo: replace with kornia to avoid numpy dependency, which breaks torchscript
def _build_gauss_kernel(cuda, size=5, sigma=1.0, n_channels=3):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)

def _conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    result = F.conv2d(img, kernel, groups=n_channels)
    return result

def _gaussian_pyramid(img, kernel, max_levels, pyramid_scale_factor):
    current = img
    pyr = [current]

    for level in range(0, max_levels-1):
        filtered = _conv_gauss(current, kernel)
        current = F.interpolate(filtered, scale_factor=pyramid_scale_factor)
        pyr.append(current)

    return pyr