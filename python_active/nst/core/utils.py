import os

import torch
from torch import nn
from torch.autograd import Variable # deprecated - use Tensor
import torchvision.transforms.functional

import kornia

import math

import numpy as np

from .vgg import VGG



# def get_fft_scale(h, w, decay_power=.75, **kwargs):
#     d = .5 ** .5  # set center frequency scale to 1
#     fy = np.fft.fftfreq(h, d=d)[:,None]
#     if w % 2 == 1:
#         fx = np.fft.fftfreq(w, d=d)[: w // 2 + 2]
#     else:
#         fx = np.fft.fftfreq(w, d=d)[: w // 2 + 1]
#     freqs = (fx*fx + fy*fy) ** decay_power
#     scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
#     scale = torch.tensor(scale).float()[None, None, ..., None].cuda()
#     return scale
#
# def fft_to_rgb(h, w, t, **kwargs):
#     scale = get_fft_scale(h, w, **kwargs)
#     t = scale * t
#     t = torch.fft.irfft(t, 2, norm='forward')
#     return t
#
# def rgb_to_fft(h, w, t, **kwargs):
#     t = torch.fft.rfft(t, norm='forward')
#     scale = get_fft_scale(h, w, **kwargs)
#     t = t / scale
#     return t



def jitter(img):
    b, c, h, w = img.size()

    resample = kornia.constants.Resample.BICUBIC

    rand_rotation = 1.0
    # rand_rotation = 0.0
    # rand_crop_scale = (0.97, 1.0)
    # rand_crop_ratio = (0.97, 1.03)
    rand_crop_scale = (0.999, 1.0)
    # rand_crop_ratio = (h/w*0.98, h/w*1.02)
    rand_crop_ratio = (h/w, h/w)

    transform = nn.Sequential(
        kornia.augmentation.RandomResizedCrop(size=(h, w), scale=rand_crop_scale, ratio=rand_crop_ratio, resample=resample),
        # kornia.augmentation.RandomRotation(degrees=rand_rotation, resample=resample)
    )

    return transform(img)


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


def rescale_tensor_by_tensor(tensor1, tensor2, requires_grad=False):
    b, c, w, h = tensor2.size()

    tensor = torch.nn.functional.interpolate(tensor1, size=[int(w), int(h)],
                                             mode='bilinear')

    if requires_grad:
        tensor = Variable(tensor.data.clone(), requires_grad=True)

    return tensor


def rescale_tensor(tensor, scale_factor, requires_grad=False):
    b, c, w, h = tensor.size()
    tensor = torch.nn.functional.interpolate(tensor, size=[int(w*scale_factor), int(h*scale_factor)],
                                             mode='bilinear')

    if requires_grad:
        tensor = Variable(tensor.data.clone(), requires_grad=True)

    return tensor


def zoom_image(img, zoom, output_path=''):

    if zoom == 1.0:
        return img

    if zoom >= 1:
        result = centre_crop_image(img, zoom)
    else:
        result = tile(img, zoom)

    # for debug purposes, write out the zoomed image as a .pt
    if output_path:
        print('writing zoom tensor')
        torch.save(result, output_path)

    return result


def random_rotate_crop(t, scale=(0.3, 1.0), rotate=(0, 360), mode='circular'):
    '''
    modes: reflect, circular
    '''
    b, c, h, w = t.size()

    transform_c = torchvision.transforms.RandomResizedCrop((h, w), scale=scale, ratio=(1.0, 1.0))
    img = transform_c(t)

    crop_x = w - 1
    crop_y = h - 1
    img = torch.nn.functional.pad(img, (crop_x, crop_x, crop_y, crop_y), mode=mode)

    transform_r = torchvision.transforms.RandomRotation(rotate)
    img = transform_r(img)

    img = torchvision.transforms.functional.center_crop(img, (h, w))

    return img


def tile(img, zoom, cuda=False):
    # zoom out, i.e. zoom is between zero and one
    b, c, old_width, old_height = img.size()
    img = torch.nn.functional.interpolate(img, scale_factor=zoom, mode='bicubic', antialias='bicubic')
    b, c, new_width, new_height = img.size()

    # determine how many tiles are needed
    # does torchscript allow math?  need to use torch ceil?
    x_tile = math.ceil(old_width / new_width)
    y_tile = math.ceil(old_height / new_height)

    img = img.tile((x_tile, y_tile))
    img = torchvision.transforms.functional.center_crop(img, (old_width, old_height))

    return img


def centre_crop_image(img, zoom):
    b, c, old_x, old_y = img.size()

    if zoom != 1.0:
        crop_width = int(old_x / zoom)
        crop_height = int(old_y / zoom)
        top = int((old_y - crop_height) / 2.)
        left = int((old_x - crop_width) / 2.)
        img = torchvision.transforms.functional.crop(img, top, left, crop_width, crop_height)

    out_x = int(old_x)
    out_y = int(old_y)
    img = torch.nn.functional.interpolate(img, size=(out_x, out_y))
    return img


def log_cuda_memory():
    max_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000000
    max_mem_cached = torch.cuda.max_memory_reserved(0) / 1000000
    print('memory used: %s of %s' % (max_mem_cached, max_memory))


def write_pyramid(pyramid, outdir):
    outdir = outdir + '/pyramids'
    os.makedirs(outdir, exist_ok=True)

    for mip in range(0, len(pyramid)):
        filepath = outdir + "/%02d.pt" % mip
        torch.save(pyramid[mip], filepath)


def make_gaussian_pyramid(img, span, mips, cuda=True):
    """
    given an image, generate a series of guassian pyramids to fill a given span.
    the span is a multipler of the scale for the limit pyramid.

    mips defines the total number of mips (minimum 2).  more mips results
    in better sampling of the style, at the cost of more memory.

    """
    kernel = _build_gauss_kernel(cuda)

    if mips > 1:
        pyramid_scale_factor = span**(1/(mips-1))
    else:
        pyramid_scale_factor = 1.0

    gaus_pyramid = _gaussian_pyramid(img, kernel, mips, pyramid_scale_factor)
    return gaus_pyramid


# todo: kernel sizes should come from settings so we can scale them for full res image
def _build_gauss_kernel(cuda, size=5, sigma=1.0, n_channels=3):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
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
    img = torch.nn.functional.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    result = torch.nn.functional.conv2d(img, kernel, groups=n_channels)
    return result

def _gaussian_pyramid(img, kernel, max_levels, pyramid_scale_factor):
    current = img
    pyr = [current]

    for level in range(1, max_levels):
        filtered = _conv_gauss(current, kernel)
        current = torch.nn.functional.interpolate(filtered, scale_factor=pyramid_scale_factor)
        pyr.append(current)

    return pyr

def write_tensor(grad, fp):
    outdir = os.path.abspath(os.path.join(fp, os.path.pardir))
    os.makedirs(outdir, exist_ok=True)
    torch.save(grad, fp)
