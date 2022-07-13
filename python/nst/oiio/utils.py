"""
Repetitive utility functions that have nothing to do with style transfer
"""
from typing import List
import math
import copy
import subprocess
import shutil

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from nst.core import utils as core_utils

import cv2
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import os
import numpy as np
from operator import itemgetter


# to do: this should move to temporal coherence
def output_motion_mask(frame, render, motion_fore, render_denoised, dilate=50, no_dilate=False, debug_mask="", threshold_value=1,
            threshold_max=10000):
    # need to get thse programmatically
    # render_x = 512
    # render_y = 512
    # to do: handle when motion vec is arbitrary dimensions.

    # check if a render_denoised frame is available, use it if it is
    render_prev_denoised = render_denoised.replace('*', '%04d' % (frame - 1))
    if os.path.isfile(render_prev_denoised):
        render_prev = render_prev_denoised
    else:
        render_prev = render.replace('*', '%04d' % (frame - 1))

    if not os.path.isfile(render_prev):
        print("no previous frame for frame %04d, skipping" % frame)
        print(render_prev)
        return

    render = render.replace('*', '%04d' % frame)
    # render_np = oiio.ImageBuf(render).get_pixels()

    t_ = render_denoised.split('/')
    t_.pop()
    d_ = ('/'.join(t_))
    if not os.path.isdir(d_):
        os.makedirs(d_)

    t_ = debug_mask.split('/')
    t_.pop()
    d_ = ('/'.join(t_))
    if not os.path.isdir(d_):
        os.makedirs(d_)

    render_np_prev = oiio.ImageBuf(render_prev).get_pixels()
    motion_fore = motion_fore.replace('*', '%04d' % (frame - 1))
    motion_fore_np = oiio.ImageBuf(motion_fore).get_pixels()

    x, y, z = render_np_prev.shape

    # 1. warp prev frame into current, keep only pixels that have change
    frame_mask = np.zeros((x, y, z))

    pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):
            render_value = render_np_prev[old_x][old_y]
            mx_, my_, _ = motion_fore_np[old_x][old_y]

            # flip axes
            mx = int(my_)
            my = int(mx_)

            # if mx == 0 and my == 0:
            #     continue

            nx = old_x + mx
            ny = old_y + my

            # move weakest moves first, strongest last.
            render_mag = render_value[0] + render_value[1] + render_value[2]
            pixels_to_warp.append(
                (old_x, old_y, nx, ny, (render_value[0], render_value[1], render_value[2]), render_mag, mx_, my_))

    sorted_pixels_to_warp = sorted(pixels_to_warp, key=itemgetter(5))

    for sp in sorted_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            frame_mask[nx][ny] = render_value
        except IndexError:
            continue

    if not no_dilate:
        kernel = np.ones((dilate, dilate), np.uint8)
        frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)
        # _, frame_mask = cv2.threshold(frame_mask, threshold_value, threshold_max, cv2.THRESH_BINARY)
        frame_mask = cv2.GaussianBlur(frame_mask, (5, 5), 0)

    if debug_mask:
        frame_mask_buf = oiio.ImageBuf(oiio.ImageSpec(x, y, 3, oiio.FLOAT))
        frame_mask_buf.set_pixels(oiio.ROI(), frame_mask.copy())
        debug_mask = debug_mask.replace("*", "%04d" % frame)
        frame_mask_buf.write(debug_mask)

    return


def tensor_to_image(tensor):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])

    # what's this do?
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor.data[0].cpu().squeeze())
    t[t > 1] = 1
    t[t < 0] = 0
    out_img = postpb(t)
    return out_img



def rgba_style_image_to_tensors(image: str, do_cuda: bool, resize: float=None, colorspace=None,
                                raw=False) -> List[torch.Tensor]:

    buf = ImageBuf(image)
    rgba_np = buf.get_pixels()
    # style1_np = image_list[2]
    rgb_tensor = torch.Tensor(rgba_np.copy())
    rgb_tensor = rgb_tensor.transpose(0, 2)
    rgb_tensor = rgb_tensor[:3:]
    rgb_tensor = rgb_tensor.transpose(0, 2)
    # rgb_tensor = color_in(style1_tensor, do_cuda=cuda)

    alpha_tensor = torch.Tensor(rgba_np.copy())
    alpha_tensor = alpha_tensor.transpose(0, 2)
    alpha_tensor[0] = alpha_tensor[3]
    alpha_tensor[1] = alpha_tensor[3]
    alpha_tensor[2] = alpha_tensor[3]
    alpha_tensor = alpha_tensor[:3:]
    alpha_tensor = alpha_tensor.transpose(0, 2)
    # style1_alpha_tensor = color_in(style1_alpha_tensor, do_cuda=cuda, raw=True)

    pass



# new - handle rgba inputs, return alpha as a second tensor
def style_image_to_tensors(image: str, do_cuda: bool, resize: float = None, colorspace=None) -> List[torch.Tensor]:
    tensors = []

    buf = ImageBuf(image)

    print(1.1, image, buf.nchannels)

    o_width = buf.oriented_full_width
    o_height = buf.oriented_full_height

    if resize:
        n_width = int(float(o_width) * resize)
        n_height = int(float(o_height) * resize)
        buf = oiio.ImageBufAlgo.resize(buf, roi=ROI(0, n_width, 0, n_height, 0, 1, 0, 3))

    if colorspace:
        if colorspace != 'srgb_texture':
            buf = oiio.ImageBufAlgo.colorconvert(buf, colorspace, 'srgb_texture')

    if buf.nchannels == 3:
        rgb_tensor = torch.Tensor(buf.get_pixels().copy())
        rgb_tensor = transform_image_tensor(rgb_tensor, do_cuda)
        tensors += [rgb_tensor]

    elif buf.nchannels == 4:
        rgb_tensor = torch.Tensor(buf.get_pixels().copy())
        rgb_tensor = rgb_tensor.transpose(0, 2)
        rgb_tensor = rgb_tensor[:3:]
        rgb_tensor = rgb_tensor.transpose(0, 2)
        rgb_tensor = transform_image_tensor(rgb_tensor, do_cuda)

        alpha_tensor = torch.Tensor(buf.get_pixels().copy())
        alpha_tensor = alpha_tensor.transpose(0, 2)
        alpha_tensor[0] = alpha_tensor[3]
        alpha_tensor[1] = alpha_tensor[3]
        alpha_tensor[2] = alpha_tensor[3]
        alpha_tensor = alpha_tensor[:3:]
        alpha_tensor = alpha_tensor.transpose(0, 2)
        alpha_tensor = transform_image_tensor(alpha_tensor, do_cuda, raw=True)
        tensors += [rgb_tensor, alpha_tensor]

    return tensors


# new
def transform_image_tensor(tensor: torch.Tensor, do_cuda: bool, raw=False) -> torch.Tensor:

    tforms_ = []

    #  turn to BGR
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    if not raw:
        # subtract imagenet mean
        tforms_ += [transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1])]

        # scale to imagenet values
        tforms_ += [transforms.Lambda(lambda x: x.mul_(255.))]

    tforms = transforms.Compose(tforms_)

    tensor = torch.transpose(tensor, 2, 0)
    tensor = torch.transpose(tensor, 2, 1)

    tensor = tforms(tensor)

    if do_cuda:
        device = core_utils.get_cuda_device()
        tensor = tensor.detach().to(torch.device(device))
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)


# deprecate
def image_to_tensor(image: str, do_cuda: bool, resize: float=None, colorspace=None, raw=False) -> torch.Tensor:
        # note: oiio implicitely converts to 0-1 floating point data here regardless of format:
        buf = ImageBuf(image)

        o_width = buf.oriented_full_width
        o_height = buf.oriented_full_height

        if resize:
            n_width = int(float(o_width) * resize)
            n_height = int(float(o_height) * resize)
            buf = oiio.ImageBufAlgo.resize(buf, roi=ROI(0, n_width, 0, n_height, 0, 1, 0, 3))

        if colorspace:
            if colorspace != 'srgb_texture':
                buf = oiio.ImageBufAlgo.colorconvert(buf, colorspace, 'srgb_texture')

        tensor = buf_to_tensor(buf, do_cuda, raw=raw)

        return tensor


# deprecate
def buf_to_tensor(buf: oiio.ImageBuf, do_cuda: bool, raw=False) -> torch.Tensor:

    tforms_ = []

    #  turn to BGR
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    if not raw:
        # subtract imagenet mean
        tforms_ += [transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1])]

        # scale to imagenet values
        tforms_ += [transforms.Lambda(lambda x: x.mul_(255.))]

    tforms = transforms.Compose(tforms_)

    # note: oiio implicitely converts to 0-1 floating point data here regardless of format:
    it = torch.Tensor(buf.get_pixels())

    it = torch.transpose(it, 2, 0)
    it = torch.transpose(it, 2, 1)

    it = tforms(it)

    if do_cuda:
        device = core_utils.get_cuda_device()
        it = it.detach().to(torch.device(device))
        return it.unsqueeze(0).cuda()
    else:
        return it.unsqueeze(0)


def tensor_to_buf(tensor: torch.Tensor) -> oiio.ImageBuf:
    tforms_ = []

    tforms_ += [transforms.Lambda(lambda x: x.mul_(1. / 255.))]

    # add imagenet mean
    tforms_ += [transforms.Normalize(mean=[(-0.40760392), -0.45795686, -0.48501961], std=[1, 1, 1])]

    # turn to RGB
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    tforms = transforms.Compose(tforms_)

    tensor_ = torch.clone(tensor)
    t = tforms(tensor_.data[0].cpu().squeeze())

    t = torch.transpose(t, 2, 0)
    t = torch.transpose(t, 0, 1)
    t = t.contiguous()
    n = t.numpy()
    x, y, z = n.shape
    buf = ImageBuf(ImageSpec(y, x, z, oiio.FLOAT))
    buf.set_pixels(ROI(), n)
    return buf


def write_exr(exr_buf: oiio.ImageBuf, filepath: str) -> None:
    pardir = os.path.abspath(os.path.join(filepath, os.path.pardir))
    os.makedirs(pardir, exist_ok=True)
    exr_buf.write(filepath, oiio.FLOAT)


def write_jpg(buf: oiio.ImageBuf, filepath: str) -> None:
    pardir = os.path.abspath(os.path.join(filepath, os.path.pardir))
    os.makedirs(pardir, exist_ok=True)
    buf.write(filepath, dtype=oiio.FLOAT, fileformat='jpg')


def get_full_path(filename):
    if not filename.startswith('/'):
        return os.getcwd() + '/' + filename
    return filename


def do_ffmpeg(output_dir, temp_dir=None):
# ffmpeg -start_number 1001 -i ./comp.%04d.png -c:v libx264 -crf 15 -y ./comp.mp4
    if not temp_dir:
        temp_dir = '%s/tmp' % output_dir

    ffmpeg_cmd = []
    ffmpeg_cmd += ['ffmpeg', '-i', '%s/render.%%04d.png' % temp_dir]
    ffmpeg_cmd += ['-c:v', 'libx264', '-crf', '15', '-y']
    ffmpeg_cmd += ['%s/prog.mp4' % output_dir]
    subprocess.check_output(ffmpeg_cmd)
    shutil.rmtree(temp_dir)


#####################################################################################################
# This code was mostly ruthlessly appropriated from tyneumann's "Minimal PyTorch implementation of Generative Latent Optimization" https://github.com/tneumann/minimal_glo. Thank the lord for clever germans.


def zoom_image(img, zoom, rescale, cuda=False, zoom_factor=0.17):
    if zoom == 1.0:
        return img

    if zoom >= 1:
        return centre_crop_image(img, zoom, rescale, cuda=cuda)
    else:
        return tile(img, zoom, rescale, cuda=cuda)


def tile(img, zoom, rescale, cuda=False):
    # zoom out, i.e. zoom is between zero and one

    img = torch.nn.functional.interpolate(img, scale_factor=rescale)
    b, c, old_width, old_height = img.size()
    img = torch.nn.functional.interpolate(img, scale_factor=zoom)
    b, c, new_width, new_height = img.size()

    # determine how many tiles are needed
    x_tile = math.ceil(old_width / new_width)
    y_tile = math.ceil(old_height / new_height)

    img = img.tile((x_tile, y_tile))

    # crop to old size
    buf = tensor_to_buf(copy.deepcopy(img))
    roi = oiio.ROI(int(0), int(old_width), int(0), int(old_height))
    buf = oiio.ImageBufAlgo.crop(buf, roi=roi)
    img = buf_to_tensor(buf, cuda)

    return img


def centre_crop_image(img, zoom, rescale, cuda=False, zoom_factor=0.17):
    _, _, old_x, old_y = img.size()

    if zoom != 1.0:
        zoom_ = 1+(zoom*zoom_factor) # 1.612
        crop_width = old_x / zoom_
        crop_height = old_y / zoom_
        left = (old_x - crop_width) / 2.
        right = crop_width + left
        bottom = (old_y - crop_height) / 2.
        top = bottom + crop_height
        buf = tensor_to_buf(copy.deepcopy(img)) # transpose happens here
        roi = oiio.ROI(int(bottom), int(top), int(left), int(right)) # reverse transpose
        buf = oiio.ImageBufAlgo.crop(buf, roi=roi)
        img = buf_to_tensor(buf, cuda)

    out_x = int(old_x * rescale)
    out_y = int(old_y * rescale)
    img = torch.nn.functional.interpolate(img, size=(out_x, out_y))
    return img


class Pyramid(object):

    @classmethod
    def make_gaussian_pyramid(cls, img, mips=5, cuda=True, pyramid_scale_factor=0.63):
        kernel = cls._build_gauss_kernel(cuda)
        gaus_pyramid = cls._gaussian_pyramid(img, kernel, mips, pyramid_scale_factor)
        return gaus_pyramid

    @classmethod
    def write_gaussian_pyramid(cls, gauss_pyramid, outdir, ext='jpg'):
        for index, level in enumerate(gauss_pyramid):
            os.makedirs(outdir, exist_ok=True)
            fp = outdir + '/gaus_pyr_lvl_%s.%s' % (index, ext)
            buf = tensor_to_buf(level)
            write_jpg(buf, fp)

    @staticmethod
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

    @staticmethod
    def _conv_gauss(img, kernel):
        """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        result = F.conv2d(img, kernel, groups=n_channels)
        return result

    @classmethod
    def _gaussian_pyramid(cls, img, kernel, max_levels, pyramid_scale_factor):
        current = img
        pyr = [current]

        for level in range(0, max_levels-1):
            filtered = cls._conv_gauss(current, kernel)
            current = F.interpolate(filtered, scale_factor=pyramid_scale_factor)
            pyr.append(current)

        return pyr


def lerp_points(levels, keys):
    """
    lerp_points(5, [0.1, 1.0, 0.2])
    """
    points = [x for x, y in enumerate([keys])]
    values = [x for x in keys]
    x = np.linspace(points[0], points[-1], num=levels)
    return np.interp(x, points, values)


def write_activations(activations, outdir, layer_limit=10, ext='exr'):
    _, z, x, y = activations.size()

    if not layer_limit:
        layer_limit = z

    for i in range(0,layer_limit):
        fp = '%s/%03d.exr' % (outdir, i)
        a = activations[0][i]
        b = torch.zeros(3, x, y)
        b[0] = a
        b[1] = a
        b[2] = a
        c = torch.transpose(b, 0, 2)
        d = torch.transpose(c, 0, 1)
        e = d.cpu().numpy()
        np_write(e, fp, ext='exr', silent=True)


def write_activation_atlas(vgg_layer_activations, layer_name, outdir):
    fp = '%s/%s.exr' % (outdir, layer_name)

    _, num_tiles, tile_size_x, tile_size_y = vgg_layer_activations.size()

    num_x_tiles = int(round(math.sqrt(num_tiles)))
    num_y_tiles = num_x_tiles

    atlas_size_x = num_x_tiles * tile_size_x
    atlas_size_y = num_y_tiles * tile_size_y

    a = np.zeros((atlas_size_x, atlas_size_y, 3))

    print(1.2, a.shape)

    atlas_tile_x = 0
    atlas_tile_y = 0
    #
    # for tile in range(0, num_tiles):
    #     print(1.3)
    #     for x in range(0, tile_size_x):
    #         for y in range(0, tile_size_y):
    #             value = vgg_layer_activations[0][tile][x][y]
    #             atlas_x = x + atlas_tile_x
    #             atlas_y = y + atlas_tile_y
    #             try:
    #                 print(1.6)
    #                 a[atlas_x][atlas_y] = value
    #             except:
    #                 print(1.7)
    #                 pass

        # if atlas_tile_x >= atlas_size_x:
        #     atlas_tile_x = 0
        #     atlas_tile_y += tile_size_y
        # elif atlas_tile_y >= atlas_size_y:
        #     atlas_tile_y = 0
        #     atlas_tile_x += tile_size_x
        # else:
        #     atlas_tile_x += tile_size_x

    #np_write(a, filepath, ext='exr')


def np_write(np, fp, ext='jpg', silent=False):
    os.makedirs(os.path.abspath(os.path.join(fp, os.path.pardir)), exist_ok=True)
    x, y, z = np.shape
    buf = ImageBuf(ImageSpec(y, x, z, oiio.FLOAT)) # flip x and y
    buf.set_pixels(ROI(), np.copy())
    if not silent:
        print('writing:', fp)
    buf.write(fp, oiio.FLOAT, fileformat=ext)


def make_output_dirs(output):
    t_ = output.split('/')
    t_.pop()
    d_ = ('/'.join(t_))
    try:
        os.makedirs(d_)
    except:
        pass


def write_gradient(grad, fp):
    t = grad.data[0].cpu().squeeze()
    t = torch.transpose(t, 2, 0)
    t = torch.transpose(t, 0, 1)
    t = t.contiguous()
    ni = t.numpy()
    np_write(ni, fp, silent=True)


def write_gram(tensor, fp):
    t = tensor.data.cpu()
    z, x, y = t.size()
    a = t[0]
    b = torch.zeros(3, x, y)
    b[0] = a
    b[1] = a
    b[2] = a
    c = torch.transpose(b, 0, 2)
    d = torch.transpose(c, 0, 1)
    e = d.cpu().numpy()
    np_write(e, fp, ext='exr', silent=True)