"""
Repetitive utility functions that have nothing to do with style transfer
"""
import math

from . import entities

import copy
import subprocess
import shutil
import os
import random

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot

from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image

import cv2
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import os
import numpy as np
from operator import itemgetter


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

def normalise_weights(style_layers):
    for layer in style_layers:
        channels = entities.VGG.layers[layer]['channels']
        style_layers[layer]['weight'] = style_layers[layer]['weight'] * 1000.0 / channels ** 2


def random_crop_image(image):
    """
    Given a PIL.Image, crop it with a bbox of random location and size
    """
    x_size, y_size = image.size
    min_size = min(x_size, y_size)

    bbox_min_ratio = 0.2
    bbox_max_ratio = 0.7

    bbox_min = int(min_size * bbox_min_ratio)
    bbox_max = int(min_size * bbox_max_ratio)

    # bbox_min, bbox_max = 80, 400
    bbox_size = random.randrange(bbox_min, bbox_max)

    # the bbox_size determins where the center can be placed
    x_range = (0+(bbox_size/2), x_size-(bbox_size/2))
    y_range = (0 + (bbox_size / 2), y_size - (bbox_size / 2))

    bbox_ctr_x = random.randrange(x_range[0], x_range[1])
    bbox_ctr_y = random.randrange(y_range[0], y_range[1])

    bbox_left = bbox_ctr_x - (bbox_size/2)
    bbox_upper = bbox_ctr_y - (bbox_size/2)
    bbox_right = bbox_ctr_x + (bbox_size/2)
    bbox_lower = bbox_ctr_y + (bbox_size/2)

    return image.crop((bbox_left, bbox_upper, bbox_right, bbox_lower))


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


def image_to_tensor_old(image, do_cuda):
    """
    :param [PIL.Image]
    :return: [torch.Tensor]
    """
    tforms = transforms.Compose([transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255.)),
                               ])

    tensor = tforms(image)

    if do_cuda:
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)




def image_to_tensor(image: str, do_cuda: bool, resize:float=None, colorspace=None, sharpen:float=1.0) -> torch.Tensor:
        # note: oiio implicitely converts to 0-1 floating point data here regardless of format:
        buf = ImageBuf(image)

        o_width = buf.oriented_full_width
        o_height = buf.oriented_full_height

        if resize:
            n_width = int(float(o_width) * resize)
            n_height = int(float(o_height) * resize)
            # print(2.0, image, resize, n_width, n_height)
            buf = oiio.ImageBufAlgo.resize(buf, roi=ROI(0, n_width, 0, n_height, 0, 1, 0, 3))
        #
        # if sharpen:
        #     buf = oiio.ImageBufAlgo.unsharp_mask(buf, kernel="gaussian", width=50.0, contrast=1.0, threshold=0.0, roi=oiio.ROI.All, nthreads=0)
        #
        #     # based on the reduction in size, set an appropriate sharpening level for style transfer
        #     # "good" sharpen values
        #     #
        #     # 4k: 50
        #     # 2k: 27
        #     # 1k: 12.8
        #     # 512: 7.3
        #     # 256: 4.3
        #     #
        #     # close enough to say, for each halving of image.x, halve sharpen filter width.
        #     #
        #     # however the initial sharpen filter width for the highest mip needs to be eyeballed by the user.  generally for nst, you want "sharper than you think is necessary".

        if colorspace:
            if colorspace != 'srgb_texture':
                buf = oiio.ImageBufAlgo.colorconvert(buf, colorspace, 'srgb_texture')

        return buf_to_tensor(buf, do_cuda)

def PIL_to_tensor(image, do_cuda):
    """
    :param [PIL.Image]
    :return: [torch.Tensor]
    """
    # deprecated: don't perform a resize

    tforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    tensor = tforms(image)

    if do_cuda:
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)


def buf_to_tensor(buf: oiio.ImageBuf, do_cuda: bool) -> torch.Tensor:

    tforms_ = []

    #  turn to BGR
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

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
        device = get_cuda_device()
        it = it.detach().to(torch.device(device))
        return it.unsqueeze(0).cuda()
    else:
        return it.unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image:

    tforms_ = []

    # convert to int
    tforms_ += [transforms.Lambda(lambda x: x.mul_(1. / 255.))]

    # add imagenet mean
    tforms_ += [transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1])]

    # turn to RGB
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    prep_pil = transforms.Compose(tforms_)
    to_pil = transforms.Compose([transforms.ToPILImage()])

    t = prep_pil(tensor.data[0].cpu().squeeze())
    t[t > 1] = 1
    t[t < 0] = 0
    out_img = to_pil(t)

    return out_img


def tensor_to_buf(tensor: torch.Tensor) -> oiio.ImageBuf:
    tforms_ = []

    tforms_ += [transforms.Lambda(lambda x: x.mul_(1. / 255.))]

    # add imagenet mean
    tforms_ += [transforms.Normalize(mean=[(-0.40760392), -0.45795686, -0.48501961], std=[1, 1, 1])]

    # turn to RGB
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    tforms = transforms.Compose(tforms_)

    t = tforms(tensor.data[0].cpu().squeeze())

    # t[t > 1] = 1 # clamp
    # t[t < 0] = 0 # clamp

    t = torch.transpose(t, 2, 0)
    t = torch.transpose(t, 0, 1)
    t = t.contiguous()
    n = t.numpy()
    x, y, z = n.shape
    buf = ImageBuf(ImageSpec(y, x, z, oiio.FLOAT))
    buf.set_pixels(ROI(), n)
    return buf


def get_cuda_device() -> str:
    cuda_device = 'cuda:%s' % torch.cuda.current_device()
    if not cuda_device:
        raise Exception('no cuda device found')
    return cuda_device


def write_exr(exr_buf: oiio.ImageBuf, filepath: str) -> None:
    exr_buf.write(filepath, oiio.FLOAT)


def layer_to_image(tensor):

    s = tensor.size()[-1]
    t_ = torch.empty(1, 3, s, s)
    t_[0][0] = tensor
    t_[0][1] = tensor
    t_[0][2] = tensor


    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])

    # what's this do?
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor.data[0].cpu().squeeze())
    t[t > 1] = 1
    t[t < 0] = 0
    out_img = postpb(t)
    return out_img


def annotate_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 30)
    draw.text((0, 0), text, (255, 255, 255), font=font)
    return image


def render_image(tensor, filepath, text=None):

    out_img = tensor_to_image(tensor)
    if text:
        annotate_image(out_img, text)
    out_img.save(filepath)


def get_full_path(filename):
    if not filename.startswith('/'):
        return os.getcwd() + '/' + filename
    return filename


def graph_loss(loss_graph, output_dir):
    pyplot.plot(loss_graph[0], loss_graph[1])
    pyplot.xlabel('iterations')
    pyplot.ylabel('loss')
    loss_graph_filepath = output_dir + '/loss.png'
    pyplot.savefig(loss_graph_filepath)


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


def zoom_image(img, zoom, rescale, cuda=False):
    # if zoom == 1.0:
    #     return img

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

    print('x_tile:', x_tile, 'y_tile:', y_tile)

    img = img.tile((x_tile, y_tile))

    # crop to old size
    buf = tensor_to_buf(copy.deepcopy(img))
    roi = oiio.ROI(int(0), int(old_width), int(0), int(old_height))
    buf = oiio.ImageBufAlgo.crop(buf, roi=roi)
    img = buf_to_tensor(buf, cuda)

    return img


def centre_crop_image(img, zoom, rescale, cuda=False, zoom_factor=0.17):
    # _, _, old_x, old_y = img.size()
    #
    # if zoom != 1.0:
    #     print(2.1)
    #     zoom_ = 1+(zoom*zoom_factor)
    #     crop_width = old_x / zoom_
    #     crop_height = old_y / zoom_
    #     left = (old_x - crop_width) / 2.
    #     right = crop_width + left
    #     bottom = (old_x - crop_height) / 2.
    #     top = bottom + crop_height
    #     buf = tensor_to_buf(copy.deepcopy(img))
    #     print(old_x, old_y, left, right, bottom, top)
    #     roi = oiio.ROI(int(left), int(right), int(bottom), int(top))
    #     buf = oiio.ImageBufAlgo.crop(buf, roi=roi)
    #     img = buf_to_tensor(buf, cuda)
    #
    # print(2.2, img.size())
    #
    # out_x = int(old_x * rescale)
    # out_y = int(old_y * rescale)
    # print(2.3, img.size(), out_x, out_y)
    # print(2.4, img)
    # img = torch.nn.functional.interpolate(img, size=(out_x, out_y))
    # print(2.5, img)
    return img


class Pyramid(object):

    gauss_downsample_scale = 0.63
    crop_scale = 0.9

    @classmethod
    def make_gaussian_pyramid(cls, img, mips=5, cuda=True):
        kernel = cls._build_gauss_kernel(cuda)
        gaus_pyramid = cls._gaussian_pyramid(img, kernel, cuda, max_levels=mips)
        return gaus_pyramid

    @classmethod
    def write_gaussian_pyramid(cls, outdir, img, mips=5, cuda=True):
        gaus_pyramid = cls.make_gaussian_pyramid(img, mips=mips, cuda=cuda)
        for index, level in enumerate(gaus_pyramid):

            try:
                os.makedirs(outdir)
            except:
                pass

            fp = outdir + '/gaus_pyr_lvl_%s.exr' % index
            buf = tensor_to_buf(level)
            write_exr(buf, fp)


    @classmethod
    def make_crop_pyramid(cls, img, mips=5, cuda=True):
        crop_pyramid = cls._crop_pyramid(img, cuda, max_levels=mips)
        return crop_pyramid

    @classmethod
    def write_crop_pyramid(cls, outdir, img, mips=5, cuda=True):
        crop_pyramid = cls._crop_pyramid(img.detach(), cuda, max_levels=mips, outdir=outdir)

        for index, level in enumerate(crop_pyramid):
            try:
                os.makedirs(outdir)
            except:
                pass

            fp = outdir + '/crop_pyr_lvl_%s.exr' % index
            print('writing ', fp)
            buf = tensor_to_buf(level)
            write_exr(buf, fp)

    @classmethod
    def _crop_pyramid(cls, img, cuda, max_levels, outdir=''):
        pyr = []
        pyr.append(copy.deepcopy(img))

        for level in range(0, max_levels-1):
            b, c, old_width, old_height = img.size()
            crop_width = old_width * cls.crop_scale
            crop_height = old_height * cls.crop_scale
            left = (old_width - crop_width) / 2.
            right = crop_width + left
            bottom = (old_height - crop_height) / 2.
            top = bottom + crop_height
            buf = tensor_to_buf(copy.deepcopy(img))
            roi = oiio.ROI(int(left), int(right), int(bottom), int(top))
            buf = oiio.ImageBufAlgo.crop(buf, roi=roi)
            img = buf_to_tensor(buf, cuda)
            pyr.append(copy.deepcopy(img))

        return pyr

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
    def _conv_gauss(img, kernel, cuda):
        """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        result = F.conv2d(img, kernel, groups=n_channels)

        if cuda:
            result = result.detach().to(torch.device(get_cuda_device()))

        return result

    @classmethod
    def _gaussian_pyramid(cls, img, kernel, cuda, max_levels):
        current = img
        pyr = [current]

        for level in range(0, max_levels-1):
            filtered = cls._conv_gauss(current, kernel, cuda)
            scale =cls.gauss_downsample_scale
            current = F.interpolate(filtered, scale_factor=scale)

            if cuda:
                current = current.detach().to(torch.device(get_cuda_device()))

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


