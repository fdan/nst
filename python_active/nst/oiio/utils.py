"""
Repetitive utility functions
"""
import math
from typing import List
import subprocess
import shutil
import os

import kornia
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI

from nst.core import utils as core_utils


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


def style_image_to_tensors(image: str, do_cuda: bool, resize: float = None, colorspace=None) -> List[torch.Tensor]:
    tensors = []

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

    if buf.nchannels == 3:
        rgb_tensor = torch.Tensor(buf.get_pixels().copy())
        rgb_tensor = transform_image_tensor(rgb_tensor, do_cuda)
        #alpha_tensor = torch.zeros(0)
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


def pad_exrs(pad, outdir):
    exrs = [x for x in os.listdir(outdir) if x.endswith('.exr')]
    for exr in exrs:
        fp = outdir + '/' + exr
        pad_exr(pad, fp)


def pad_exr(pad, filepath):
    tensor = image_to_tensor(filepath, True)
    _, c, h, w = tensor.size()
    print(h, w)
    ph = int(pad * h) - h
    pw = int(pad * w) - w

    # ensure pad amounts are even:
    ph = int(math.ceil(ph / 2.) * 2.)
    pw = int(math.ceil(pw / 2.) * 2.)

    tensor = torch.nn.functional.pad(tensor, (pw, pw, ph, ph), 'reflect')
    buf = tensor_to_buf(tensor)
    outpath = filepath.replace('.exr', '.pad.exr')
    write_exr(buf, outpath)


def log_cuda_memory():
    max_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000000
    max_mem_cached = torch.cuda.max_memory_reserved(0) / 1000000
    print('memory used: %s of %s' % (max_mem_cached, max_memory))


def pts_to_exrs(outdir, raw=False, cleanup=False):
    pts = [x for x in os.listdir(outdir) if x.endswith('.pt')]
    for p in pts:
        fp = outdir + '/' + p
        pt_to_exr(fp, raw=raw, cleanup=cleanup)


def pt_to_exr(inpath, raw=False, cleanup=False):
    tensor = torch.load(inpath)
    buf = tensor_to_buf(tensor, raw=raw)
    outpath = inpath.replace('.pt', '.exr')
    write_exr(buf, outpath)
    if cleanup:
        os.remove(inpath)


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
        result = tensor.unsqueeze(0).cuda()
        return result
    else:
        result = tensor.unsqueeze(0)
        return result


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
    it = torch.Tensor(buf.get_pixels(roi=buf.roi_full))

    it = torch.transpose(it, 2, 0)
    it = torch.transpose(it, 2, 1)

    it = tforms(it)

    if do_cuda:
        device = core_utils.get_cuda_device()
        it = it.detach().to(torch.device(device))
        return it.unsqueeze(0).cuda()
    else:
        return it.unsqueeze(0)


def tensor_to_buf(tensor: torch.Tensor, raw=False) -> oiio.ImageBuf:
    tforms_ = []

    if not raw:
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

    for i in range(0, layer_limit):
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


def write_noise_img(x, y, filepath):
    t = torch.rand(3, x, y)
    t = torch.transpose(t, 2, 0)
    t = torch.transpose(t, 0, 1)
    t = t.contiguous()
    n = t.numpy()
    x, y, z = n.shape
    buf = oiio.ImageBuf(oiio.ImageSpec(y, x, z, oiio.FLOAT))
    buf.set_pixels(oiio.ROI(), n)
    write_exr(buf, filepath)
