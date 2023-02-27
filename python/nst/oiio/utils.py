"""
Repetitive utility functions that have nothing to do with style transfer
"""
import math
from typing import List
import subprocess
import shutil
import copy

import torch
from torchvision import transforms

from nst.core import utils as core_utils

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import os
import numpy as np
import numba
import kornia


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


def pts_to_exrs(outdir):
    pts = [x for x in os.listdir(outdir) if x.endswith('.pt')]
    for p in pts:
        fp = outdir + '/' + p
        pt_to_exr(fp)


def pt_to_exr(inpath):
    tensor = torch.load(inpath)
    buf = tensor_to_buf(tensor)
    outpath = inpath.replace('.pt', '.exr')
    write_exr(buf, outpath)


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


# def write_activation_atlas(vgg_layer_activations, layer_name, outdir):
#     fp = '%s/%s.exr' % (outdir, layer_name)
#
#     _, num_tiles, tile_size_x, tile_size_y = vgg_layer_activations.size()
#
#     num_x_tiles = int(round(math.sqrt(num_tiles)))
#     num_y_tiles = num_x_tiles
#
#     atlas_size_x = num_x_tiles * tile_size_x
#     atlas_size_y = num_y_tiles * tile_size_y
#
#     a = np.zeros((atlas_size_x, atlas_size_y, 3))
#
#     print(1.2, a.shape)
#
#     atlas_tile_x = 0
#     atlas_tile_y = 0
#     #
#     # for tile in range(0, num_tiles):
#     #     print(1.3)
#     #     for x in range(0, tile_size_x):
#     #         for y in range(0, tile_size_y):
#     #             value = vgg_layer_activations[0][tile][x][y]
#     #             atlas_x = x + atlas_tile_x
#     #             atlas_y = y + atlas_tile_y
#     #             try:
#     #                 print(1.6)
#     #                 a[atlas_x][atlas_y] = value
#     #             except:
#     #                 print(1.7)
#     #                 pass
#
#         # if atlas_tile_x >= atlas_size_x:
#         #     atlas_tile_x = 0
#         #     atlas_tile_y += tile_size_y
#         # elif atlas_tile_y >= atlas_size_y:
#         #     atlas_tile_y = 0
#         #     atlas_tile_x += tile_size_x
#         # else:
#         #     atlas_tile_x += tile_size_x
#
#     #np_write(a, filepath, ext='exr')


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


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.int64, numba.float32[:, :, :])],"(m,n,o),(m,n,o),()->(m,n,o)")
def _warp_np(col_np, vec_np, radius, output):
    x, y, z = vec_np.shape

    for old_x in range(0, x):
        for old_y in range(0, y):
            my, mx, _ = vec_np[old_x][old_y]  # flip axis
            nx = old_x + int(mx) # floor?
            ny = old_y + int(my) # floor?

            if nx not in range(0, x) or ny not in range(0, y):
                continue

            if radius:
                for px in range(nx - radius, nx + radius):
                    for py in range(ny - radius, ny + radius):
                        ox = old_x + px - nx
                        oy = old_y + py - ny
                        if px in range(0, x) and py in range(0, y):
                            if ox in range(0, x) and oy in range(0, y):
                                output[px][py] = col_np[ox][oy]
            else:
                output[nx][ny] = col_np[old_x][old_y]



def calc_flow_weights(motionFore, motionBack, start_frame, end_frame, step, output, radius=2):
    # output motionForeWeight, motionBackWeight
    # ie for a given frame's motionFore vector, create what is effectively an alpha channel for it
    _start_frame = motionFore.replace('####', '%04d' % start_frame)
    _start_buf = oiio.ImageBuf(_start_frame)
    np_array = _start_buf.get_pixels(roi=_start_buf.roi_full)
    x, y, z = np_array.shape

    black_img = np.zeros((x, y, z), dtype=np.float32)
    white_img = np.ones((x, y, z), dtype=np.float32)

    for i in range(start_frame, end_frame):
        j = i + step

        fore_flow = motionFore.replace('####', '%04d' % i) # eg 1010
        back_flow = motionBack.replace('####', '%04d' % j) # eg 1011

        # backwards->forwards aka disocclusionFore mask
        fore_warp = np.zeros((x, y, z), dtype=np.float32)
        _warp_np(white_img, fore_flow, radius, fore_warp)
        # now warp the fore warp backwards
        back_warp = np.zeros((x, y, z), dtype=np.float32)
        _warp_np(fore_warp, back_flow, radius, back_warp)

        # forewards-backwards aka disocclusionBack mask





    # renders/Studio/v011/data/motionFore/2048x858/acescg/exr/id01_060_Studio_data_motionFore_v011_acescg.####.exr
    # renders/Studio/v011/data/motionBack/2048x858/acescg/exr/id01_060_Studio_data_motionBack_v011_acescg.####.exr

    # ruder et al name the flo files with from->to embedded in the filename, eg:
    # for a step of 1, at frame 1010, the fore flo file would be:
    # foreward_1001_1002.flo
    # FROM 1001 TO 1002
    # the operative frame being 1002
    #
    # in prman, we don't control the "from", it is always the adjacent frame.
    # so for frame 1002:
    # * motionFore 1001 is a forward warp to this frame
    # * motionBack 1003 is a backward warp to this frame



def calc_disocclusion(mvec_fore, mvec_back, this_frame, output, step=1, radius=2):

    # note: numba vectorize requires numpy arrays
    mvec_back_frame = mvec_back.replace('####', '%04d' % this_frame)
    mvec_back_buf = oiio.ImageBuf(mvec_back_frame)
    mvec_back_np = mvec_back_buf.get_pixels(roi=mvec_back_buf.roi_full)

    x, y, z = mvec_back_np.shape

    back_warp_np = np.zeros((x, y, z), dtype=np.float32)
    back_col_np = np.ones((x, y, z), dtype=np.float32)

    render_input = back_col_np

    for s in range(0, step):
        _warp_np(render_input, mvec_back_np, radius, back_warp_np)
        render_input = copy.deepcopy(back_warp_np)

    back_frame = this_frame - step

    mvec_fore_frame = mvec_fore.replace('####', '%04d' % back_frame)
    mvec_fore_buf = oiio.ImageBuf(mvec_fore_frame)
    mvec_fore_np = mvec_fore_buf.get_pixels(roi=mvec_fore_buf.roi_full)

    fore_warp_np = np.zeros((x, y, z), dtype=np.float32)
    fore_col_np = back_warp_np

    render_input = fore_col_np

    for s in range(0, step):
        _warp_np(render_input, mvec_fore_np, radius, fore_warp_np)
        render_input = copy.deepcopy(fore_warp_np)

    # convert to pytorch tensor from here to avoid opencv 
    fore_warp_tensor = torch.Tensor(fore_warp_np)

    dilation_kernel = torch.ones(2, 2)
    kornia.morphology.dilation(fore_warp_tensor, dilation_kernel)

    erosion_kernel = torch.ones(3, 3)
    kornia.morphology.erosion(fore_warp_tensor, erosion_kernel)

    blur_kernel = (10, 10)
    blur_sigma = (10, 10)
    kornia.filters.gaussian_blur2d(fore_warp_tensor, blur_kernel, blur_sigma)

    np_write(fore_warp_np, output, ext='exr')


def warp(render_seq, mvec_seq, to_frame, output, step=1, radius=3):
    # render_seq = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/primary/beauty/2048x858/acescg/exr/id01_030_static_primary_beauty_v020_acescg_rgb.####.exr'
    # mvec_seq = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/data/motionFore/motionFore.####.exr'
    # output = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/data/motionFore/021.exr'
    # warp(render_seq, mvec_seq, 1020, output, step=1)

    # numba vectorize requires numpy arrays hence no tensors

    from_frame = to_frame - step

    render_from = render_seq.replace('####', '%04d' % from_frame)
    render_from_buf = oiio.ImageBuf(render_from)
    render_from_np = render_from_buf.get_pixels(roi=render_from_buf.roi_full)

    mvec_frame = mvec_seq.replace('####', '%04d' % from_frame)
    mvec_frame_buf = oiio.ImageBuf(mvec_frame)
    mvec_frame_np = mvec_frame_buf.get_pixels(roi=mvec_frame_buf.roi_full)

    output_np = copy.deepcopy(render_from_np)

    render_input = render_from_np

    for s in range(0, step):
        _warp_np(render_input, mvec_frame_np, radius, output_np)
        render_input = copy.deepcopy(output_np)

    np_write(output_np, output, ext='exr')





