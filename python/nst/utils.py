"""
Repetitive utility functions that have nothing to do with style transfer
"""
from . import entities

import subprocess
import shutil
import os
import random

import torch
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from PIL import ImageFont
from PIL import ImageDraw

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


def image_to_tensor(image, do_cuda):
    """
    :param [PIL.Image]
    :return: [torch.Tensor]
    """
    # pre and post processing for images
    img_size = 512

    tforms = transforms.Compose([transforms.Resize(img_size),
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

    t = postpa(t.data[0].cpu().squeeze())
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

    if not temp_dir:
        temp_dir = '%s/tmp' % output_dir

    ffmpeg_cmd = []
    ffmpeg_cmd += ['ffmpeg', '-i', '%s/render.%%04d.png' % temp_dir]
    ffmpeg_cmd += ['-c:v', 'libx264', '-crf', '15', '-y']
    ffmpeg_cmd += ['%s/prog.mp4' % output_dir]
    subprocess.check_output(ffmpeg_cmd)
    shutil.rmtree(temp_dir)
