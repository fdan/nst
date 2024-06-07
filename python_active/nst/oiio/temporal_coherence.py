import re
import skimage.color as sc
import skimage.filters as sf
import skimage.morphology as sm
import cv2
import numba
import kornia
import math
import numpy as np
import OpenImageIO as oiio
import torch

from . import utils


def depth_warp_files(img, flow, depth, output=None):
    img_buf = oiio.ImageBuf(img)
    img_np = img_buf.get_pixels(roi=img_buf.roi_full)

    flow_buf = oiio.ImageBuf(flow)
    flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

    depth_buf = oiio.ImageBuf(depth)
    depth_np = depth_buf.get_pixels(roi=depth_buf.roi_full)

    if output:
        output_buf = oiio.ImageBuf(output)
        output_np = output_buf.get_pixels(roi=output_buf.roi_full)
    else:
        x, y, z = flow_np.shape
        output_np = np.zeros((x, y, z), dtype=np.float32)

    depth_warp(img_np, flow_np, depth_np, output_np)

    return output_np


def depth_warp_step(img_fp, flow_fp, depth_fp, from_frame, to_frame, boundary=0):
    """
    Warp an input image a specified number of frames using a flow sequence
    """
    img_fp_ = img_fp.replace('####', '%04d' % from_frame)
    img_buf = oiio.ImageBuf(img_fp_)
    img_np = img_buf.get_pixels(roi=img_buf.roi_full)
    x, y, z = img_np.shape

    input_np = img_np
    frame = from_frame

    for step in range(from_frame, to_frame):
        out_np = np.zeros((x, y, z), dtype=np.float32)

        flow_fp_ = flow_fp.replace('####', '%04d' % (frame))
        flow_buf = oiio.ImageBuf(flow_fp_)
        flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

        depth_fp_ = depth_fp.replace('####', '%04d' % frame)
        depth_buf = oiio.ImageBuf(depth_fp_)
        depth_np = depth_buf.get_pixels(roi=depth_buf.roi_full)

        depth_warp(input_np, flow_np, depth_np, boundary, out_np)

        # the output of this step becomes the input of next step
        input_np = out_np.copy()

        frame += 1

    return out_np


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :],
                     numba.float32, numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c),(a,b,c),()->(a,b,c)")
def depth_warp(img_np, flow_np, depth_np, boundary, output):

    x, y, z = flow_np.shape
    d = math.ceil(depth_np.max())
    deep_img = np.zeros((x, y, d, z), dtype=np.float32)

    for old_x in range(0, x):
        for old_y in range(0, y):

            img_value = img_np[old_x][old_y]

            # flip axis
            flow_y, flow_x, _ = flow_np[old_x][old_y]

            min_depth = math.floor(depth_np[old_x][old_y][0])

            min_x = old_x + math.floor(flow_x)
            max_x = old_x + math.ceil(flow_x)
            min_y = old_y + math.floor(flow_y)
            max_y = old_y + math.ceil(flow_y)

            if min_x not in range(0, x) or min_y not in range(0, y) or max_x not in range(0, x) or max_y not in range(0, y):
                continue

            if min(old_x, x-old_x) < boundary or min(old_y, y-old_y) < boundary:
                if flow_x != 0 and flow_y != 0:
                    img_value = np.zeros(3, np.float32)

            deep_img[min_x][min_y][min_depth] = img_value
            deep_img[max_x][min_y][min_depth] = img_value
            deep_img[min_x][max_y][min_depth] = img_value
            deep_img[max_x][max_y][min_depth] = img_value

    for old_x in range(0, x):
        for old_y in range(0, y):
            v = deep_img[old_x][old_y]
            for v_ in range(0, v.shape[0]):
                if v[v_].sum() > 0.0:
                    output[old_x][old_y] = deep_img[old_x][old_y][v_]
                    break


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32, numba.float32, numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c),(),()->(a,b,c)")
def compare_images(img1, img2, rtol, atol, output):
    x, y, z = img1.shape

    for i in range(0, x):
        for j in range(0, y):
            img1_value = img1[i][j]
            img2_value = img2[i][j]

            if False in np.isclose(img1_value, img2_value, rtol=float(rtol), atol=atol):
                output[i][j] = [0.0, 0.0, 0.0]


# # not sure this is still necessary
# def make_motion_boundary(flow):
#     b, c, w, h = flow.shape
#     output = torch.ones(b, c, w, h)
#     laplacian = kornia.filters.laplacian(flow, 5, normalized=True)
#
#     @numba.guvectorize([(numba.float32[:, :, :, :], numba.float32[:, :, :, :])], "(m,node,o,p),(m,node,o,p)")
#     def _make_motion_mask(t, output):
#         b, c, w, h = t.shape
#         for i in range(0, w):
#             for j in range(0, h):
#                 lvr1 = t[0][2][i][j]
#                 if not -0.1 < lvr1 < 0.1:
#                     output[0][0][i][j] = 0.0
#                     output[0][1][i][j] = 0.0
#                     output[0][2][i][j] = 0.0
#
#                 lvr2 = t[0][1][i][j]
#                 if not -0.1 < lvr2 < 0.1:
#                     output[0][0][i][j] = 0.0
#                     output[0][1][i][j] = 0.0
#                     output[0][2][i][j] = 0.0
#
#     _make_motion_mask(laplacian, output)
#     return output


def make_motion_mask(depth_path,
                     flow_path,
                     id_path,
                     albedo_path,
                     render_path,
                     frame,
                     out_path,
                     debug_masks=False,
                     direction=1,

                     id_erode=10,
                     id_area_threshold=64,
                     id_connectivity=1,
                     id_blur=5,
                     id_rtol=1e-05,
                     id_atol=1e-02,

                     albedo_erode=10,
                     albedo_area_threshold=64,
                     albedo_connectivity=1,
                     albedo_blur=5,
                     albedo_rtol=1e-02,
                     albedo_atol=1e-01,

                     render_erode=3,
                     render_blur=4,
                     render_connectivity=1,
                     render_area_threshold=30,
                     render_rtol=0.3,
                     render_atol=1e-08):

    depth_path_ = depth_path.replace('####', '%04d' % frame)
    flow_path_ = flow_path.replace('####', '%04d' % frame)
    id_path_ = id_path.replace('####', '%04d' % frame)
    id_ref_path_ = id_path.replace('####', '%04d' % (frame + direction))
    albedo_path_ = albedo_path.replace('####', '%04d' % frame)
    albedo_ref_path_ = albedo_path.replace('####', '%04d' % (frame + direction))
    render_path_ = render_path.replace('####', '%04d' % frame)
    render_ref_path_ = render_path.replace('####', '%04d' % (frame + direction))

    depth_buf = oiio.ImageBuf(depth_path_)
    depth_np = depth_buf.get_pixels(roi=depth_buf.roi_full)

    flow_buf = oiio.ImageBuf(flow_path_)
    flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

    x, y, z = flow_np.shape

    #####################################################################
    # id
    #####################################################################
    id_buf = oiio.ImageBuf(id_path_)
    id_np = id_buf.get_pixels(roi=id_buf.roi_full)

    id_ref_buf = oiio.ImageBuf(id_ref_path_)
    id_ref_np = id_ref_buf.get_pixels(roi=id_ref_buf.roi_full)

    id_warp = np.zeros((x, y, z), dtype=np.float32)
    depth_warp(id_np, flow_np, depth_np, id_warp)

    id_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(id_warp, id_ref_np, id_rtol, id_atol, id_compare)

    id_gray = sc.rgb2gray(id_compare)
    # id_gray = sm.area_closing(id_gray, area_threshold=id_area_threshold, connectivity=id_connectivity)
    id_gray = sm.area_opening(id_gray, area_threshold=id_area_threshold, connectivity=id_connectivity)
    id_footprint = sm.disk(id_erode)
    id_gray = sm.erosion(id_gray, id_footprint)
    if id_blur:
        id_gray = sf.gaussian(id_gray, sigma=id_blur, truncate=1.0)
    id_compare = sc.gray2rgb(id_gray)

    #####################################################################
    # albedo
    #####################################################################
    # albedo_buf = oiio.ImageBuf(albedo_path_)
    # albedo_np = albedo_buf.get_pixels(roi=albedo_buf.roi_full)
    #
    # albedo_ref_buf = oiio.ImageBuf(albedo_ref_path_)
    # albedo_ref_np = albedo_ref_buf.get_pixels(roi=albedo_ref_buf.roi_full)
    #
    # albedo_warp = np.zeros((x, y, z), dtype=np.float32)
    # depth_warp(albedo_np, fore_flow_np, depth_np, albedo_warp)
    #
    # albedo_compare = np.ones((x, y, z), dtype=np.float32)
    # compare_images(albedo_warp, albedo_ref_np, albedo_rtol, albedo_atol, albedo_compare)
    #
    # albedo_gray = sc.rgb2gray(albedo_compare)
    # # albedo_gray = sm.area_closing(albedo_gray, area_threshold=albedo_area_threshold, connectivity=albedo_connectivity)
    # albedo_gray = sm.area_opening(albedo_gray, area_threshold=albedo_area_threshold, connectivity=albedo_connectivity)
    # albedo_footprint = sm.disk(albedo_erode)
    # albedo_gray = sm.erosion(albedo_gray, albedo_footprint)
    # if albedo_blur:
    #     albedo_gray = sf.gaussian(albedo_gray, sigma=albedo_blur, truncate=1.0)
    # albedo_compare = sc.gray2rgb(albedo_gray)
    

    #####################################################################
    # render
    #####################################################################
    render_buf = oiio.ImageBuf(render_path_)
    render_np = render_buf.get_pixels(roi=render_buf.roi_full)

    render_ref_buf = oiio.ImageBuf(render_ref_path_)
    render_ref_np = render_ref_buf.get_pixels(roi=render_ref_buf.roi_full)

    render_warp = np.zeros((x, y, z), dtype=np.float32)
    depth_warp(render_np, flow_np, depth_np, render_warp)

    render_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(render_warp, render_ref_np, render_rtol, render_atol, render_compare)

    render_gray = sc.rgb2gray(render_compare)
    render_gray = sm.area_closing(render_gray, area_threshold=render_area_threshold, connectivity=render_connectivity)
    render_gray = sm.area_opening(render_gray, area_threshold=render_area_threshold, connectivity=render_connectivity)
    render_footprint = sm.disk(render_erode)
    render_gray = sm.erosion(render_gray, render_footprint)
    if render_blur:
        render_gray = sf.gaussian(render_gray, sigma=render_blur, truncate=1.0)
    render_compare = sc.gray2rgb(render_gray)

    #####################################################################
    # combine
    #####################################################################

    # output = id_compare * albedo_compare * render_compare
    output = id_compare * render_compare
    # output = render_compare

    if debug_masks:
        render_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_renderCheck\1', out_path)
        utils.np_write(render_compare, render_check_path, ext='exr')

        # albedo_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_albedoCheck\1', out_path)
        # utils.np_write(albedo_compare, albedo_check_path, ext='exr')

        id_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_idCheck\1', out_path)
        utils.np_write(id_compare, id_check_path, ext='exr')

    utils.np_write(output, out_path, ext='exr')


def make_motion_mask_step(depth_path,
                          flow_path,
                          id_path,
                          render_path,
                          from_frame,
                          to_frame,
                          out_path,
                          debug_masks=False,

                          id_erode=10,
                          id_area_threshold=64,
                          id_connectivity=1,
                          id_blur=5,
                          id_rtol=1e-05,
                          id_atol=1e-02,

                          render_erode=3,
                          render_blur=4,
                          render_connectivity=1,
                          render_area_threshold=30,
                          render_rtol=0.3,
                          render_atol=1e-08):

    id_ref_path_ = id_path.replace('####', '%04d' % to_frame)
    render_ref_path_ = render_path.replace('####', '%04d' % to_frame)

    #####################################################################
    # id
    #####################################################################
    id_ref_buf = oiio.ImageBuf(id_ref_path_)
    id_ref_np = id_ref_buf.get_pixels(roi=id_ref_buf.roi_full)
    x, y, z = id_ref_np.shape

    id_warp = depth_warp_step(id_path, flow_path, depth_path, from_frame, to_frame)
    id_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(id_warp, id_ref_np, id_rtol, id_atol, id_compare)

    id_gray = sc.rgb2gray(id_compare)
    id_gray = sm.area_opening(id_gray, area_threshold=id_area_threshold, connectivity=id_connectivity)
    id_footprint = sm.disk(id_erode)
    id_gray = sm.erosion(id_gray, id_footprint)
    if id_blur:
        id_gray = sf.gaussian(id_gray, sigma=id_blur, truncate=1.0)
    id_compare = sc.gray2rgb(id_gray)

    #####################################################################
    # render
    #####################################################################
    render_ref_buf = oiio.ImageBuf(render_ref_path_)
    render_ref_np = render_ref_buf.get_pixels(roi=render_ref_buf.roi_full)

    render_warp = depth_warp_step(render_path, flow_path, depth_path, from_frame, to_frame)

    render_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(render_warp, render_ref_np, render_rtol, render_atol, render_compare)

    render_gray = sc.rgb2gray(render_compare)
    render_gray = sm.area_closing(render_gray, area_threshold=render_area_threshold, connectivity=render_connectivity)
    render_gray = sm.area_opening(render_gray, area_threshold=render_area_threshold, connectivity=render_connectivity)
    render_footprint = sm.disk(render_erode)
    render_gray = sm.erosion(render_gray, render_footprint)
    if render_blur:
        render_gray = sf.gaussian(render_gray, sigma=render_blur, truncate=1.0)
    render_compare = sc.gray2rgb(render_gray)

    #####################################################################
    # combine
    #####################################################################

    output = id_compare * render_compare

    if debug_masks:
        render_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_renderCheck\1', out_path)
        utils.np_write(render_compare, render_check_path, ext='exr')

        id_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_idCheck\1', out_path)
        utils.np_write(id_compare, id_check_path, ext='exr')

    utils.np_write(output, out_path, ext='exr')







#############################################################
### more or less deprecated stuff follows
#############################################################

# @numba.vectorize([(numba.float32[:, :, :], numba.float32[:, :, :])])
# def warp_image_cpu(image, flow):
#     x, y, z = image.shape
#
#     # init result
#     result = ''
#
#     for ay in range(0, y-1):
#         for ax in range(0, x-1):
#
#             bx = ax + flow[ax][ay][2]
#             by = ay + flow[ax][ay][1]
#
#             # interpolation
#             x1 = math.floor(bx)
#             y1 = math.floor(by)
#             x2 = x1 + 1
#             y2 = y1 + 1
#
#             if x1 < 0 or x2 >= x or y1 < 0 or y2 >= y:
#                 result[ax][ay] = 0.0
#                 continue
#
#             alphaX = bx - x1
#             alphaY = by - y1
#
#             ra = ((1.0 - alphaX) * image[x1][y1][0]) + (alphaX * image[x2][y1][0])
#             rb = ((1.0 - alphaX) * image[x1][y2][0]) + (alphaX * image[x2][y2][0])
#             red = ((1.0 - alphaY) * ra) + (alphaY * rb)
#
#             ga = ((1.0 - alphaX) * image[x1][y1][1]) + (alphaX * image[x2][y1][1])
#             gb = ((1.0 - alphaX) * image[x1][y2][1]) + (alphaX * image[x2][y2][1])
#             green = ((1.0 - alphaY) * ga) + (alphaY * gb)
#
#             ba = ((1.0 - alphaX) * image[x1][y1][2]) + (alphaX * image[x2][y1][2])
#             bb = ((1.0 - alphaX) * image[x1][y2][2]) + (alphaX * image[x2][y2][2])
#             blue = ((1.0 - alphaY) * ba) + (alphaY * bb)
#
#             result[ax][ay][0] = red
#             result[ax][ay][1] = green
#             result[ax][ay][2] = blue
#
#         return result
#
#
# def warp_image(image, flow, cuda):
#
#     result = torch.zeros(image.size())
#     img_np = image[0].transpose(0, 2).transpose(0, 1).numpy()
#     fore_flow_np = flow[0].transpose(0, 2).transpose(0, 1).numpy()
#
#     if cuda:
#         result_np = torch.zeros(img_np.shape).numpy()
#         warp_image_gpu(img_np, fore_flow_np, result_np)
#     else:
#         result_np = warp_image_cpu(image, flow)
#
#     result[0] = torch.Tensor(result_np).transpose(0, 1).transpose(0, 2)
#     return result
#
#
#
# def warp_image_cpu(image, flow):
#     raise NotImplementedError
#
#
# @numba.guvectorize([(numba.float32[:, :, :],
#                      numba.float32[:, :, :],
#                      numba.float32[:, :, :])],
#                      "(m,node,o),(m,node,o)->(m,node,o)")
# def warp_image_gpu(image, flow, result):
#     x, y, z = image.shape
#
#     for ay in range(0, y-1):
#         for ax in range(0, x-1):
#
#             bx = ax + flow[ax][ay][2]
#             by = ay + flow[ax][ay][1]
#
#             # interpolation
#             x1 = math.floor(bx)
#             y1 = math.floor(by)
#             x2 = x1 + 1
#             y2 = y1 + 1
#
#             if x1 < 0 or x2 >= x or y1 < 0 or y2 >= y:
#                 result[ax][ay] = 0.0
#                 continue
#
#             alphaX = bx - x1
#             alphaY = by - y1
#
#             ra = ((1.0 - alphaX) * image[x1][y1][0]) + (alphaX * image[x2][y1][0])
#             rb = ((1.0 - alphaX) * image[x1][y2][0]) + (alphaX * image[x2][y2][0])
#             red = ((1.0 - alphaY) * ra) + (alphaY * rb)
#
#             ga = ((1.0 - alphaX) * image[x1][y1][1]) + (alphaX * image[x2][y1][1])
#             gb = ((1.0 - alphaX) * image[x1][y2][1]) + (alphaX * image[x2][y2][1])
#             green = ((1.0 - alphaY) * ga) + (alphaY * gb)
#
#             ba = ((1.0 - alphaX) * image[x1][y1][2]) + (alphaX * image[x2][y1][2])
#             bb = ((1.0 - alphaX) * image[x1][y2][2]) + (alphaX * image[x2][y2][2])
#             blue = ((1.0 - alphaY) * ba) + (alphaY * bb)
#
#             result[ax][ay][0] = red
#             result[ax][ay][1] = green
#             result[ax][ay][2] = blue
#
#
# @numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :])],"(m,node,o),(m,node,o)->(m,node,o)")
# def compute_disocclusion(flow1, flow2, result):
#     # result is given as a white image.  we find regions to make black from
#     # disocclusion and motion boundaries.  this serves as an alpha for the flow.
#
#     x, y, z = flow1.shape
#
#     for ay in range(0, y-1):
#         for ax in range(0, x-1):
#
#             bx = ax + flow1[ax][ay][2]
#             by = ay + flow1[ax][ay][1]
#
#             # INTERPOLATION
#             x1 = math.floor(bx)
#             y1 = math.floor(by)
#             x2 = x1 + 1
#             y2 = y1 + 1
#
#             if x1 < 0 or x2 >= x or y1 < 0 or y2 >= y:
#                 result[ax][ay] = 0.0
#                 continue
#
#             alphaX = bx - x1
#             alphaY = by - y1
#
#             a = ((1.0 - alphaX) * flow2[x1][y1][2]) + (alphaX * flow2[x2][y1][2])
#             b = ((1.0 - alphaX) * flow2[x1][y2][2]) + (alphaX * flow2[x2][y2][2])
#             u = ((1.0 - alphaY) * a) + (alphaY * b)
#
#             a_ = ((1.0 - alphaX) * flow2[x1][y1][1]) + (alphaX * flow2[x2][y1][1])
#             b_ = ((1.0 - alphaX) * flow2[x1][y2][1]) + (alphaX * flow2[x2][y2][1])
#             v = ((1.0 - alphaY) * a_) + (alphaY * b_)
#
#             # END INTERPOLATION
#             # (u, v) is the interpolated value of flow2 at bx, by
#
#             cx = bx + u
#             cy = by + v
#
#             u2 = flow1[ax][ay][2]
#             v2 = flow1[ax][ay][1]
#
#             # (u2, v2) is the values of flow1 at ax, ay
#             # the simplistic criteria is that (u2, v2) == -(u, v)
#
#             # cx-ax is equivalent to u + u2?
#             # cy-ay is equivalent to v + v2?
#             # (u+v+u2+v2)*(u+v+u2+v2)
#
#             # if (pow(cx-ax, 2) + pow(cy-ay, 2)) >= ((0.01 * (pow(u2, 2) + pow(v2, 2) + pow(u, 2) + pow(v, 2)) + 0.5):
#             # if ((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01*(u2*u2 + v2*v2 + u*u + v*v) + 0.05:
#             if (cx-ax)**2 + (cy-ay)**2 >= 0.01 * (u**2 + v**2 + u2**2 + v2**2) + 0.05:
#                 result[ax][ay][0] = 0.0
#                 result[ax][ay][1] = 0.0
#                 result[ax][ay][2] = 0.0
#
#
# def get_motion_boundary(flow):
#     b, c, w, h = flow.shape
#     output = torch.ones(b, c, w, h)
#     laplacian = kornia.filters.laplacian(flow, 5, normalized=True)
#
#     @numba.guvectorize([(numba.float32[:, :, :, :], numba.float32[:, :, :, :])], "(m,node,o,p),(m,node,o,p)")
#     def _make_motion_mask(t, output):
#         b, c, w, h = t.shape
#         for i in range(0, w):
#             for j in range(0, h):
#                 lvr1 = t[0][2][i][j]
#                 if not -0.1 < lvr1 < 0.1:
#                     output[0][0][i][j] = 0.0
#                     output[0][1][i][j] = 0.0
#                     output[0][2][i][j] = 0.0
#
#                 lvr2 = t[0][1][i][j]
#                 if not -0.1 < lvr2 < 0.1:
#                     output[0][0][i][j] = 0.0
#                     output[0][1][i][j] = 0.0
#                     output[0][2][i][j] = 0.0
#
#     _make_motion_mask(laplacian, output)
#     # output = kornia.filters.median_blur(output, 5)
#     return output
#
#
# def calc_flow_weights(motion_fore, motion_back, start_frame, end_frame, step, output, write_intermediate=False):
#     # ie for a given frame's motionFore vector, create what is effectively an alpha channel for it
#     _start_frame = motion_fore.replace('####', '%04d' % start_frame)
#     np_array = image_to_tensor(_start_frame, False, raw=True)
#
#     b, c, h, w = np_array.shape
#
#     for i in range(start_frame, end_frame):
#         j = i + step
#
#         motion_back_disocc = np.ones((h, w, c), dtype=np.float32)
#         motion_fore_disocc = np.ones((h, w, c), dtype=np.float32)
#
#         back_flow = motion_back.replace('####', '%04d' % j) # eg 1011
#         fore_flow = motion_fore.replace('####', '%04d' % i) # eg 1010
#
#         fore_flow_tensor = image_to_tensor(fore_flow, False, raw=True)
#         back_flow_tensor = image_to_tensor(back_flow, False, raw=True)
#
#         # calculate motion boundaries via laplacian 2nd order derivatives
#         fore_boundary_tensor = get_motion_boundary(fore_flow_tensor)
#         back_boundary_tensor = get_motion_boundary(back_flow_tensor)
#
#         # calculate disocclusion via motion vector consistency check
#         fore_flow_np = fore_flow_tensor.numpy()[0]
#         fore_flow_np = fore_flow_np.transpose(1, 2, 0)
#
#         back_flow_np = back_flow_tensor.numpy()[0]
#         back_flow_np = back_flow_np.transpose(1, 2, 0)
#
#         back_disocc_np = compute_disocclusion(back_flow_np, fore_flow_np, motion_back_disocc)
#         fore_disocc_np = compute_disocclusion(fore_flow_np, back_flow_np, motion_fore_disocc)
#
#         back_disocc_tensor = torch.zeros(b, c, h, w)
#         back_disocc_tensor[0] = torch.Tensor(back_disocc_np.transpose(2, 0, 1))
#
#         fore_disocc_tensor = torch.zeros(b, c, h, w)
#         fore_disocc_tensor[0] = torch.Tensor(fore_disocc_np.transpose(2, 0, 1))
#
#         fore_combined = fore_disocc_tensor * fore_boundary_tensor
#         back_combined = back_disocc_tensor * back_boundary_tensor
#
#         fore_combined = kornia.filters.median_blur(fore_combined, 5)
#         back_combined = kornia.filters.median_blur(back_combined, 5)
#
#         # write to disk
#         back_combined_buf = tensor_to_buf(back_combined, raw=True)
#         write_exr(back_combined_buf, '%s/back_combined.%04d.exr' % (output, i))
#
#         fore_combined_buf = tensor_to_buf(fore_combined, raw=True)
#         write_exr(fore_combined_buf, '%s/fore_combined.%04d.exr' % (output, i))
#
#         if write_intermediate:
#             back_dissoc_buf = tensor_to_buf(back_disocc_tensor, raw=True)
#             write_exr(back_dissoc_buf,'%s/motion_back_disocc.%04d.exr' % (output, i))
#
#             fore_dissoc_buf = tensor_to_buf(fore_disocc_tensor, raw=True)
#             write_exr(fore_dissoc_buf, '%s/motion_fore_disocc.%04d.exr' % (output, i))
#
#             back_boundary_buf = tensor_to_buf(back_boundary_tensor, raw=True)
#             write_exr(back_boundary_buf, '%s/motion_back_boundary.%04d.exr' % (output, i))
#
#             fore_boundary_buf = tensor_to_buf(fore_boundary_tensor, raw=True)
#             write_exr(fore_boundary_buf, '%s/motion_fore_boundary.%04d.exr' % (output, i))
#

# def calc_disocclusion(mvec_fore, mvec_back, this_frame, output, step=1, radius=2):
#
#     # note: numba vectorize requires numpy arrays
#     mvec_back_frame = mvec_back.replace('####', '%04d' % this_frame)
#     mvec_back_buf = oiio.ImageBuf(mvec_back_frame)
#     mvec_back_np = mvec_back_buf.get_pixels(roi=mvec_back_buf.roi_full)
#
#     x, y, z = mvec_back_np.shape
#
#     back_warp_np = np.zeros((x, y, z), dtype=np.float32)
#     back_col_np = np.ones((x, y, z), dtype=np.float32)
#
#     render_input = back_col_np
#
#     for s in range(0, step):
#         _warp_np(render_input, mvec_back_np, radius, back_warp_np)
#         render_input = copy.deepcopy(back_warp_np)
#
#     back_frame = this_frame - step
#
#     mvec_fore_frame = mvec_fore.replace('####', '%04d' % back_frame)
#     mvec_fore_buf = oiio.ImageBuf(mvec_fore_frame)
#     mvec_fore_np = mvec_fore_buf.get_pixels(roi=mvec_fore_buf.roi_full)
#
#     fore_warp_np = np.zeros((x, y, z), dtype=np.float32)
#     fore_col_np = back_warp_np
#
#     render_input = fore_col_np
#
#     for s in range(0, step):
#         _warp_np(render_input, mvec_fore_np, radius, fore_warp_np)
#         render_input = copy.deepcopy(fore_warp_np)
#
#     # convert to pytorch tensor from here to avoid opencv
#     fore_warp_tensor = torch.Tensor(fore_warp_np)
#
#     dilation_kernel = torch.ones(2, 2)
#     kornia.morphology.dilation(fore_warp_tensor, dilation_kernel)
#
#     erosion_kernel = torch.ones(3, 3)
#     kornia.morphology.erosion(fore_warp_tensor, erosion_kernel)
#
#     blur_kernel = (10, 10)
#     blur_sigma = (10, 10)
#     kornia.filters.gaussian_blur2d(fore_warp_tensor, blur_kernel, blur_sigma)
#
#     np_write(fore_warp_np, output, ext='exr')


# # note: deprecated
# def warp(render_seq, mvec_seq, to_frame, output, step=1, radius=3):
#     # render_seq = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/primary/beauty/2048x858/acescg/exr/id01_030_static_primary_beauty_v020_acescg_rgb.####.exr'
#     # mvec_seq = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/data/motionFore/motionFore.####.exr'
#     # output = '/mnt/ala/mav/2021/wip/s121/sequences/id01/id01_030/light/lighting/daniel.student/katana/renders/static/v020/data/motionFore/021.exr'
#     # warp(render_seq, mvec_seq, 1020, output, step=1)
#
#     # numba vectorize requires numpy arrays hence no tensors
#
#     from_frame = to_frame - step
#
#     render_from = render_seq.replace('####', '%04d' % from_frame)
#     render_from_buf = oiio.ImageBuf(render_from)
#     render_from_np = render_from_buf.get_pixels(roi=render_from_buf.roi_full)
#
#     mvec_frame = mvec_seq.replace('####', '%04d' % from_frame)
#     mvec_frame_buf = oiio.ImageBuf(mvec_frame)
#     mvec_frame_np = mvec_frame_buf.get_pixels(roi=mvec_frame_buf.roi_full)
#
#     output_np = copy.deepcopy(render_from_np)
#
#     render_input = render_from_np
#
#     for s in range(0, step):
#         _warp_np(render_input, mvec_frame_np, radius, output_np)
#         render_input = copy.deepcopy(output_np)
#
#     np_write(output_np, output, ext='exr')
#
#



