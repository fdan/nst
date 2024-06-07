import re
import skimage.color as sc
import skimage.filters as sf
import skimage.morphology as sm
import numba
import math
import numpy as np
import OpenImageIO as oiio

from . import utils


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :])],"(a,b,c),(a,b,c),(a,b,c)->(a,b,c)")
def a_over_b(a, b, alpha, output):
    x, y, z = a.shape

    for i in range(0, x):
        for j in range(0, y):

            av = alpha[i][j][0]

            output[i][j] = (av*a[i][j]) + ((1-av)*b[i][j])



@numba.guvectorize([(numba.float32[:, :, :],
                     numba.float32[:, :, :],
                     numba.bool_[:, :],
                     numba.float32,
                     numba.float32[:, :, :])],
                   "(a,b,d),(a,b,c),(a,b),()->(a,b,d)")
def sample_back_nn(img_np, flow_np, check_np, boundary, output):
    """fore_flow_np is the motion back vectors, which maps pixels coordinates in the current frame
    to the previous frame.  eg image_np.1002, fore_flow_np.1003, will fill output.1003
    nearest neighbor interpolation

    to do: enforce a rule that an indice from img_np may only be sampled once
    """
    x, y, z = flow_np.shape

    for dest_x in range(0, x):
        for dest_y in range(0, y):

            # flip axes
            src_y, src_x, _ = flow_np[dest_x][dest_y]

            min_dx = math.floor(src_x)
            max_dx = min_dx + 1

            min_dy = math.floor(src_y)
            max_dy = min_dy + 1

            min_src_x = dest_x + min_dx
            min_src_y = dest_y + min_dy
            max_src_x = dest_x + max_dx
            max_src_y = dest_y + max_dy

            if min_src_x not in range(0, x) or min_src_y not in range(0, y) or max_src_x not in range(0, x) or max_src_y not in range(0, y):
                continue

            if min(dest_x, x - dest_x) < boundary or min(dest_y, y - dest_y) < boundary:
                if src_x != 0 and src_y != 0:
                    continue

            x1 = abs(min_dx - src_x)
            x2 = abs(max_dx - src_x)
            xo = min_dx + dest_x if x1 <= x2 else max_dx + dest_x

            y1 = abs(min_dy - src_y)
            y2 = abs(max_dy - src_y)
            yo = min_dy + dest_y if y1 <= y2 else max_dy + dest_y

            # mark xo, yo as False in our check array
            if not check_np[xo][yo]:
                output[dest_x][dest_y] = [-1.0, -1.0, -1.0]
                continue

            output[dest_x][dest_y] = img_np[xo][yo]
            check_np[xo][yo] = False


@numba.guvectorize([(numba.float32[:, :, :],
                     numba.float32[:, :, :],
                     numba.float32,
                     numba.float32[:, :, :])],
                   "(a,b,d),(a,b,c),()->(a,b,d)")
def sample_back_av(img_np, flow_np, boundary, output):
    """fore_flow_np is the motion back vectors, which maps pixels coordinates in the current frame
    to the previous frame.  eg image_np.1002, fore_flow_np.1003, will fill output.1003
    average interpolation
    """
    x, y, z = flow_np.shape

    for dest_x in range(0, x):
        for dest_y in range(0, y):

            # flip axes
            src_y, src_x, _ = flow_np[dest_x][dest_y]

            min_dx = math.floor(src_x)
            max_dx = min_dx + 1

            min_dy = math.floor(src_y)
            max_dy = min_dy + 1

            min_src_x = dest_x + min_dx
            min_src_y = dest_y + min_dy
            max_src_x = dest_x + max_dx
            max_src_y = dest_y + max_dy

            if min_src_x not in range(0, x) or min_src_y not in range(0, y) or max_src_x not in range(0, x) or max_src_y not in range(0, y):
                continue

            if min(dest_x, x - dest_x) < boundary or min(dest_y, y - dest_y) < boundary:
                if src_x != 0 and src_y != 0:
                    # c = np.zeros(3, np.float32)
                    continue

            c1 = img_np[min_src_x][min_src_y] * 0.25
            c2 = img_np[min_src_x][max_src_y] * 0.25
            c3 = img_np[max_src_x][min_src_y] * 0.25
            c4 = img_np[max_src_x][max_src_y] * 0.25

            output[dest_x][dest_y] = c1 + c2 + c3 + c4


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32, numba.float32[:, :, :])],
                   "(a,b,d),(a,b,c),()->(a,b,d)")
def sample_back_bl(img_np, flow_np, boundary, output):
    """fore_flow_np is the motion back vectors, which maps pixels coordinates in the current frame
    to the previous frame.  eg image_np.1002, fore_flow_np.1003, will fill output.1003
    bilinear interpolation
    """
    x, y, z = flow_np.shape

    for dest_x in range(0, x):
        for dest_y in range(0, y):

            # flip axes
            src_y, src_x, _ = flow_np[dest_x][dest_y]

            min_dx = math.floor(src_x)
            max_dx = min_dx + 1

            min_dy = math.floor(src_y)
            max_dy = min_dy + 1

            min_src_x = dest_x + min_dx
            min_src_y = dest_y + min_dy
            max_src_x = dest_x + max_dx
            max_src_y = dest_y + max_dy

            if min_src_x not in range(0, x) or min_src_y not in range(0, y) or max_src_x not in range(0, x) or max_src_y not in range(0, y):
                continue

            if min(dest_x, x - dest_x) < boundary or min(dest_y, y - dest_y) < boundary:
                if src_x != 0 and src_y != 0:
                    output[dest_x][dest_y] = np.zeros(3, np.float32)
                    continue
                    # c = np.zeros(3, np.float32)

            x1 = (abs(src_x - min_dx) + abs(src_y - min_dy)) / 4.
            x2 = (abs(src_x - min_dx) + abs(src_y - max_dy)) / 4.
            x3 = (abs(src_x - max_dx) + abs(src_y - min_dy)) / 4.
            x4 = (abs(src_x - max_dx) + abs(src_y - max_dy)) / 4.

            c1 = img_np[min_src_x][min_src_y] * x1
            c2 = img_np[min_src_x][max_src_y] * x2
            c3 = img_np[max_src_x][min_src_y] * x3
            c4 = img_np[max_src_x][max_src_y] * x4

            output[dest_x][dest_y] = c1 + c2 + c3 + c4


def sample_back(img_np, flow_np, boundary, output_np, interpolation='nn'):
    if interpolation == 'nn':
        x, y, z = img_np.shape
        check_np = np.ones((x, y), dtype=np.bool_)
        sample_back_nn(img_np, flow_np, check_np, boundary, output_np)
    elif interpolation == 'bl':
        sample_back_bl(img_np, flow_np, boundary, output_np)
    elif interpolation == 'av':
        sample_back_av(img_np, flow_np, boundary, output_np)
    return output_np


def sample_back_files(img, flow, boundary=0, interpolation='bl'):
    """
    given some filepaths, perform a simple sample back
    """
    img_buf = oiio.ImageBuf(img)
    img_np = img_buf.get_pixels(roi=img_buf.roi_full)
    x, y, z = img_np.shape

    flow_buf = oiio.ImageBuf(flow)
    flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

    output_np = np.zeros((x, y, z), dtype=np.float32)

    return sample_back(img_np, flow_np, boundary, output_np, interpolation=interpolation)


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32, numba.float32, numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c),(),()->(a,b,c)")
def compare_images_binary(img1, img2, rtol, atol, output):
    x, y, z = img1.shape

    for i in range(0, x):
        for j in range(0, y):
            img1_value = img1[i][j]
            img2_value = img2[i][j]

            if False in np.isclose(img1_value, img2_value, rtol=float(rtol), atol=atol):
                output[i][j] = [0.0, 0.0, 0.0]


# @numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32, numba.float32, numba.float32[:, :, :])],
#                    "(a,b,c),(a,b,c),(),()->(a,b,c)")
# def compare_images_soft(img1, img2, rtol, atol, output):
#     x, y, z = img1.shape
#
#     for i in range(0, x):
#         for j in range(0, y):
#             img1_value = img1[i][j]
#             img2_value = img2[i][j]
#
#             if False in np.isclose(img1_value, img2_value, rtol=float(rtol), atol=atol):
#                 diff = abs(img2_value - img1_value)
#                 output[i][j] = [diff, diff, diff]


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


def make_motion_mask_step(flow_path,
                          id_path,
                          render_path,
                          from_frame,
                          to_frame,
                          out_path,

                          debug_masks=False,
                          opposite_flow_path='',
                          direction=1,

                          id_erode=0,
                          id_area_threshold=20,
                          id_blur=3,
                          id_rtol=0.00001,
                          id_atol=0.01,

                          render_erode=0,
                          render_blur=3,
                          render_area_threshold=5,
                          render_rtol=0.3,
                          render_atol=0.00000001,

                          cycle_check_boundary=10,
                          cycle_check_rtol=0.6,
                          cycle_check_atol=0.6
                          ):

    flow_path = flow_path.replace('####', '%04d' % (to_frame))
    flow_buf = oiio.ImageBuf(flow_path)
    flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

    o_frame = 1 if direction else -1
    opp_flow_path = opposite_flow_path.replace('####', '%04d' % (to_frame - o_frame))
    opp_flow_buf = oiio.ImageBuf(opp_flow_path)
    opp_flow_np = opp_flow_buf.get_pixels(roi=opp_flow_buf.roi_full)

    x, y, z = flow_np.shape

    # -----------------------------------------------------------------------
    # id
    # -----------------------------------------------------------------------

    # id ref is the ground truth to compare our generated frame with
    id_ref_path_ = id_path.replace('####', '%04d' % to_frame)
    id_ref_buf = oiio.ImageBuf(id_ref_path_)
    id_ref_np = id_ref_buf.get_pixels(roi=id_ref_buf.roi_full)

    id_path = id_path.replace('####', '%04d' % from_frame)
    id_buf = oiio.ImageBuf(id_path)
    id_np = id_buf.get_pixels(roi=id_buf.roi_full)

    id_warp = np.zeros((x, y, z), dtype=np.float32)

    sample_back(id_np, flow_np, 10, id_warp, interpolation='nn')
    id_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(id_warp, id_ref_np, id_rtol, id_atol, id_compare)

    id_gray = sc.rgb2gray(id_compare)
    if id_area_threshold:
        id_gray = sm.area_opening(id_gray, area_threshold=id_area_threshold)
    if id_erode:
        id_gray = sm.erosion(id_gray, sm.disk(id_erode))
    if id_blur:
        id_gray = sf.gaussian(id_gray, sigma=id_blur, truncate=1.0)
    id_compare = sc.gray2rgb(id_gray)

    # -----------------------------------------------------------------------
    # render
    # -----------------------------------------------------------------------

    # render ref is the ground truch to compare our generated frame with
    render_ref_path_ = render_path.replace('####', '%04d' % to_frame)
    render_ref_buf = oiio.ImageBuf(render_ref_path_)
    render_ref_np = render_ref_buf.get_pixels(roi=render_ref_buf.roi_full)

    render_path = render_path.replace('####', '%04d' % from_frame)
    render_buf = oiio.ImageBuf(render_path)
    render_np = render_buf.get_pixels(roi=render_buf.roi_full)

    render_warp = np.zeros((x, y, z), dtype=np.float32)
    sample_back(render_np, flow_np, 10, render_warp, interpolation='nn')

    render_compare = np.ones((x, y, z), dtype=np.float32)
    compare_images(render_warp, render_ref_np, render_rtol, render_atol, render_compare)

    render_gray = sc.rgb2gray(render_compare)
    if render_erode:
        render_gray = sm.area_closing(render_gray, area_threshold=render_area_threshold)
    if render_erode:
        render_gray = sm.erosion(render_gray, sm.disk(render_erode))
    if render_blur:
        render_gray = sf.gaussian(render_gray, sigma=render_blur, truncate=1.0)
    render_compare = sc.gray2rgb(render_gray)

    # -----------------------------------------------------------------------
    # cycle check
    # -----------------------------------------------------------------------
    # cycle_check = np.ones((x, y, z), dtype=np.float32)
    # motion_cycle_check(flow_np, opp_flow_np, cycle_check_boundary,
    #                    cycle_check_rtol, cycle_check_atol, cycle_check)

    # -----------------------------------------------------------------------
    # combine
    # -----------------------------------------------------------------------
    # output = id_compare * render_compare * cycle_check
    output = id_compare * render_compare

    # -----------------------------------------------------------------------
    # write outputs
    # -----------------------------------------------------------------------
    if debug_masks:

        # id_warp_path = re.sub(r'(\.[0-9]{4}\.)', r'_idWarp\1', out_path)
        # utils.np_write(id_warp, id_warp_path, ext='exr')

        render_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_renderCheck\1', out_path)
        utils.np_write(render_compare, render_check_path, ext='exr')

        id_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_idCheck\1', out_path)
        utils.np_write(id_compare, id_check_path, ext='exr')

        # cycle_check_path = re.sub(r'(\.[0-9]{4}\.)', r'_cycleCheck\1', out_path)
        # utils.np_write(cycle_check, cycle_check_path, ext='exr')

    utils.np_write(output, out_path, ext='exr')



def make_motion_mask(id_,
                     fore_flow,
                     back_flow,
                     render,
                     mask_fore_out,
                     mask_back_out,
                     start,
                     end,
                     id_blur=0,
                     render_blur=0,
                     fore=True,
                     back=True,
                     debug_masks=False,
                     render_rtol=0.4,
                     render_atol=0.00000001,
                     render_erode=0,
                     render_area_threshold=5,
                     id_erode=0,
                     id_rtol=0.00001,
                     id_atol=0.01,
                     id_area_threshold=5):
    back_start = start + 1
    back_end = end + 1

    fore_frames = [x for x in range(start, end)]
    back_frames = [x for x in range(back_start, back_end)]

    if fore:
        print('fore')
        for frame in fore_frames:
            to_frame = frame+1
            from_frame = frame
            print('from frame:', from_frame, 'to frame:', to_frame)
            output_path = mask_fore_out.replace('####', '%04d' % (to_frame))
            make_motion_mask_step(back_flow,
                                  id_,
                                  render,
                                  from_frame,
                                  to_frame,
                                  output_path,
                                  debug_masks=debug_masks,
                                  id_blur=id_blur,
                                  render_blur=render_blur,
                                  render_rtol=render_rtol,
                                  render_atol=render_atol,
                                  render_erode=render_erode,
                                  render_area_threshold=render_area_threshold,
                                  id_area_threshold=id_area_threshold,
                                  id_atol=id_atol,
                                  id_rtol=id_rtol,
                                  id_erode=id_erode,
                                  direction=1,
                                  opposite_flow_path=fore_flow)

    if back:
        print('back')
        back_frames.reverse()
        for frame in back_frames:
            to_frame = frame-1
            from_frame = frame
            print('from frame:', from_frame, 'to frame:', to_frame)

            output_path = mask_back_out.replace('####', '%04d' % (to_frame))

            make_motion_mask_step(fore_flow, id_, render, from_frame, to_frame, output_path,
                                  debug_masks=True, id_blur=id_blur, render_blur=render_blur,
                                  render_rtol=render_rtol, direction=0, opposite_flow_path=back_flow)



@numba.guvectorize([(numba.float32[:, :, :],
                     numba.float32[:, :, :],
                     numba.float32,
                     numba.float32,
                     numba.float32,
                     numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c),(),(),()->(a,b,c)")
def motion_cycle_check(fore_flow_np, back_flow_np, boundary, rtol, atol, output):
    x, y, z = fore_flow_np.shape

    for start_x in range(0, x):
        for start_y in range(0, y):

            # flip axes
            back_y, back_x, _ = back_flow_np[start_x][start_y]

            # if start_x == 406 and start_y == 956:
            #     print('back flow:', back_x, back_y)
            # else:
            #     continue

            min_dx = math.floor(back_x)
            max_dx = min_dx + 1

            min_dy = math.floor(back_y)
            max_dy = min_dy + 1

            min_src_x = start_x + min_dx
            min_src_y = start_y + min_dy
            max_src_x = start_x + max_dx
            max_src_y = start_y + max_dy

            if min_src_x not in range(0, x) or min_src_y not in range(0, y) or max_src_x not in range(0, x) or max_src_y not in range(0, y):
                continue

            if min(start_x, x - start_x) < boundary or min(start_y, y - start_y) < boundary:
                if back_x != 0 and back_y != 0:
                    continue

            x1 = (abs(back_x - min_dx) + abs(back_y - min_dy)) / 4.
            x2 = (abs(back_x - min_dx) + abs(back_y - max_dy)) / 4.
            x3 = (abs(back_x - max_dx) + abs(back_y - min_dy)) / 4.
            x4 = (abs(back_x - max_dx) + abs(back_y - max_dy)) / 4.

            fore_1_y, fore_1_x, _ = fore_flow_np[min_src_x][min_src_y] * x1
            fore_2_y, fore_2_x, _ = fore_flow_np[min_src_x][max_src_y] * x2
            fore_3_y, fore_3_x, _ = fore_flow_np[max_src_x][min_src_y] * x3
            fore_4_y, fore_4_X, _ = fore_flow_np[max_src_x][max_src_y] * x4

            fore_y = fore_1_y + fore_2_y + fore_3_y + fore_4_y
            fore_x = fore_1_x + fore_2_x + fore_3_x + fore_4_X

            # print('fore flow:', fore_x, fore_y)
            # print('diff:', abs(fore_x+back_x), abs(fore_y+back_y))

            if np.isclose(abs(back_x), abs(fore_x), rtol=rtol, atol=atol):
                if np.isclose(abs(back_y), abs(fore_y), rtol=rtol, atol=atol):
                    continue

            output[start_x][start_y] = np.zeros(3, np.float32)










