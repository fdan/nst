import math
import random
import os
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import numpy as np
import os
import cv2
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import numpy as np
from operator import itemgetter
from PIL import Image


def apply_patch(patch, target, x, y, brush=None):
    """
    Apply a patch to a target
    """
    t_width, t_height, _ = target.shape
    patch_radius = int(float(patch.shape[0]) / 2.0)

    # find the position in the target image
    t_x_min = x - patch_radius
    t_x_max = x + patch_radius
    t_y_min = y - patch_radius
    t_y_max = y + patch_radius

    patch_px = 0
    for x_ in range(t_x_min, t_x_max):
        patch_py = 0
        for y_ in range(t_y_min, t_y_max):

            try:
                patch_pvalue = patch[patch_px][patch_py]

                if patch_pvalue.sum() == 0.:
                    continue

                if brush.any():
                    brush_p_val = brush[patch_px][patch_py]
                    if np.count_nonzero(brush_p_val) == 3:
                        target[x_][y_] = patch_pvalue

            except IndexError:
                continue
            finally:
                patch_py += 1

        patch_px += 1


def get_patch(output, px, py, radius=2):
    """
    Given a pixel, return a patch of neighbors in a square shape
    """
    o_x, o_y, _ = output.shape

    patch = np.zeros((2 * radius, 2 * radius, 3))

    pxmin = px - radius
    pxmax = px + radius
    pymin = py - radius
    pymax = py + radius

    patch_px = 0
    for x in range(pxmin, pxmax):
        patch_py = 0
        for y in range(pymin, pymax):
            try:
                patch[patch_px][patch_py] = output[x][y]
            except IndexError:
                continue
            finally:
                patch_py += 1
        patch_px += 1

    return patch


# def get_patch_density(image_np, x, y):
#     patch = get_patch(image_np, x, y, radius=4)
#
#     area = float(x) * float(y)
#     sum = float(patch.sum())
#
#     density = area / sum
#
#     return density
#
#
# def cull_sparse_pixels(x, y, pixels, threshold):
#     for out_x in range(0, x):
#         for out_y in range(0, y):
#             d = get_patch_density()
#
#     pass


def _get_pixels_to_warp(render_np_prev, p_warp_dict, id_current, id_prev):
    """
    Given the prior rendered frame, and a warp P image dict, determine
    where the pixels of the frame should be shifted to.

    Return a list of pixels with information about their new destination.


    """
    print('getting pixels to warp')
    x, y, z = render_np_prev.shape
    pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):

            # the value we want to shift
            render_value = render_np_prev[old_x][old_y]

            # to do: get depth too?  sort by depth?

            # the pixel value is just the coordinates.:
            warp_key = '%s,%s,%s' % (old_x, old_y, 0)

            try:
                warp_loc = np.array([float(a) for a in p_warp_dict[warp_key]])
            except KeyError:
                # some warped pixels are occluded by other pixels, in case we can't find them
                continue

            new_x = int(warp_loc[0])
            new_y = int(warp_loc[1])

            # ensure new pixels are within range
            if new_x == x or new_y == y:
                continue

            _id_prev = id_prev[old_x][old_y]
            _id_current = id_current[new_x][new_y]

            if (_id_prev[0], _id_prev[1], _id_prev[2]) != (_id_current[0], _id_current[1], _id_current[2]):
                continue

            render_mag = render_value[0] + render_value[1] + render_value[2]
            pixels_to_warp.append((old_x, old_y, new_x, new_y, (render_value[0], render_value[1], render_value[2]), render_mag))

    print('sorting pixels to warp')
    sorted_pixels_to_warp = sorted(pixels_to_warp, key=itemgetter(5))

    return sorted_pixels_to_warp


def merge(fore_frame, fore_weight, back_frame, back_weight, nst_render, out_path):

    fore_buf = oiio.ImageBuf(fore_frame)
    fore_np = fore_buf.get_pixels(roi=fore_buf.roi_full)

    back_buf = oiio.ImageBuf(back_frame)
    back_np = back_buf.get_pixels(roi=back_buf.roi_full)

    nst_render_buf = oiio.ImageBuf(nst_render)
    nst_render_np = nst_render_buf.get_pixels(roi=nst_render_buf.roi_full)

    x, y, z = fore_np.shape

    # copy fore values if non-zero
    for x_ in range(0, x):
        for y_ in range(0, y):
            f_ = fore_np[x_][y_]
            if not f_.any():
                continue
            nst_render_np[x_][y_] = fore_np[x_][y_]

    # # copy back values if non-zero
    # for x_ in range(0, x):
    #     for y_ in range(0, y):
    #         f_ = back_np[x_][y_]
    #         if not f_.any():
    #             continue
    #         nst_render_np[x_][y_] = back_np[x_][y_]

    np_write(nst_render_np, out_path)


def warp_frame(render_fore_prior, mvec, out_path, id_current, id_prev, id_next_key, nst_render, next_key, back_mvecs,
               render_back_prior, fore_weight, back_weight):
    """
    to do: rewrite to transform pixels from a warped P image rather than mvec.
    """
    print('warping input frame')

    mvec_buf = oiio.ImageBuf(mvec)
    mvec_np = mvec_buf.get_pixels(roi=mvec_buf.roi_full)

    x, y, z = mvec_np.shape

    back_mvec = np.zeros((x, y, z))

    for bm in back_mvecs:
        bm_buf = oiio.ImageBuf(bm)
        bm_np = bm_buf.get_pixels(roi=bm_buf.roi_full)
        back_mvec += bm_np

    id_current_buf = oiio.ImageBuf(id_current)
    id_current_np = id_current_buf.get_pixels(roi=id_current_buf.roi_full)

    id_prev_buf = oiio.ImageBuf(id_prev)
    id_prev_np = id_prev_buf.get_pixels(roi=id_prev_buf.roi_full)

    id_next_key_buf = oiio.ImageBuf(id_next_key)
    id_next_key_np = id_next_key_buf.get_pixels(roi=id_next_key_buf.roi_full)

    render_fore_prior_buf = oiio.ImageBuf(render_fore_prior)
    render_fore_prior_np = render_fore_prior_buf.get_pixels(roi=render_fore_prior_buf.roi_full)

    render_back_prior_buf = oiio.ImageBuf(render_back_prior)
    render_back_prior_np = render_back_prior_buf.get_pixels(roi=render_back_prior_buf.roi_full)

    nst_render_buf = oiio.ImageBuf(nst_render)
    nst_render_np = nst_render_buf.get_pixels(roi=nst_render_buf.roi_full)

    output = nst_render_np
    # output = np.zeros((x, y, z))

    fore_pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):

            try:
                render_value = render_fore_prior_np[old_x][old_y]
            except:
                print(135, old_x, old_y, render_back_prior_np.shape)

            try:
                mx_, my_, _ = mvec_np[old_x][old_y]
            except IndexError:
                continue

            # flip axes
            mx = int(my_)
            my = int(mx_)

            if mx == 0 and my == 0:
                continue

            nx = old_x + mx
            ny = old_y + my

            if nx >= x or ny >= y:
                continue

            _id_prev = id_prev_np[old_x][old_y]
            _id_current = id_current_np[nx][ny]

            if (_id_prev[0], _id_prev[1], _id_prev[2]) != (_id_current[0], _id_current[1], _id_current[2]):
                continue

            # move weakest moves first, strongest last.
            render_mag = render_value[0] + render_value[1] + render_value[2]
            fore_pixels_to_warp.append(
                (old_x, old_y, nx, ny, (render_value[0], render_value[1], render_value[2]), render_mag, mx_, my_, 'fore'))

    for sp in fore_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            output[nx][ny] = render_value
        except IndexError:
            continue

    cull_sparse_neighbors(x, y, output)
    culled_fore_pixels_to_warp = []

    # remove null values from sorted_pixels_to_warp
    for sp in fore_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        try:
            if not output[nx][ny].any():
                continue
            culled_fore_pixels_to_warp.append(sp)
        except IndexError:
            continue

    ######

    back_pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):
            render_value = render_back_prior_np[old_x][old_y]

            try:
                mx_, my_, _ = back_mvec[old_x][old_y]
            except IndexError:
                continue

            # flip axes
            mx = int(my_)
            my = int(mx_)

            if mx == 0 and my == 0:
                continue

            nx = old_x + mx
            ny = old_y + my

            if nx >= x or ny >= y:
                continue

            _id_next_key = id_next_key_np[old_x][old_y]
            _id_current = id_current_np[nx][ny]

            if (_id_next_key[0], _id_next_key[1], _id_next_key[2]) != (_id_current[0], _id_current[1], _id_current[2]):
                continue

            # move weakest moves first, strongest last.
            render_mag = render_value[0] + render_value[1] + render_value[2]
            back_pixels_to_warp.append(
                (old_x, old_y, nx, ny, (render_value[0], render_value[1], render_value[2]), render_mag, mx_, my_, 'back'))

    for sp in back_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            output[nx][ny] = render_value
        except IndexError:
            continue

    cull_sparse_neighbors(x, y, output)
    culled_back_pixels_to_warp = []

    # remove null values from sorted_pixels_to_warp
    for sp in back_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        try:
            if not output[nx][ny].any():
                continue
            culled_back_pixels_to_warp.append(sp)
        except IndexError:
            continue

    # reduce based on weight
    fore_weight_ = int(fore_weight * float(len(culled_fore_pixels_to_warp)))
    random.shuffle(culled_fore_pixels_to_warp)
    culled_fore_pixels_to_warp = culled_fore_pixels_to_warp[:fore_weight_:]

    back_weight_ = int(back_weight * float(len(culled_back_pixels_to_warp)))
    random.shuffle(culled_back_pixels_to_warp)
    culled_back_pixels_to_warp = culled_back_pixels_to_warp[:back_weight_:]

    # to do: apply the patches from each pic in a random fashion
    # maybe if the data structure included whether they were fore or back
    # we could add to one giant list and shuffle

    final_pixels_to_warp = culled_fore_pixels_to_warp + culled_back_pixels_to_warp
    random.shuffle(final_pixels_to_warp)

    for fp in final_pixels_to_warp:
        ox = fp[0]
        oy = fp[1]
        nx = fp[2]
        ny = fp[3]
        dir_ = fp[8]

        if dir_ == 'fore':
            patch = get_patch(render_fore_prior_np, ox, oy, radius=6)
        elif dir_ == 'back':
            patch = get_patch(render_back_prior_np, ox, oy, radius=6)
        else:
            continue

        apply_patch(patch, output, nx, ny)

    np_write(output, out_path)


def warp_back_frame_in_forward_pass(render_fore_prior, mvec, out_path, id_current, id_prev, id_next_key, nst_render, next_key, back_mvecs, render_back_prior):
    """
    to do: rewrite to transform pixels from a warped P image rather than mvec.
    """
    print('warping input frame')

    mvec_buf = oiio.ImageBuf(mvec)
    mvec_np = mvec_buf.get_pixels(roi=mvec_buf.roi_full)

    x, y, z = mvec_np.shape

    back_mvec = np.zeros((x, y, z))

    for bm in back_mvecs:
        bm_buf = oiio.ImageBuf(bm)
        bm_np = bm_buf.get_pixels(roi=bm_buf.roi_full)
        back_mvec += bm_np

    id_current_buf = oiio.ImageBuf(id_current)
    id_current_np = id_current_buf.get_pixels(roi=id_current_buf.roi_full)
    id_prev_buf = oiio.ImageBuf(id_prev)
    id_next_key_buf = oiio.ImageBuf(id_next_key)
    id_next_key_np = id_next_key_buf.get_pixels(roi=id_next_key_buf.roi_full)
    render_fore_prior_buf = oiio.ImageBuf(render_fore_prior)
    render_back_prior_buf = oiio.ImageBuf(render_back_prior)
    render_back_prior_np = render_back_prior_buf.get_pixels(roi=render_back_prior_buf.roi_full)
    nst_render_buf = oiio.ImageBuf(nst_render)

    # output = nst_render_np
    output = np.zeros((x, y, z))

    back_pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):
            render_value = render_back_prior_np[old_x][old_y]

            try:
                mx_, my_, _ = back_mvec[old_x][old_y]
            except IndexError:
                continue

            # flip axes
            mx = int(my_)
            my = int(mx_)

            if mx == 0 and my == 0:
                continue

            nx = old_x + mx
            ny = old_y + my

            if nx >= x or ny >= y:
                continue

            _id_next_key = id_next_key_np[old_x][old_y]
            _id_current = id_current_np[nx][ny]

            if (_id_next_key[0], _id_next_key[1], _id_next_key[2]) != (_id_current[0], _id_current[1], _id_current[2]):
                continue

            # move weakest moves first, strongest last.
            render_mag = render_value[0] + render_value[1] + render_value[2]
            back_pixels_to_warp.append(
                (old_x, old_y, nx, ny, (render_value[0], render_value[1], render_value[2]), render_mag, mx_, my_))

    for sp in back_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            output[nx][ny] = render_value
        except IndexError:
            continue

    cull_sparse_neighbors(x, y, output)
    culled_back_pixels_to_warp = []

    # remove null values from sorted_pixels_to_warp
    for sp in back_pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        try:
            if not output[nx][ny].any():
                continue
            culled_back_pixels_to_warp.append(sp)
        except IndexError:
            continue

    for cp in culled_back_pixels_to_warp:
        ox = cp[0]
        oy = cp[1]
        nx = cp[2]
        ny = cp[3]
        patch = get_patch(render_back_prior_np, ox, oy, radius=4)
        apply_patch(patch, output, nx, ny)

    np_write(output, out_path)


def cull_sparse_neighbors_2(output, density_threshold=0.5, patch_radius=4):
    x, y, z = output.shape
    new_output = np.zeros((x, y, z))
    area = math.pow((2.*float(patch_radius)), 2) * 3.

    for x_ in range(0, x):
        for y_ in range(0, y):
            patch = get_patch(output, x_, y_, radius=patch_radius)
            non_zero = np.count_nonzero(patch)

            if non_zero != 0:
                density = float(non_zero) / float(area)
                if density > density_threshold:
                    new_output[x_][y_] = output[x_][y_]

            if non_zero == 0:
                new_output[x_][y_] = output[x_][y_]

    return new_output


def cull_sparse_neighbors(x, y, output, min_neighbors=4):
    # do some analysis of output and cull certain pixels
    for out_x in range(0, x):
        for out_y in range(0, y):
            neighbors = []
            try:
                neighbors.append(output[out_x - 1][out_y])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x + 1][out_y])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x + 1][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])

            # determine how many neighbors are zero.  if below a threshold then reset this pixel to zero.
            num_neighbors = len([sum(x_) for x_ in neighbors if sum(x_) > 0])
            if num_neighbors < min_neighbors:
                output[out_x][out_y] = [0.0, 0.0, 0.0]


def p_warp_frame(prev_render, mvec, out_path, id_current, id_prev, nst_render, comp_background, brush=None):
    """
    to do: rewrite to transform pixels from a warped P image rather than mvec.
    """
    # print('warping input frame')
    # print('prev_render:', prev_render)
    # print('mvec:', mvec)
    # print('out_path:', out_path)
    # print('id_current:', id_current)
    # print('id_prev:', id_prev)
    # print('nst_render:', nst_render)

    id_current_buf = oiio.ImageBuf(id_current)
    id_current_np = id_current_buf.get_pixels(roi=id_current_buf.roi_full)

    id_prev_buf = oiio.ImageBuf(id_prev)
    id_prev_np = id_prev_buf.get_pixels(roi=id_prev_buf.roi_full)

    render_buf_prev = oiio.ImageBuf(prev_render)
    render_np_prev = render_buf_prev.get_pixels(roi=render_buf_prev.roi_full)

    nst_render_buf = oiio.ImageBuf(nst_render)
    nst_render_np = nst_render_buf.get_pixels(roi=nst_render_buf.roi_full)

    mvec_buf = oiio.ImageBuf(mvec)
    mvec_np = mvec_buf.get_pixels(roi=mvec_buf.roi_full)

    x, y, z = render_np_prev.shape

    if comp_background:
        output = nst_render_np
    else:
        output = np.zeros((x, y, z))


    pixels_to_warp = []

    for old_x in range(0, x):
        for old_y in range(0, y):
            render_value = render_np_prev[old_x][old_y]

            try:
                mx_, my_, _ = mvec_np[old_x][old_y]
            except IndexError:
                continue

            # flip axes
            mx = int(my_)
            my = int(mx_)

            if mx == 0 and my == 0:
                continue

            nx = old_x + mx
            ny = old_y + my

            if nx >= x or ny >= y:
                continue

            _id_prev = id_prev_np[old_x][old_y]
            _id_current = id_current_np[nx][ny]

            if (_id_prev[0], _id_prev[1], _id_prev[2]) != (_id_current[0], _id_current[1], _id_current[2]):
                continue

            # move weakest moves first, strongest last.
            render_mag = render_value[0] + render_value[1] + render_value[2]
            pixels_to_warp.append(
                (old_x, old_y, nx, ny, (render_value[0], render_value[1], render_value[2]), render_mag, mx_, my_))

    # warp pixels:
    for sp in pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            output[nx][ny] = render_value
        except IndexError:
            continue

    # cull_sparse_neighbors(x, y, output, min_neighbors=3)
    output = cull_sparse_neighbors_2(output, density_threshold=0.7, patch_radius=4)

    # remove null values from sorted_pixels_to_warp
    culled_pixels_to_warp = []
    for sp in pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        try:
            if not output[nx][ny].any():
                continue
            culled_pixels_to_warp.append(sp)
        except IndexError:
            continue

    # shuffle and lose some pixels since we are doing a patch warp and we don't need all of them
    # to do: there should be a relationship between brush size and the density of pixels needed
    # random.shuffle(culled_pixels_to_warp)
    # num_culled_pixels = len(culled_pixels_to_warp)
    # r_ = int(0.8 * float(num_culled_pixels))
    # culled_pixels_to_warp = culled_pixels_to_warp[:r_:]

    if brush:
        patch_radius = 4
        brush_buf = oiio.ImageBuf(brush)
        brush_buf = oiio.ImageBufAlgo.resize(brush_buf, roi=ROI(0, patch_radius*2, 0, patch_radius*2, 0, 1, 0, 3))
        brush = brush_buf.get_pixels(roi=brush_buf.roi_full)

    culled_pixels_to_warp = pixels_to_warp
    for cp in culled_pixels_to_warp:
        ox = cp[0]
        oy = cp[1]
        nx = cp[2]
        ny = cp[3]
        patch = get_patch(render_np_prev, ox, oy, radius=4)
        apply_patch(patch, output, nx, ny, brush=brush)

    np_write(output, out_path)


def p_warp_frame2(prev_render, p_warp_dict, out_path, id_current, id_prev, nst_render):
    """
    to do: rewrite to transform pixels from a warped P image rather than mvec.
    """
    print('warping input frame')
    id_current_buf = oiio.ImageBuf(id_current)
    id_current_np = id_current_buf.get_pixels(roi=id_current_buf.roi_full)

    id_prev_buf = oiio.ImageBuf(id_prev)
    id_prev_np = id_prev_buf.get_pixels(roi=id_prev_buf.roi_full)

    render_buf_prev = oiio.ImageBuf(prev_render)
    render_np_prev = render_buf_prev.get_pixels(roi=render_buf_prev.roi_full)

    nst_render_buf = oiio.ImageBuf(nst_render)
    nst_render_np = nst_render_buf.get_pixels(roi=nst_render_buf.roi_full)

    x, y, z = render_np_prev.shape
    # output = np.zeros((x, y, z))
    output = nst_render_np
    pixels_to_warp = _get_pixels_to_warp(render_np_prev, p_warp_dict,id_current_np, id_prev_np)

    print('iterating through pixels to warp')
    for sp in pixels_to_warp:
        nx = sp[2]
        ny = sp[3]
        render_value = sp[4]

        try:
            output[nx][ny] = render_value
        except IndexError:
            continue

    # do some analysis of output and cull certain pixels
    min_neighbors = 3
    for out_x in range(0, x):
        for out_y in range(0, y):

            neighbors = []

            try:
                neighbors.append(output[out_x - 1][out_y])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x + 1][out_y])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x + 1][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x - 1][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x][out_y + 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])
            try:
                neighbors.append(output[out_x][out_y - 1])
            except IndexError:
                neighbors.append([0.0, 0.0, 0.0])

            # determine how many neighbors are zero.  if below a threshold then reset this pixel to zero.
            num_neighbors = len([sum(x_) for x_ in neighbors if sum(x_) > 0])
            if num_neighbors < min_neighbors:
                output[out_x][out_y] = [0.0, 0.0, 0.0]

    culled_pixels_to_warp = []

    # remove null values from sorted_pixels_to_warp
    for sp in pixels_to_warp:
        nx = sp[2]
        ny = sp[3]

        try:
            if not output[nx][ny].any():
                continue
            culled_pixels_to_warp.append(sp)
        except IndexError:
            continue

    for cp in culled_pixels_to_warp:
        ox = cp[0]
        oy = cp[1]
        nx = cp[2]
        ny = cp[3]
        patch = get_patch(render_np_prev, ox, oy, radius=4)
        apply_patch(patch, output, nx, ny)

    np_write(output, out_path)


def average_p(fore_dict, fore_weight, back_dict, back_weight, x, y, z, out_path):
    """
    We are given two image dicts, for the fore warped and also back warped position image.
    The position image just encodes each pixel's location in x and y as red and green values.
    This means we can easily take a pixel from one image and find it's new position in the
    other image.

    The warped P for a given frame will have two versions: one created by a fore warp
    and the other by a back warp.

    The purpose of this function average between these two warps, such that they converge.

    What are we averaging and how do we do it?  And what are we outputting?

    We iterate through all the pixels in the image, each pixel x/y position being a key
    in the image dict.  For each pixel p in the position image, P, we can find it's warped
    position in the fore and also the back image.  The image dicts will return the new position
    as the dict value.

    note: this doesn't work, it just ensures nothing moves!  how to make work...
    """
    img_avg = np.zeros((x, y, z))
    for x_ in range(0, x):
        for y_ in range(0, y):
            key = '%s,%s,%s' % (x_, y_, 0)

            fore_loc = np.array(0)
            back_loc = np.array(0)

            try:
                # these are the locations the given pixel has shifted to in the for and back images:
                fore_loc = np.array([float(a) for a in fore_dict[key]])
            except KeyError:
                pass

            try:
                # these are the locations the given pixel has shifted to in the for and back images:
                back_loc = np.array([float(a) for a in back_dict[key]])
            except KeyError:
                pass

            # pixel is in both warps - compute average
            if fore_loc.any() and back_loc.any():
                fore_delta = (fore_loc - np.array([x_, y_])) * fore_weight
                back_delta = (back_loc - np.array([x_, y_])) * back_weight

                avg_delta = np.mean(np.array([fore_delta, back_delta]), axis=0)

            # pixel is in fore warp but not back - just use fore warp
            elif fore_loc.any() and not back_loc.any():
                fore_delta = (fore_loc - np.array([x_, y_]))
                avg_delta = fore_delta

            # pixel is in back warp but not fore - just use back warp
            elif not fore_loc.any() and back_loc.any():
                back_delta = (back_loc - np.array([x_, y_]))
                avg_delta = back_delta

            else:
                continue

            new_x = x_ + avg_delta[0]
            new_y = y_ + avg_delta[1]

            try:
                img_avg[int(new_x)][int(new_y)] = [x_, y_, 0]
            except IndexError:
                pass

    np_write(img_avg, out_path)


def build_image_dict(img_path):
    """
    Build a dict of an image, such that the pixel values are keys.
    Assumpe the incoming image is a simple positional image, where
    a pixels co-ordinates are also it's red and green values.

    This will give us the ability to lookup a particular unique
    pixel value in another image, and find it's location.
    """
    img_buf = oiio.ImageBuf(img_path)
    img_np = img_buf.get_pixels(roi=img_buf.roi_full)
    x, y, z = img_np.shape

    fore_dict = {}
    for x_ in range(0, x):
        for y_ in range(0, y):
            value = img_np[x_][y_]
            if not value.any():
                continue
            key = '%s,%s,%s' % (int(value[0]), int(value[1]), 0);
            fore_dict[key] = [x_, y_]
    return fore_dict


def generate_p(sparsity, x, y, z, out_path):
    img_sparse = np.zeros((x, y, z))
    for x_ in range(0, x):
        if x_ % sparsity:
            continue
        for y_ in range(0, y):
            if y_ % sparsity:
                continue
            g_val = float(y_)
            r_val = float(x_)
            img_sparse[x_][y_] = [r_val, g_val, 0.0]
    np_write(img_sparse, out_path)


def make_output_dirs(output):
    t_ = output.split('/')
    t_.pop()
    d_ = ('/'.join(t_))
    try:
        os.makedirs(d_)
    except:
        pass
    
    
def np_write(img, path):
    x, y, z = img.shape
    buf = ImageBuf(ImageSpec(y, x, z, oiio.FLOAT)) # flip x and y
    buf.set_pixels(ROI(), img)
    print('writing:', path)
    buf.write(path)


def warp_p(position, vec, out_path):
    pos_buf = oiio.ImageBuf(position)
    pos_np = pos_buf.get_pixels(roi=pos_buf.roi_full)
    vec_buf = oiio.ImageBuf(vec)
    vec_np = vec_buf.get_pixels(roi=vec_buf.roi_full)
    x, y, z = vec_np.shape
    output = np.zeros((x, y, z))

    for old_x in range(0, x):
        for old_y in range(0, y):
            p_img = pos_np[old_x][old_y]

            try:
                mx_, my_, _ = vec_np[old_x][old_y]
            except IndexError:
                continue

            # flip axes
            mx = int(my_)
            my = int(mx_)

            if mx == 0 and my == 0:
                continue

            nx = old_x + mx
            ny = old_y + my

            try:
                output[nx][ny] = p_img
            except IndexError:
                continue

    np_write(output, out_path)