import os
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import numpy as np


def reverse_warp_vectors1(position, img_dict, out_path):
    pos_buf = oiio.ImageBuf(position)
    pos_np = pos_buf.get_pixels(roi=pos_buf.roi_full)

    x, y, z = pos_np.shape
    rev_warp = np.zeros((x, y, z))

    for x_ in range(0, x):
        for y_ in range(0, y):
            if not pos_np[x_][y_].any():
                continue
            key = '%s,%s,%s' % (x_, y_, 0)
            try:
                p_loc = np.array([float(b) for b in key.split(',')])
                fore_loc = np.array([float(a) for a in img_dict[key] + [0]])
            except KeyError:
                continue

            vec = p_loc - fore_loc

            try:
                rev_warp[x_][y_] = np.array([0. - vec[1], vec[0], 0])
            except:
                pass

    np_write(rev_warp, out_path)


def reverse_warp_vectors1(img_dict, x, y, z, out_path):
    """
    Given a warped position image, and

    """
    rev_warp = np.zeros((x, y, z))

    for x_ in range(0, x):
        for y_ in range(0, y):
            key = '%s,%s,%s' % (x_, y_, 0)
            p_loc = np.array([float(b) for b in key.split(',')])
            try:
                fore_loc = np.array([float(a) for a in img_dict[key]])
            except KeyError:
                fore_loc = np.array(x_, y_)

            vec = p_loc - fore_loc
            rev_warp[x_][y_] = np.array([0. - vec[1], vec[0], 0])

    np_write(rev_warp, out_path)


def reverse_warp_vectors(img_dict, x, y, z, out_path):
    """
    Given a warped position image, and

    """
    rev_warp = np.zeros((x, y, z))

    for x_ in range(0, x):
        for y_ in range(0, y):

            # the key is the pixel values of the warped p.  it's value gives us the ability to
            # locate where the pixel has been moved to.
            key = '%s,%s,%s' % (x_, y_, 0)

            try:
                # this is where pixel [x_, y_] has been shifted to in the warped image
                vec_loc = np.array([float(a) for a in img_dict[key]])
            except KeyError:
                # in this case we just say no shift
                vec_loc = np.array((x_, y_))

            # how to get from [x_, y_] to the warped location
            # delta = vec_loc - np.array([x_, y_])
            delta = np.array([x_, y_]) - vec_loc
            # now we need to encode this as the image color

            mvec_r = delta[1]
            mvec_g = delta[0]

            rev_warp[x_][y_] = [mvec_r, mvec_g, 0]

    np_write(rev_warp, out_path)


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