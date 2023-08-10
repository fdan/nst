import cv2
import numba
import kornia
import math
import numpy as np
import OpenImageIO as oiio
import torch


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c),(a,b,c)->(a,b,c)")
def depth_warp(img_np, flow_np, depth_np, output):
    x, y, z = flow_np.shape
    d = math.ceil(depth_np.max())
    deep_img = np.zeros((x, y, d, z), dtype=np.float32)

    for old_x in range(0, x):
        for old_y in range(0, y):
            img_value = img_np[old_x][old_y]

            # flip axis
            flow_y, flow_x, _ = flow_np[old_x][old_y]

            depth = depth_np[old_x][old_y][0]

            min_depth = math.floor(depth_np[old_x][old_y][0])

            min_x = old_x + math.floor(flow_x)
            max_x = old_x + math.ceil(flow_x)
            min_y = old_y + math.floor(flow_y)
            max_y = old_y + math.ceil(flow_y)

            if min_x not in range(0, x) or min_y not in range(0, y) or max_x not in range(0, x) or max_y not in range(0,
                                                                                                                      y):
                continue

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


@numba.guvectorize([(numba.float32[:, :, :], numba.float32[:, :, :], numba.float32[:, :, :])],
                   "(a,b,c),(a,b,c)->(a,b,c)")
def check_id(crypto1, crypto2, output):
    x, y, z = crypto1.shape

    for i in range(0, x):
        for j in range(0, y):
            crypto1_value = crypto1[i][j]
            crypto2_value = crypto2[i][j]

            if not np.array_equal(crypto1_value, crypto2_value):
                output[i][j] = [0.0, 0.0, 0.0]


# not sure this is still necessary
def make_motion_boundary(flow):
    b, c, w, h = flow.shape
    output = torch.ones(b, c, w, h)
    laplacian = kornia.filters.laplacian(flow, 5, normalized=True)

    @numba.guvectorize([(numba.float32[:, :, :, :], numba.float32[:, :, :, :])], "(m,node,o,p),(m,node,o,p)")
    def _make_motion_mask(t, output):
        b, c, w, h = t.shape
        for i in range(0, w):
            for j in range(0, h):
                lvr1 = t[0][2][i][j]
                if not -0.1 < lvr1 < 0.1:
                    output[0][0][i][j] = 0.0
                    output[0][1][i][j] = 0.0
                    output[0][2][i][j] = 0.0

                lvr2 = t[0][1][i][j]
                if not -0.1 < lvr2 < 0.1:
                    output[0][0][i][j] = 0.0
                    output[0][1][i][j] = 0.0
                    output[0][2][i][j] = 0.0

    _make_motion_mask(laplacian, output)
    return output


def make_motion_mask(id_path, depth_path, flow_path, id_ref_path, depth=50, median_kernel=5, median_reps=2,
                     erode_kernel=3):
    """
    id_ref_path is the "to" frame, which depends on the direction of the pass,
    i.e. t-1 for a backwards pass, t+1 for a forewards pass

    depth = '/mnt/ala/research/danielf/2023/disocc/v06/depth.0015.exr'
    crypto = '/mnt/ala/research/danielf/2023/disocc/v02/cryptoRGB.0015.exr'
    crypto2 = '/mnt/ala/research/danielf/2023/disocc/v02/cryptoRGB.0014.exr'
    mfore = '/mnt/ala/research/danielf/2023/disocc/v06/motionBack.0015.exr'
    mask = make_motion_mask(crypto, depth, mfore, crypto2)
    mask_output = '/mnt/ala/research/danielf/warp/disocc/motionMask.v002.1015.exr'
    utils.np_write(mask, mask_output, ext='exr')
    """


    id_buf = oiio.ImageBuf(id_path)
    id_np = id_buf.get_pixels(roi=id_buf.roi_full)

    depth_buf = oiio.ImageBuf(depth_path)
    depth_np = depth_buf.get_pixels(roi=depth_buf.roi_full)

    flow_buf = oiio.ImageBuf(flow_path)
    flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)

    x, y, z = flow_np.shape
    warp_output = np.zeros((x, y, z), dtype=np.float32)

    # get motionm boundaries - dont think we need this anymore
    # flow_tensor = utils.buf_to_tensor(flow_buf, False, raw=True)
    # motion_boundary = make_motion_boundary(flow_tensor)[0].transpose(0,1).transpose(1, 2)
    # motion_boundary_np = motion_boundary.numpy()

    # do warp
    depth_warp(id_np, flow_np, depth_np, warp_output)

    id_ref_buf = oiio.ImageBuf(id_ref_path)
    id_ref_np = id_ref_buf.get_pixels(roi=id_ref_buf.roi_full)

    check_id_output = np.ones((x, y, z), dtype=np.float32)

    # mask by id correspondences
    check_id(warp_output, id_ref_np, check_id_output)

    for med in range(0, median_reps):
        check_id_output = cv2.medianBlur(check_id_output, median_kernel)

    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    check_id_output = cv2.erode(check_id_output, kernel)

    # return np.multiply(check_id_output, motion_boundary_np)
    return check_id_output





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
#     flow_np = flow[0].transpose(0, 2).transpose(0, 1).numpy()
#
#     if cuda:
#         result_np = torch.zeros(img_np.shape).numpy()
#         warp_image_gpu(img_np, flow_np, result_np)
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



