import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import shutil
import os
import cv2
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI
import numpy as np
from operator import itemgetter
from PIL import Image


class FrameSmoother(object):

    def __init__(self):
        self.key_dist = 5
        self.batches = 47
        self.start = 1003
        self.out_dir = ''
        self.x = 1080
        self.y = 1920
        self.z = 3
        self.p_sparsity = 1
        self.source_mvec_fore = ''
        self.source_mvec_back = ''

    def blend_mvec(self):
        """
        create a new blended fore and back mvec sequence
        """
        start = self.start

        # process batches
        for b in range(0, self.batches):
            end = start + self.key_dist - 2

            # create the data we need before we start warping the nst render
            self._write_warp_p_batch([start, end])
            self._write_mvec_batch([start, end])

            

            start += self.key_dist -1

    def _warp_p(self, position, vec):
        vec_buf = oiio.ImageBuf(vec)
        vec_np = vec_buf.get_pixels(roi=vec_buf.roi_full)
        output = np.zeros((self.x, self.y, self.z))

        for old_x in range(0, self.x):
            for old_y in range(0, self.y):
                p_img = position[old_x][old_y]

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
        return output

    def _make_output_dirs(self, output):
        t_ = output.split('/')
        t_.pop()
        d_ = ('/'.join(t_))
        try:
            os.makedirs(d_)
        except:
            pass

    def _write_warp_p_batch(self, frames):
        key_counter = 1
        frame_offset = -1
        vec_warp_p = self.out_dir + '/p_warp/p_warp.*.exr'
        self._make_output_dirs(vec_warp_p)

        frames_ = []
        for x in range(frames[0], frames[1] + 1):
            frames_.append(x)

        p = self._generate_p()

        for x in frames_:
            print('\nframe:', x)

            fore_vec_prior_frame = self.source_mvec_fore.replace('*', '%04d' % (x + frame_offset))
            back_vec_next_frame = self.source_mvec_back.replace('*', '%04d' % (x - frame_offset))

            vec_warp_p_current_frame = vec_warp_p.replace('*', '%04d' % (x))

            # hardcore for now...
            if key_counter == 3:
                print('warping', x, key_counter)

                # do fore warp:
                fore_p = self._warp_p(p, fore_vec_prior_frame)
                fore_dict = self._build_image_dict(fore_p)

                # do back warp
                back_p = self._warp_p(p, back_vec_next_frame)
                back_dict = self._build_image_dict(back_p)

                self._average_p(fore_dict, back_dict, vec_warp_p_current_frame)
                key_counter += 1

            else:
                key_counter += 1

    def _write_mvec_batch(self, frames):
        key_counter = 1
        fore_output_vec = self.out_dir + '/mvec/fore.*.exr'
        back_output_vec = self.out_dir + '/mvec/back.*.exr'
        p_warp = self.out_dir + '/p_warp/p_warp.*.exr'
        self._make_output_dirs(fore_output_vec)

        for x in range(frames[0], frames[1]+1):
            print('\nframe:', x)

            if key_counter == 1:
                print(x, 'is a key, not warping', key_counter)
                mvec_current_frame = self.source_mvec_fore.replace('*', '%04d' % (x))
                fore_output_vec_current_frame = fore_output_vec.replace('*', '%04d' % (x))
                shutil.copy(mvec_current_frame, fore_output_vec_current_frame)
                key_counter += 1

            elif key_counter == 2:
                print('warping', x, key_counter)
                fore_output_vec_current_frame = fore_output_vec.replace('*', '%04d' % (x))
                p_warp_frame = p_warp.replace('*', '%04d' % (x + 1))
                print(2, p_warp_frame)
                p_warp_buf = ImageBuf(p_warp_frame)
                p_warp_np = p_warp_buf.get_pixels(roi=p_warp_buf.roi_full)
                fore_dict = self._build_image_dict(p_warp_np)
                self._reverse_warp_vectors(p_warp_np, fore_dict, fore_output_vec_current_frame)
                key_counter += 1

            elif key_counter == 4:
                print('warping', x, key_counter)
                back_output_vec_current_frame = back_output_vec.replace('*', '%04d' % (x))
                p_warp_frame = p_warp.replace('*', '%04d' % (x - 1))
                p_warp_buf = ImageBuf(p_warp_frame)
                p_warp_np = p_warp_buf.get_pixels(roi=p_warp_buf.roi_full)
                back_dict = self._build_image_dict(p_warp_np)
                self._reverse_warp_vectors(p_warp_np, back_dict, back_output_vec_current_frame)

                # copy back movec for next / last frame
                mvec_current_frame = self.source_mvec_fore.replace('*', '%04d' % (x+1))
                back_output_vec_current_frame = back_output_vec.replace('*', '%04d' % (x+1))
                shutil.copy(mvec_current_frame, back_output_vec_current_frame)

                key_counter += 1

            else:
                key_counter += 1

    def _average_p(self, fore_dict, back_dict, out_path):
        # construct a new averaged image.
        img_avg = np.zeros((self.x, self.y, self.z))
        for x in range(0, self.x):
            for y in range(0, self.y):
                key = '%s,%s,%s' % (x, y, 0)
                try:
                    fore_loc = np.array(fore_dict[key])
                    back_loc = np.array(back_dict[key])
                except KeyError:
                    continue
                avg_loc = np.mean(np.array([fore_loc, back_loc]), axis=0)
                try:
                    img_avg[int(avg_loc[0])][int(avg_loc[1])] = [x, y, 0]
                except:
                    print(avg_loc)
        self._np_write(img_avg, out_path)

    def _generate_p(self):
        img_sparse = np.zeros((self.x, self.y, self.z))
        for x in range(0, self.x):
            if x % self.p_sparsity:
                continue
            for y in range(0, self.y):
                if y % self.p_sparsity:
                    continue
                g_val = int(y)
                r_val = int(x)
                img_sparse[x][y] = [r_val, g_val, 0.0]
        return img_sparse
        # np_write(img_sparse, '/mnt/ala/tmp/position_sparse_20.exr')

    def _build_image_dict(self, img):
        fore_dict = {}
        for x_ in range(0, self.x):
            for y_ in range(0, self.y):
                value = img[x_][y_]
                if not value.any():
                    continue
                key = '%s,%s,%s' % (int(value[0]), int(value[1]), 0)
                fore_dict[key] = [x_, y_]
        return fore_dict

    def _reverse_warp_vectors(self, position, img_dict, out_path):
        # reverse warp fore vectors
        rev_warp = np.zeros((self.x, self.y, self.z))

        for x in range(0, self.x):
            for y in range(0, self.y):
                if not position[x][y].any():
                    continue
                key = '%s,%s,%s' % (x, y, 0)
                try:
                    p_loc = np.array([float(b) for b in key.split(',')])
                    fore_loc = np.array( [float(a) for a in img_dict[key] + [0]] )
                except KeyError:
                    continue

                vec = p_loc - fore_loc
                rev_warp[x][y] = np.array([0.-vec[1], vec[0], 0])

        self._np_write(rev_warp, out_path)

    def _np_write(self, img, path):
        buf = ImageBuf(ImageSpec(self.y, self.x, self.z, oiio.FLOAT))
        buf.set_pixels(ROI(), img)
        buf.write(path)

