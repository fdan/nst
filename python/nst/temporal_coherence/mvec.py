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

from . import utils as tcutils


class BatchConverger(object):
    """
    Handles batch logic.
    """

    def __init__(self):
        self.key_dist = 5
        self.batches = 1
        self.start = 1001
        self.out_dir = ''
        self.x = 1080
        self.y = 1920
        self.z = 3
        self.p_sparsity = 1
        self.source_mvec_fore = ''
        self.source_mvec_back = ''
        self.mvc = MvecConverger()

    def run(self):
        start = self.start

        for b in range(0, self.batches):
            end = start + self.key_dist - 2
            self.mvc.start = start
            self.mvc.end = end
            self.mvc.out_dir = self.out_dir
            self.mvc.x = self.x
            self.mvc.y = self.y
            self.mvc.p_sparsity = self.p_sparsity
            self.mvc.source_mvec_fore = self.source_mvec_fore
            self.mvc.source_mvec_back = self.source_mvec_back
            self.mvc.run()


class MvecConverger(object):
    """
    Given a sequence of fore and back motion vectors, warp them into
    each other so their results converge.

    Knows nothing of batches.  Holds no computation code.
    """

    def __init__(self):
        self.start = 1003
        self.end = 1006
        self.key_dist = self.end - self.start
        self.out_dir = ''
        self.x = 1080
        self.y = 1920
        self.z = 3
        self.p_sparsity = 1
        self.source_mvec_fore = ''
        self.source_mvec_back = ''

    def run(self):
        # self.write_p_fore()
        # self.write_p_back()

        # self.average_p_fore()
        # self.average_p_back()

        # self.warp_mvec_fore()
        self.warp_mvec_back()

    def write_p_fore(self):
        """
        Write out the p fore warped position image, carrying the result each frame
        """
        frame_offset = -1
        key_counter = 1
        p_warp_fore_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'
        tcutils.make_output_dirs(p_warp_fore_filepath)

        frames = []
        for x in range(self.start, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            p_warp_fore_filepath_current_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame))
            p_warp_fore_filepath_prior_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame + frame_offset))

            # keyframe requires no warp - just write out p as-is
            if key_counter == 1:
                tcutils.generate_p(self.p_sparsity, self.x, self.y, self.z, p_warp_fore_filepath_current_frame)
                key_counter += 1
                continue

            # non-keyframe - warp the prior warped p
            mvec_fore_prior_frame = self.source_mvec_fore.replace('*', '%04d' % (frame + frame_offset))
            tcutils.warp_p(p_warp_fore_filepath_prior_frame, mvec_fore_prior_frame, p_warp_fore_filepath_current_frame)
            key_counter += 1

    def write_p_back(self):
        """
        Write out the p back warped position image, carrying the result each frame
        """
        frame_offset = 1
        key_counter = 1
        p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'
        tcutils.make_output_dirs(p_warp_back_filepath)

        frames = []
        for x in range(self.start, self.end + frame_offset):
            frames.append(x)
        frames.reverse()

        for frame in frames:
            p_warp_back_filepath_current_frame = p_warp_back_filepath.replace('*', '%04d' % (frame))
            p_warp_back_filepath_prior_frame = p_warp_back_filepath.replace('*', '%04d' % (frame + frame_offset))

            # keyframe requires no warp - just write out p as-is
            if key_counter == 1:
                tcutils.generate_p(self.p_sparsity, self.x, self.y, self.z, p_warp_back_filepath_current_frame)
                key_counter += 1
                continue

            # non-keyframe - warp the prior warped p (note "prior" is relative
            # to the direction of iteration, reversed in this case
            mvec_back_prior_frame = self.source_mvec_back.replace('*', '%04d' % (frame + frame_offset))
            tcutils.warp_p(p_warp_back_filepath_prior_frame, mvec_back_prior_frame, p_warp_back_filepath_current_frame)
            key_counter += 1

    def average_p_fore(self):
        """
        Once both p fore and back are written, compute a weighted average of p fore
        against p back
        """
        frame_offset = -1
                
        p_warp_fore_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'
        p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'

        p_warp_fore_avg_filepath = self.out_dir + '/p_warp_fore_avg/p_warp_fore_avg.*.exr'
        tcutils.make_output_dirs(p_warp_fore_avg_filepath)

        frames = []
        for x in range(self.start, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            print('frame:', frame)
            frame_counter = frames.index(frame)
            batch_length = self.end+1 - self.start
            fore_frame_weight = 1. - float(frame_counter) / float(batch_length)
            back_frame_weight = float(frame_counter) / float(batch_length)

            p_warp_fore_avg_filepath_frame = p_warp_fore_avg_filepath.replace('*', '%04d' % (frame))

            p_warp_fore_filepath_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame))
            fore_image_dict = tcutils.build_image_dict(p_warp_fore_filepath_frame)

            p_warp_back_filepath_frame = p_warp_back_filepath.replace('*', '%04d' % (frame))
            back_image_dict = tcutils.build_image_dict(p_warp_back_filepath_frame)

            tcutils.average_p(fore_image_dict, fore_frame_weight, back_image_dict, back_frame_weight, self.x, self.y,
                              self.z, p_warp_fore_avg_filepath_frame)

    def average_p_back(self):
        """
        Once both p fore and back are written, compute a weighted average of p back
        against p fore
        """
        frame_offset = 1
        
        p_warp_fore_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'
        p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'

        p_warp_back_avg_filepath = self.out_dir + '/p_warp_back_avg/p_warp_back_avg.*.exr'
        tcutils.make_output_dirs(p_warp_back_avg_filepath)

        frames = []
        for x in range(self.start, self.end + frame_offset):
            frames.append(x)

        frames.reverse()

        for frame in frames:
            frame_counter = frames.index(frame)+1
            batch_length = self.end+1 - self.start
            back_frame_weight = 1. - float(frame_counter) / float(batch_length)
            fore_frame_weight = float(frame_counter) / float(batch_length)

            p_warp_back_avg_filepath_frame = p_warp_back_avg_filepath.replace('*', '%04d' % (frame))

            p_warp_back_filepath_frame = p_warp_back_filepath.replace('*', '%04d' % (frame))
            back_image_dict = tcutils.build_image_dict(p_warp_back_filepath_frame)

            p_warp_fore_filepath_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame))
            fore_image_dict = tcutils.build_image_dict(p_warp_fore_filepath_frame)

            tcutils.average_p(back_image_dict, fore_frame_weight, fore_image_dict, back_frame_weight, self.x, self.y,
                              self.z, p_warp_back_avg_filepath_frame)

    def warp_mvec_fore(self):
        """
        Using the averaged p fore sequence, compute a warp of fore mvec, carrying the result
        each frame
        """
        frame_offset = -1
        p_warp_fore_avg_filepath = self.out_dir + '/p_warp_fore_avg/p_warp_fore_avg.*.exr'
        mvec_fore_warped_filepath = self.out_dir + '/mvec_fore_warped/mvec_fore_warped.*.exr'

        tcutils.make_output_dirs(mvec_fore_warped_filepath)

        frames = []
        for x in range(self.start, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            p_warp_fore_avg_filepath_frame = p_warp_fore_avg_filepath.replace('*', '%04d' % (frame))
            mvec_fore_warped_filepath_frame = mvec_fore_warped_filepath.replace('*', '%04d' % (frame))

            fore_image_dict = tcutils.build_image_dict(p_warp_fore_avg_filepath_frame)
            tcutils.reverse_warp_vectors(fore_image_dict, self.x, self.y, self.z, mvec_fore_warped_filepath_frame)

    def warp_mvec_back(self):
        """
        Using the averaged p back sequence, compute a warp of back mvec, carrying the result
        each frame
        """
        frame_offset = 1
        p_warp_back_avg_filepath = self.out_dir + '/p_warp_back_avg/p_warp_back_avg.*.exr'
        mvec_back_warped_filepath = self.out_dir + '/mvec_back_warped/mvec_back_warped.*.exr'

        tcutils.make_output_dirs(mvec_back_warped_filepath)

        frames = []
        for x in range(self.start, self.end + frame_offset):
            frames.append(x)

        frames.reverse()

        for frame in frames:
            p_warp_back_avg_filepath_frame = p_warp_back_avg_filepath.replace('*', '%04d' % (frame))
            mvec_back_warped_filepath_frame = mvec_back_warped_filepath.replace('*', '%04d' % (frame))

            fore_image_dict = tcutils.build_image_dict(p_warp_back_avg_filepath_frame)
            tcutils.reverse_warp_vectors(fore_image_dict, self.x, self.y, self.z, mvec_back_warped_filepath_frame)




