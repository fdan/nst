import os
import shutil

from . import utils as tcutils

class BatchSmoother(object):
    """
    Handles batch logic.
    """

    def __init__(self):
        self.comp_background = True
        self.key_dist = 6
        self.batches = 1
        self.start = 1007
        self.out_dir = ''
        self.x = 1080
        self.y = 1920
        self.z = 3
        self.brush = ''
        self.p_sparsity = 1
        self.source_mvec_fore = ''
        self.source_mvec_back = ''
        self.object_id = ''
        self.smoothed_output = ''
        self.nst_render = ''
        self.skip_existing = True

    def run(self):
        num_frames = self.batches * self.key_dist
        keys = [x for x in range(self.start, num_frames+self.start)][::self.key_dist+1]

        for key in keys:
            self.smooth_key(key)


    def smooth_key(self, key):
        # must be divisible by two
        if self.key_dist % 2:
            raise RuntimeError('key_dist must be an integer divisible by 2')

        stride = self.key_dist / 2

        output = self.out_dir + '/smoothed/smoothed.*.exr'
        tcutils.make_output_dirs(output)

        # copy key frame
        nst_render_key_frame = self.nst_render.replace('*', '%04d' % (key))
        output_key_frame = output.replace('*', '%04d' % (key))

        if self.skip_existing:
            if os.path.isfile(output_key_frame):
                print('frame exists, skipping', key)
            else:
                print('copying:', nst_render_key_frame, output_key_frame)
                shutil.copy(nst_render_key_frame, output_key_frame)
        else:
            print('copying:', nst_render_key_frame, output_key_frame)
            shutil.copy(nst_render_key_frame, output_key_frame)

        # fore warp
        for frame in range(key+1, key+int(stride)+1):
            print(frame)
            frame_offset = -1
            object_id_prior_frame = self.object_id.replace('*', '%04d' % (frame + frame_offset))
            object_id_current_frame = self.object_id.replace('*', '%04d' % (frame))
            mvec_fore_prior_frame = self.source_mvec_fore.replace('*', '%04d' % (frame + frame_offset))
            fore_output_prior_frame = output.replace('*', '%04d' % (frame + frame_offset))
            output_current_frame = output.replace('*', '%04d' % (frame))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            if self.skip_existing and os.path.isfile(output_current_frame):
                print('frame exists, skipping:', frame, output_current_frame)
                continue

            print('writing', frame, output_current_frame)

            tcutils.p_warp_frame(fore_output_prior_frame,
                                 mvec_fore_prior_frame,
                                 output_current_frame,
                                 object_id_current_frame,
                                 object_id_prior_frame,
                                 nst_render_current_frame,
                                 self.comp_background,
                                 self.brush)


        # back warp
        back_frames = list(range(key-int(stride), key))
        back_frames.reverse()

        for frame in back_frames:
            print(frame)
            frame_offset = 1
            back_output_frame = output.replace('*', '%04d' % (frame))
            object_id_current_frame = self.object_id.replace('*', '%04d' % (frame))
            object_id_prior_frame = self.object_id.replace('*', '%04d' % (frame + frame_offset))
            back_output_prior_frame = output.replace('*', '%04d' % (frame + frame_offset))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            mvec_back_prior_frame = self.source_mvec_back.replace('*', '%04d' % (frame + frame_offset))

            if self.skip_existing and os.path.isfile(back_output_frame):
                print('frame exists, skipping:', frame)
                continue

            print('writing', frame)

            tcutils.p_warp_frame(back_output_prior_frame,
                                 mvec_back_prior_frame,
                                 back_output_frame,
                                 object_id_current_frame,
                                 object_id_prior_frame,
                                 nst_render_current_frame,
                                 self.comp_background,
                                 self.brush)







