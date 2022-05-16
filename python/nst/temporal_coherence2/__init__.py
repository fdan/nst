import shutil

from . import utils as tcutils


class BatchSmoother(object):
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
        self.brush = ''
        self.p_sparsity = 1
        self.source_mvec_fore = ''
        self.source_mvec_back = ''
        self.object_id = ''
        self.mvc = MvecConverger()
        self.fs = FrameSmoother()
        self.smoothed_output = ''
        self.nst_render = ''

    def run(self):
        batch_counter = 0

        for b in range(0, self.batches):
            start = self.start + (batch_counter * self.key_dist) - batch_counter
            end = start + self.key_dist - 2

            # self.mvc.start = start
            # self.mvc.end = end
            # self.mvc.out_dir = self.out_dir
            # self.mvc.x = self.x
            # self.mvc.y = self.y
            # self.mvc.p_sparsity = self.p_sparsity
            # self.mvc.source_mvec_fore = self.source_mvec_fore
            # self.mvc.source_mvec_back = self.source_mvec_back
            # self.mvc.run()

            self.fs.start = start
            self.fs.end = end
            self.fs.out_dir = self.out_dir
            self.fs.x = self.x
            self.fs.y = self.y
            self.fs.brush = self.brush
            self.fs.p_warp_fore_avg_filepath = self.mvc.p_warp_fore_avg_filepath
            self.fs.p_warp_back_avg_filepath = self.mvc.p_warp_back_avg_filepath
            self.fs.source_mvec_fore = self.source_mvec_fore
            self.fs.source_mvec_back = self.source_mvec_back
            self.fs.output = self.smoothed_output
            self.fs.nst_render = self.nst_render
            self.fs.object_id = self.object_id
            self.fs.run()

            batch_counter += 1


class FrameSmoother(object):

    def __init__(self):
        self.start = 1003
        self.end = 1006
        self.key_dist = self.end - self.start
        self.out_dir = ''
        self.x = 1080
        self.y = 1920
        self.z = 3
        self.brush = ''
        self.comp_background = False
        self.nst_render = ''
        self.object_id = ''
        self.source_mvec_fore = ''
        self.source_mvec_back = ''

    def run(self):
        self.smooth()
        # self.smooth_fore()
        # self.smooth_back()
        # self.merge()


    def smooth(self):
        print('smooth')
        key_counter = 1
        frame_offset = -1

        output = self.out_dir + '/smoothed/smoothed.*.exr'
        tcutils.make_output_dirs(output)

        frames = []
        for x in range(self.start, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            print('frame', frame)
            next_key = self.end + 1
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))
            nst_render_next_key = self.nst_render.replace('*', '%04d' % (self.end+1))

            output_current_frame = output.replace('*', '%04d' % (frame))

            object_id_current_frame = self.object_id.replace('*', '%04d' % (frame))
            object_id_prior_frame = self.object_id.replace('*', '%04d' % (frame + frame_offset))
            object_id_next_key_frame = self.object_id.replace('*', '%04d' % (next_key))

            frame_counter = frames.index(frame)
            batch_length = self.end + 1 - self.start
            fore_frame_weight = 1. - float(frame_counter) / float(batch_length)
            back_frame_weight = float(frame_counter) / float(batch_length)

            # keyframe is a special case - just copy it over to the output
            if key_counter == 1:
                print('copying:', nst_render_current_frame, output_current_frame)
                shutil.copy(nst_render_current_frame, output_current_frame)
                key_counter += 1
                continue

            # for back pass, need to get cumulated sum of mvecs from next key
            back_mvecs = []

            for i in range(frame+1, next_key+1):
                back_mvecs.append(self.source_mvec_back.replace('*', '%04d' % i))

            # this should be safe as they first batch frame aka keyframe is always copied
            output_fore_prior_frame = output.replace('*', '%04d' % (frame + frame_offset))
            back_prior_frame = self.nst_render.replace('*', '%04d' % next_key)
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))
            mvec_fore_prior_frame = self.source_mvec_fore.replace('*', '%04d' % (frame + frame_offset))

            tcutils.warp_frame(output_fore_prior_frame, mvec_fore_prior_frame, output_current_frame,
                                 object_id_current_frame, object_id_prior_frame, object_id_next_key_frame,
                               nst_render_current_frame, nst_render_next_key, back_mvecs, back_prior_frame,
                               fore_frame_weight, back_frame_weight)

            key_counter += 1


    def smooth_fore(self):
        print('smooth fore')
        key_counter = 1
        frame_offset = -1

        # are we substituting the prior smoothed frame as in put?  I think we aren't?
        fore_output = self.out_dir + '/fore_smoothed/fore_smoothed.*.exr'
        tcutils.make_output_dirs(fore_output)
        
        p_warp_fore_avg_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'

        frames = []
        for x in range(self.start, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            p_warp_fore_avg_frame = p_warp_fore_avg_filepath.replace('*', '%04d' % (frame))
            fore_output_frame = fore_output.replace('*', '%04d' % (frame))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            object_id_current_frame = self.object_id.replace('*', '%04d' % (frame))
            object_id_prior_frame = self.object_id.replace('*', '%04d' % (frame + frame_offset))

            # keyframe is a special case - just copy it over to the output
            if key_counter == 1:
                print('copying:', nst_render_current_frame, fore_output_frame)
                shutil.copy(nst_render_current_frame, fore_output_frame)
                key_counter += 1
                continue

            # this should be safe as they first batch frame aka keyframe is always copied
            fore_output_prior_frame = fore_output.replace('*', '%04d' % (frame + frame_offset))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            # warp_fore_dict = tcutils.build_image_dict(p_warp_fore_avg_frame)
            # tcutils.p_warp_frame(fore_output_prior_frame, warp_fore_dict, fore_output_frame, object_id_current_frame, object_id_prior_frame, nst_render_current_frame)

            mvec_fore_prior_frame = self.source_mvec_fore.replace('*', '%04d' % (frame + frame_offset))
            tcutils.p_warp_frame(fore_output_prior_frame, mvec_fore_prior_frame, fore_output_frame,
                                 object_id_current_frame, object_id_prior_frame, nst_render_current_frame,
                                 self.brush)

            key_counter += 1

    def smooth_back(self):
        print('smooth back')
        key_counter = 1
        frame_offset = 1

        back_output = self.out_dir + '/back_smoothed/back_smoothed.*.exr'
        tcutils.make_output_dirs(back_output)

        p_warp_back_avg_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'

        frames = []
        for x in range(self.start + frame_offset, self.end + frame_offset + 1): # offset offset
            frames.append(x)

        frames.reverse()

        for frame in frames:
            p_warp_back_avg_frame = p_warp_back_avg_filepath.replace('*', '%04d' % (frame))
            back_output_frame = back_output.replace('*', '%04d' % (frame))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            object_id_current_frame = self.object_id.replace('*', '%04d' % (frame))
            object_id_prior_frame = self.object_id.replace('*', '%04d' % (frame + frame_offset))

            # in the backwards pass, key_counter == 1 is not a keyframe, it's prior is
            if key_counter == 1:
                print('copying:', nst_render_current_frame, back_output_frame)
                shutil.copy(nst_render_current_frame, back_output_frame)
                key_counter += 1
                continue

            back_output_prior_frame = back_output.replace('*', '%04d' % (frame + frame_offset))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))

            # warp_back_dict = tcutils.build_image_dict(p_warp_back_avg_frame)
            # tcutils.p_warp_frame(back_output_prior_frame, warp_back_dict, back_output_frame, object_id_current_frame, object_id_prior_frame, nst_render_current_frame)

            mvec_back_prior_frame = self.source_mvec_back.replace('*', '%04d' % (frame + frame_offset))
            tcutils.p_warp_frame(back_output_prior_frame, mvec_back_prior_frame, back_output_frame, object_id_current_frame, object_id_prior_frame, nst_render_current_frame)

            key_counter += 1

    def merge(self):
        print("merging back and fore outputs")
        frame_offset = -1

        back_output = self.out_dir + '/back_smoothed/back_smoothed.*.exr'
        fore_output = self.out_dir + '/fore_smoothed/fore_smoothed.*.exr'
        merge_output = self.out_dir + '/merge/merge.*.exr'

        tcutils.make_output_dirs(merge_output)

        # note: fore and back warps are offset, so drop the start:
        frames = []
        for x in range(self.start - frame_offset, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            print('frame', frame)
            frame_counter = frames.index(frame)
            batch_length = self.end + 1 - self.start
            fore_frame_weight = 1. - float(frame_counter) / float(batch_length)
            back_frame_weight = float(frame_counter) / float(batch_length)
            merge_output_frame = merge_output.replace('*', '%04d' % (frame))
            fore_output_frame = fore_output.replace('*', '%04d' % (frame))
            back_output_frame = back_output.replace('*', '%04d' % (frame))
            nst_render_current_frame = self.nst_render.replace('*', '%04d' % (frame))
            tcutils.merge(fore_output_frame, fore_frame_weight, back_output_frame, back_frame_weight, nst_render_current_frame, merge_output_frame)



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
        self.p_warp_fore_avg_filepath = ''
        self.p_warp_back_avg_filepath = ''

    def run(self):
        self.write_p_fore()
        self.write_p_back()

    def write_p_fore(self):
        """
        Write out the p fore warped position image, carrying the result each frame
        """
        print("write p fore")
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
        print("write p back")
        frame_offset = 1
        key_counter = 1
        p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'
        tcutils.make_output_dirs(p_warp_back_filepath)

        frames = []
        for x in range(self.start + frame_offset, self.end + frame_offset + 1):
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
        print("average p fore")
        frame_offset = -1

        p_warp_fore_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'
        p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'

        self.p_warp_fore_avg_filepath = self.out_dir + '/p_warp_fore_avg/p_warp_fore_avg.*.exr'
        tcutils.make_output_dirs(self.p_warp_fore_avg_filepath)

        # note: fore and back warps are offset, so drop the start:
        frames = []
        for x in range(self.start - frame_offset, self.end - frame_offset):
            frames.append(x)

        for frame in frames:
            print('frame', frame)
            frame_counter = frames.index(frame)
            batch_length = self.end + 1 - self.start
            fore_frame_weight = 1. - float(frame_counter) / float(batch_length)
            back_frame_weight = float(frame_counter) / float(batch_length)

            p_warp_fore_avg_filepath_frame = self.p_warp_fore_avg_filepath.replace('*', '%04d' % (frame))

            p_warp_fore_filepath_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame))
            fore_image_dict = tcutils.build_image_dict(p_warp_fore_filepath_frame)

            p_warp_back_filepath_frame = p_warp_back_filepath.replace('*', '%04d' % (frame))
            back_image_dict = tcutils.build_image_dict(p_warp_back_filepath_frame)
            tcutils.average_p(fore_image_dict, fore_frame_weight, back_image_dict, back_frame_weight, self.x, self.y, self.z, p_warp_fore_avg_filepath_frame)

    # def average_p_back(self):
    #     """
    #     Once both p fore and back are written, compute a weighted average of p back
    #     against p fore
    #     """
    #     frame_offset = 1
    #
    #     p_warp_fore_filepath = self.out_dir + '/p_warp_fore/p_warp_fore.*.exr'
    #     p_warp_back_filepath = self.out_dir + '/p_warp_back/p_warp_back.*.exr'
    #
    #     self.p_warp_back_avg_filepath = self.out_dir + '/p_warp_back_avg/p_warp_back_avg.*.exr'
    #     tcutils.make_output_dirs(self.p_warp_back_avg_filepath)
    #
    #     frames = []
    #     for x in range(self.start + frame_offset, self.end + frame_offset + 1):
    #         frames.append(x)
    #
    #     frames.reverse()
    #
    #     for frame in frames:
    #         print('frame', frame)
    #         frame_counter = frames.index(frame) + 1
    #         batch_length = self.end + 1 - self.start
    #         back_frame_weight = 1. - float(frame_counter) / float(batch_length)
    #         fore_frame_weight = float(frame_counter) / float(batch_length)
    #
    #         p_warp_back_avg_filepath_frame = self.p_warp_back_avg_filepath.replace('*', '%04d' % (frame))
    #
    #         p_warp_back_filepath_frame = p_warp_back_filepath.replace('*', '%04d' % (frame))
    #         back_image_dict = tcutils.build_image_dict(p_warp_back_filepath_frame)
    #
    #         p_warp_fore_filepath_frame = p_warp_fore_filepath.replace('*', '%04d' % (frame))
    #         fore_image_dict = tcutils.build_image_dict(p_warp_fore_filepath_frame)
    #
    #         tcutils.average_p(back_image_dict, fore_frame_weight, fore_image_dict, back_frame_weight, self.x, self.y,
    #                           self.z, p_warp_back_avg_filepath_frame)
    #
