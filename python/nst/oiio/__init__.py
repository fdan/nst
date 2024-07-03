import skimage.color as sc
import skimage.filters as sf
import skimage.morphology as sm

import re
import shutil
from typing import List
import os

import OpenImageIO as oiio
import numpy as np
from torch.autograd import Variable
import torch

from . import utils
from . import temporal_coherence

from nst.core import model, loss
from nst.core import utils as core_utils
import nst.settings as settings


FRAME_REGEX = r"\.([0-9]{4}|[#]{4})\."


class StyleWriter(object):

    def __init__(self,
                 styles: List[settings.StyleImage]=None,
                 opt_image: settings.Image=None,
                 content: settings.Image=None
                 # temporal_content: settings.Image = None,
                 ):

        self.settings = settings.WriterSettings()
        self.settings.styles = styles
        self.settings.opt_image = opt_image
        self.settings.content = content
        # self.settings.temporal_content = temporal_content
        self.output_dir = ''
        self._prepared = False

        # # pretty sure this can live in the constructor?
        # if self.settings.core.engine == "gpu":
        #     self.settings.core.cuda = True
        # elif self.settings.core.engine == "cpu":
        #     self.settings.core.cuda = False
        #
        # self.nst = model.Nst()
        # self.nst.settings = self.settings.core

    def load(self, fp):
        self.settings.load(fp)

    def prepare_model(self):
        # pretty sure this is unecessary
        # self.vgg, self.settings.core.cuda = core_utils.get_vgg(self.settings.core.engine,
        #                                                        self.settings.core.model_path)
        if self._prepared:
            return

        if self.settings.core.engine == "gpu":
            self.settings.core.cuda = True
        elif self.settings.core.engine == "cpu":
            self.settings.core.cuda = False

        self.nst = model.Nst()
        self.nst.settings = self.settings.core

        self._prepared = True

    def get_output_dir(self):
        if not self.output_dir:
            self.output_dir = os.path.abspath(os.path.join(self.settings.out, (os.path.pardir)))
        return self.output_dir

    def write(self) -> None:

        frame = self.settings.frame
        if frame:
            if self.settings.content:
                if self.settings.content.rgb_filepath:
                    self.settings.content.rgb_filepath = re.sub(FRAME_REGEX, ".%s." % frame, self.settings.content.rgb_filepath)

            if self.settings.opt_image:
                if self.settings.opt_image.rgb_filepath:
                    self.settings.opt_image.rgb_filepath = re.sub(FRAME_REGEX, ".%s." % frame, self.settings.opt_image.rgb_filepath)

            if self.settings.out:
                self.settings.out = re.sub(FRAME_REGEX, ".%s." % frame, self.settings.out)

        out_dir = os.path.abspath(os.path.join(self.settings.out, os.path.pardir))
        os.makedirs(out_dir, exist_ok=True)

        self.prepare_model()

        if self.settings.content:
            self.nst.content = self.prepare_content()

        # if self.settings.temporal_content:
        #     self.nst.temporal_content = self.prepare_temporal_content()

        if self.settings.temporal_mask:
            print('temporal mask provided')
            self.nst.temporal_weight_mask = self.prepare_temporal_mask()
        else:
            print('no temporal mask')

        if self.nst.opt_tensor.numel() == 0:
            self.nst.opt_tensor = self.prepare_opt()

        self.nst.styles = self.prepare_styles()
        self.nst.prepare()
        assert self.settings.core == self.nst.settings

        # set num cpus
        torch.set_num_threads = int(self.settings.core.cpu_threads)

        # do style transfer
        tensor = self.nst()

        # print('out format:', self.settings.output_format)

        if self.settings.output_format == 'exr':
            # print('writing exr %s' % self.settings.out)
            buf = utils.tensor_to_buf(tensor, colorspace=self.settings.out_colorspace)
            # buf = utils.tensor_to_buf(tensor)
            # if self.settings.out_colorspace != 'srgb_texture':
            #     buf = oiio.ImageBufAlgo.colorconvert(buf, 'srgb_texture', self.settings.out_colorspace)

            # out = self.settings.out.split('/')
            # out[-1] = '%03d_' % self.pass_ + out[-1]
            # out_pass = '/'.join(out)
            # print('writing:', out_pass)
            # buf.write(out_pass)
            buf.write(self.settings.out)

        # elif self.settings.output_format == 'pt':
        #     out = self.settings.out.replace('.exr', '.pt')
        #     print('writing pt %s \n' % out)
        #     torch.save(tensor, out)

        else:
            print('not writing anything')

    def prepare_styles(self):
        styles = []

        for style in self.settings.styles:

            style_tensor, style_alpha = utils.style_image_to_tensors(style.rgba_filepath,
                                                                          self.settings.core.cuda,
                                                                          colorspace=style.colorspace)

            s = model.TorchStyle(style_tensor)
            s.alpha = style_alpha

            if style.target_map_filepath:
                s.target_map = utils.image_to_tensor(style.target_map_filepath,
                                                     self.settings.core.cuda,
                                                     raw=True)
            styles.append(s)

        return styles

    def prepare_content(self):
        content_tensor = utils.image_to_tensor(self.settings.content.rgb_filepath, self.settings.core.cuda,
                                               colorspace=self.settings.content.colorspace)
        return content_tensor

    def prepare_temporal_mask(self):
        mask_tensor = utils.image_to_tensor(self.settings.temporal_mask,
                                            self.settings.core.cuda,
                                            raw=True)

        return mask_tensor

    # def prepare_temporal_content(self):
    #
    #     tc_tensor, tc_alpha = utils.style_image_to_tensors(self.settings.temporal_content.rgba_filepath,
    #                                                              self.settings.core.cuda,
    #                                                              colorspace=self.settings.temporal_content.colorspace)
    #
    #     t = model.TorchMaskedImage(tc_tensor)
    #     t.alpha = tc_alpha
    #
    #     return t

    def prepare_opt(self):
        # print(3.0)
        # if self.settings.core.temporal_weight:
        #     this_frame = int(self.settings.frame) + int(self.settings.pre)
        #     re.sub(FRAME_REGEX, ".%s." % this_frame)
        #     pass
        #     # 1.  check if prev frame output exists
        opt_filepath = self.settings.opt_image.rgb_filepath

        # if direction is backwards, use the forewards disocclusion mask

        print('preparing opt image:', opt_filepath)

        if opt_filepath.endswith('.exr'):
            opt_tensor = utils.image_to_tensor(opt_filepath, self.settings.core.cuda,
                                           colorspace=self.settings.opt_image.colorspace)

        # elif opt_filepath.endswith('.pt'):
        #     opt_tensor = torch.load(opt_filepath)

        opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
        return opt_tensor

    def write_style_gram(self, ext='jpg'):
        s_index = 0
        for s in self.settings.styles:

            if s.alpha_filepath:
                style_in_mask_tensor = utils.image_to_tensor(s.alpha_filepath, self.settings.core.cuda, raw=True)
            else:
                style_in_mask_tensor = None

            style_tensor = utils.image_to_tensor(s.rgb_filepath, self.settings.core.cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.settings.core.cuda,
                                                                mips=self.settings.core.style_mips,
                                                                pyramid_scale_factor=self.settings.core.pyramid_scale_factor)
            style_activations = []
            style_layer_names = [x.name for x in self.settings.core.style_layers]
            for layer_activation_pyramid in self.vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            vgg_layer_index = 0
            for vgg_layer in style_activations:
                gram_pyramid = []
                mip_index = 0
                for mip_activations in vgg_layer:
                    layer_name = self.settings.core.style_layers[vgg_layer_index].name
                    gram = loss.GramMatrix()(mip_activations).detach()
                    if self.settings.write_gram:
                        fp = '%s/gram/style_%s/layer_%s/mip_%s.%s' % ( self.get_output_dir(), s_index, layer_name,
                                                                       mip_index, ext)
                        utils.write_gram(gram, fp)
                    gram_pyramid += [gram]
                    mip_index += 1
                vgg_layer_index += 1

            s_index += 1

    def write_style_pyramids(self, ext='jpg'):
        s_index = 0
        for s in self.settings.styles:
            style_tensor = utils.image_to_tensor(s.rgb_filepath, self.settings.core.cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.settings.core.cuda,
                                                                mips=self.settings.core.style_mips,
                                                                pyramid_scale_factor=self.settings.core.pyramid_scale_factor)
            outdir = '%s/style_%02d/pyramid' % (self.get_output_dir(), s_index)
            utils.Pyramid.write_gaussian_pyramid(style_pyramid, outdir, ext=ext)
            s_index += 1

    def write_style_activations(self, layer_limit=10, ext='exr'):
        s_index = 0
        for s in self.settings.styles:

            if s.alpha_filepath:
                style_in_mask_tensor = utils.image_to_tensor(s.alpha_filepath, self.settings.core.cuda, raw=True)
            else:
                style_in_mask_tensor = None

            style_tensor = utils.image_to_tensor(s.rgb_filepath, self.settings.core.cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.settings.core.cuda,
                                                                mips=self.settings.core.style_mips,
                                                                pyramid_scale_factor=self.settings.core.pyramid_scale_factor)
            style_activations = []
            style_layer_names = [x.name for x in self.settings.core.style_layers]
            for layer_activation_pyramid in self.vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            vgg_layer_index = 0
            for vgg_layer in style_activations:
                layer_name = self.settings.core.style_layers[vgg_layer_index].name
                mip_index = 0
                for mip_activations in vgg_layer:
                    _, c, w, h = mip_activations.size()
                    outdir = '%s/style_%02d/activations/layer_%s/mip_%s' % (self.get_output_dir(), s_index, layer_name, mip_index)
                    utils.write_activations(mip_activations, outdir, layer_limit=layer_limit, ext=ext)
                    mip_index += 1
                vgg_layer_index += 1

            s_index += 1

    def write_content_activations(self, layer_limit=10, ext='exr'):

        if self.settings.content:
            content_tensor = self.prepare_content()
            content_layers = [self.settings.core.content_layer]
            content_activations = []
            content_layer_names = [x.name for x in content_layers]
            for layer_activation_pyramid in self.vgg([content_tensor], content_layer_names):
                content_activations.append(layer_activation_pyramid)

            if self.write_content_activations:
                vgg_layer_index = 0
                for vgg_layer in content_activations:
                    layer_name = content_layers[vgg_layer_index]
                    mip_index = 0
                    for mip_activations in vgg_layer:
                        _, c, w, h = mip_activations.size()
                        outdir = '%s/activations/content/layer_%s/mip_%s' % (
                        self.get_output_dir(), layer_name, mip_index)
                        utils.write_activations(mip_activations, outdir, layer_limit=layer_limit, ext=ext)
                        mip_index += 1
                    vgg_layer_index += 1


class AnimWriter(StyleWriter):

    def __init__(self,
                 styles: List[settings.StyleImage] = None,
                 opt_image: settings.Image = None,
                 content: settings.Image = None):
        super(AnimWriter, self).__init__(styles, opt_image, content)

        self.settings = settings.AnimSettings()
        self.settings.output_format = 'exr'
        self.out_fp = ''
        self.this_frame = 0
        self.direction = 1
        self.pass_ = 1
        self.outdir = ''

    # def write(self) -> None:
    #     self.prepare_model()
    #     super(AnimWriter, self).write()

    def run(self):
        if self.settings.interleaved:
            self._run_interleaved()
        else:
            self._run_sequential()

    def _run_sequential(self):
        """
        Process one frame at a time to full optimisation, from start to finish
        """
        print('running sequential')
        for this_frame in range(self.settings.first_frame, self.settings.last_frame):
            print('frame:', this_frame)
            self.settings.frame = this_frame
            with torch.no_grad():
                self.write()
                self.nst = None
                self._prepared = False
            torch.cuda.empty_cache()

    def _run_interleaved(self):
        """
        Interleave results across the sequence, slowly optimise each frame and propagate results
        """
        # self.settings.output_format = 'pt'
        self.outdir = os.path.abspath(os.path.join(self.settings.out, os.path.pardir))

        # clean any previous output
        if not self.settings.skip_frames and not self.settings.skip_passes:
            if os.path.isdir(self.outdir):
                shutil.rmtree(self.outdir)
            os.makedirs(self.outdir)

        # todo: how to handle when there's a remainder?
        # print('iterations per pass:', self.settings.iterations_per_pass)
        passes = int(self.settings.total_iterations / self.settings.iterations_per_pass)
        # self.settings.core.iterations = int(self.settings.core.iterations / passes)
        # for pass_ in range(self.settings.starting_pass, self.settings.passes+1):
        for pass_ in range(self.settings.starting_pass, passes+1):

            if pass_ in self.settings.skip_passes:
                continue

            print('pass: %s' % pass_)
            # print('iterations per pass:', self.settings.iterations_per_pass)

            self.pass_ = pass_

            if pass_ == self.settings.starting_pass:
                self.settings.core.iterations = self.settings.first_pass_iterations
            else:
                self.settings.core.iterations = int(self.settings.total_iterations / passes)

            direction = pass_ % 2

            start = self.settings.first_frame if direction else self.settings.last_frame
            end = self.settings.last_frame+1 if direction else self.settings.first_frame-1
            increment_by = 1 if direction else -1

            for this_frame in range(start, end, increment_by):

                if this_frame in self.settings.skip_frames:
                    continue

                print('\nframe: %s' % (this_frame))
                # print('iterations per pass:', self.settings.iterations_per_pass)
                self.settings.frame = this_frame
                self.this_frame = this_frame
                self.direction = direction

                # prevent accumulation of graph on gpu
                # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
                with torch.no_grad():
                    self.write()
                    self.nst = None
                    self._prepared = False
                torch.cuda.empty_cache()
                # self.nst.opt_tensor = torch.zeros(0)

        # print('converting output to exrs')
        # utils.pts_to_exrs(self.outdir)
        # print('finished')

    def prepare_opt(self):

        if not self.settings.interleaved:
            return super(AnimWriter, self).prepare_opt()

        # backwards->forwards pass
        if self.direction == 1:
            prev_frame = self.this_frame - 1
            flow = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_back)
            flow_weight = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_fore_weight)

        # forwards->backwards pass
        elif self.direction == 0:
            prev_frame = self.this_frame + 1
            flow = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_fore)
            flow_weight = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_back_weight)

        else:
            raise Exception('Direction must be 0 or 1')

        # curr_frame_checkpoint_fp = self.settings.out
        # don't use the output, for better debugging use the pass output
        out = self.settings.out.split('/')
        out[-1] = '%03d_' % (self.pass_-1) + out[-1]
        curr_frame_checkpoint_fp = '/'.join(out)

        if self.pass_ == 1:
            print('first pass, not resuming from checkpoint')
            return super(AnimWriter, self).prepare_opt()

        # if we're at the first frame of a pass, there is no prior frame to warp
        if (self.direction == 1 and self.this_frame == self.settings.first_frame) or \
                (self.direction == 0 and self.this_frame == self.settings.last_frame):
            print('first frame of pass, no prior frame to warp')

            # the first frame of a pass may not have a checkpoint if we're early in the process
            if os.path.isfile(curr_frame_checkpoint_fp):
                print('resuming from checkpoint:', curr_frame_checkpoint_fp)

                opt_tensor = utils.image_to_tensor(curr_frame_checkpoint_fp, self.settings.core.cuda,
                                                   colorspace=self.settings.out_colorspace)

                opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
                return opt_tensor
            else:
                print('no checkpoint to resume from')
                return super(AnimWriter, self).prepare_opt()

        print('resuming from checkpoint:', curr_frame_checkpoint_fp)

        curr_frame_checkpoint_buf = oiio.ImageBuf(curr_frame_checkpoint_fp)
        curr_frame_checkpoint_np = curr_frame_checkpoint_buf.get_pixels(roi=curr_frame_checkpoint_buf.roi_full)

        # warp the prior frame checkpoint and blend with the current frame checkpoint
        prev_frame_checkpoint_fp = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.out)

        # t = 1 if self.direction else -1
        fs = prev_frame_checkpoint_fp.split('/')
        fs[-1] = '%03d_' % (self.pass_) + fs[-1]
        prev_frame_checkpoint_fp = '/'.join(fs)

        print('warping from previous frame checkpoint:', prev_frame_checkpoint_fp)

        prev_frame_checkpoint_buf = oiio.ImageBuf(prev_frame_checkpoint_fp)
        prev_frame_checkpoint_np = prev_frame_checkpoint_buf.get_pixels(roi=prev_frame_checkpoint_buf.roi_full)

        print('flow file:', flow)

        this_frame_flow_buf = oiio.ImageBuf(flow)
        this_frame_flow_np = this_frame_flow_buf.get_pixels(roi=this_frame_flow_buf.roi_full)

        depth = re.sub(FRAME_REGEX, '.%s.' % self.this_frame, self.settings.depth)

        curr_frame_depth_buf = oiio.ImageBuf(depth)
        curr_frame_depth_np = curr_frame_depth_buf.get_pixels(roi=curr_frame_depth_buf.roi_full)

        x, y, z = curr_frame_depth_np.shape
        # prev_frame_warped_np = np.zeros((x, y, z), dtype=np.float32)
        prev_frame_warped_np = np.copy(curr_frame_checkpoint_np)

        boundary = 10

        temporal_coherence.sample_back(prev_frame_checkpoint_np, this_frame_flow_np, boundary,
                                       prev_frame_warped_np, interpolation='nn')

        curr_frame_mask_buf = oiio.ImageBuf(flow_weight)
        curr_frame_mask_np = curr_frame_mask_buf.get_pixels(roi=curr_frame_mask_buf.roi_full)

        if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            # a = '%s/%s_prev_frame_checkpoint.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            # utils.np_write(prev_frame_checkpoint_np, a, silent=True)

            b = '%s/%s_prev_frame_warped.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            utils.np_write(prev_frame_warped_np, b, silent=True)

            c = '%s/%s_curr_frame_checkpoint.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            utils.np_write(curr_frame_checkpoint_np, c, silent=True)

        # to do: should weighted random take an alpha?
        if self.settings.blend_method == 'random_weighted':

            curr_frame_checkpoint_np = utils.weighted_random_alpha(prev_frame_warped_np,
                                                                   curr_frame_checkpoint_np,
                                                                   curr_frame_mask_np,
                                                                   weight=self.settings.warp_weight
                                                                   )

            if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
                d = '%s/%s_weighted_random.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
                utils.np_write(curr_frame_checkpoint_np, d, silent=True)

        elif self.settings.blend_method == 'none':
            # curr_frame_checkpoint_np = prev_frame_warped_np
            temporal_coherence.a_over_b(prev_frame_warped_np, curr_frame_checkpoint_np, curr_frame_mask_np, curr_frame_checkpoint_np)

        elif self.settings.blend_method == 'blend_weighted':

            blend_output = np.zeros((x, y, z))

            utils.blend_a_b(prev_frame_warped_np, curr_frame_checkpoint_np, curr_frame_mask_np,
                            self.settings.warp_weight, blend_output)

            curr_frame_checkpoint_np = blend_output

            # if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            #     c = '%s/%s_prev_frame_warped_weighted.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            #     utils.np_write(prev_frame_warped_np, c, silent=True)
            #     d = '%s/%s_curr_frame_checkpoint.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            #     utils.np_write(curr_frame_checkpoint_np, d, silent=True)
            #
            # weighted_prev_frame_warped_np = prev_frame_warped_np * self.settings.warp_weight * curr_frame_mask_np
            # weighted_curr_frame_checkpoint_np = curr_frame_checkpoint_np * (1.0 - self.settings.warp_weight)

            # curr_frame_checkpoint_np += (prev_frame_warped_np * curr_frame_mask_np)
            # if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            #     e = '%s/%s_curr_frame_sum.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            #     utils.np_write(curr_frame_checkpoint_np, e, silent=True)

            # normalise
            # warp_divisor = np.full((x, y, z), self.settings.warp_weight, dtype=np.float32)
            # checkpoint_divisor = np.full((x, y, z), (1 - self.settings.warp_weight), dtype=np.float32)

            # divisor = warp_divisor + checkpoint_divisor
            # current_frame_sum_np = weighted_prev_frame_warped_np + weighted_curr_frame_checkpoint_np
            # current_frame_normalised_np = current_frame_sum_np / divisor

            # divisor = np.ones((x, y, z), dtype=np.float32)
            # divisor += curr_frame_mask_np
            # curr_frame_checkpoint_np /= divisor

            # if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            #     e = '%s/%s_curr_frame_sum.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            #     utils.np_write(current_frame_sum_np, e, silent=True)
            #     f = '%s/%s_curr_frame_normalised.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            #     utils.np_write(current_frame_normalised_np, f, silent=True)

        # composite over the given opt image using the motion mask as an aplha channel
        # orig_opt_fp = self.settings.opt_image.rgb_filepath
        # orig_opt_buf = oiio.ImageBuf(orig_opt_fp)
        # orig_opt_np = orig_opt_buf.get_pixels(roi=orig_opt_buf.roi_full)

        # temporal_coherence.a_over_b(prev_frame_warped_np, curr_frame_checkpoint_np, curr_frame_mask_np, curr_frame_checkpoint_np)

        # comp over the original opt input:
        # temporal_coherence.a_over_b(curr_frame_checkpoint_np, orig_opt_np, curr_frame_mask_np, curr_frame_checkpoint_np)

        if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            # g = '%s/%s_a_over_b.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            # utils.np_write(curr_frame_checkpoint_np, g, silent=True)

            h = '%s/%s_mask.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            utils.np_write(curr_frame_mask_np, h)

        curr_frame_mask_np_inv = np.ones((x, y, z), dtype=np.float32) - curr_frame_mask_np

        if self.settings.grad_mask_blur:
            # blur for gradient domain blurring, should harmonise edges (?)
            curr_frame_mask_np_inv = sc.rgb2gray(curr_frame_mask_np_inv)
            curr_frame_mask_np_inv = sf.gaussian(curr_frame_mask_np_inv, sigma=self.settings.grad_mask_blur, truncate=1.0)
            curr_frame_mask_np_inv = sc.gray2rgb(curr_frame_mask_np_inv)

        temporal_weight_mask = utils.np_to_tensor(curr_frame_mask_np_inv, self.settings.core.cuda, raw=True)

        # self.nst.temporal_weight_mask = torch.clamp(temporal_weight_mask, 0.0, 1.0)
        self.nst.temporal_weight_mask = torch.clamp(temporal_weight_mask, self.settings.grad_mask_min, 1.0)
        # self.nst.temporal_weight_mask = temporal_weight_mask

        if self.settings.debug_output and self.this_frame in self.settings.debug_frames:
            i = '%s/%s_mask_inv.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
            mpib = utils.tensor_to_buf(self.nst.temporal_weight_mask, raw=True)
            utils.write_exr(mpib, i)

        # self.nst.temporal_weight_mask = utils.np_to_tensor(curr_frame_mask_np, self.settings.core.cuda, raw=True)
        # print(4.5, self.nst.temporal_weight_mask.max())

        # move to imagenet space
        opt_tensor = utils.np_to_tensor(curr_frame_checkpoint_np, self.settings.core.cuda,
                                        colorspace=self.settings.out_colorspace)

        # move to gpu
        if self.settings.core.cuda:
            device = core_utils.get_cuda_device()
            opt_tensor = opt_tensor.detach().to(torch.device(device))
        opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)

        return opt_tensor

    # def prepare_opt(self):
    #
    #     if not self.settings.interleaved:
    #         return super(AnimWriter, self).prepare_opt()
    #
    #     # backwards->forwards pass
    #     if self.direction == 1:
    #         prev_frame = self.this_frame - 1
    #         flow = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_back)
    #         flow_weight = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_fore_weight)
    #
    #     # forwards->backwards pass
    #     elif self.direction == 0:
    #         prev_frame = self.this_frame + 1
    #         flow = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_fore)
    #         flow_weight = re.sub(FRAME_REGEX, ".%s." % self.this_frame, self.settings.motion_back_weight)
    #
    #     else:
    #         raise Exception('Direction must be 0 or 1')
    #
    #     # print('flow filepath:', flow)
    #
    #     curr_frame_checkpoint_fp = self.settings.out
    #     # print('output filepath:', curr_frame_checkpoint_fp)
    #
    #     if self.pass_ == 1:
    #         # print('first pass, not resuming from checkpoint')
    #         return super(AnimWriter, self).prepare_opt()
    #
    #     # if we're at the first frame of a pass, there is no prior frame to warp
    #     if (self.direction == 1 and self.this_frame == self.settings.first_frame) or \
    #             (self.direction == 0 and self.this_frame == self.settings.last_frame):
    #         # print('first frame of pass, no prior frame to warp')
    #
    #         # the first frame of a pass may not have a checkpoint if we're early in the process
    #         if os.path.isfile(curr_frame_checkpoint_fp):
    #             # print('resuming from checkpoint:', curr_frame_checkpoint_fp)
    #
    #             opt_tensor = utils.image_to_tensor(curr_frame_checkpoint_fp, self.settings.core.cuda,
    #                                                colorspace=self.settings.out_colorspace)
    #
    #             opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
    #             return opt_tensor
    #         else:
    #             # print('no checkpoint to resume from')
    #             return super(AnimWriter, self).prepare_opt()
    #
    #     # print('resuming from checkpoint:', curr_frame_checkpoint_fp)
    #
    #     curr_frame_checkpoint_buf = oiio.ImageBuf(curr_frame_checkpoint_fp)
    #     curr_frame_checkpoint_np = curr_frame_checkpoint_buf.get_pixels(roi=curr_frame_checkpoint_buf.roi_full)
    #
    #     # warp the prior frame checkpoint and blend with the current frame checkpoint
    #     prev_frame_checkpoint_fp = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.out)
    #     # print('warping from frame:', prev_frame_checkpoint_fp)
    #
    #     prev_frame_checkpoint_buf = oiio.ImageBuf(prev_frame_checkpoint_fp)
    #     prev_frame_checkpoint_np = prev_frame_checkpoint_buf.get_pixels(roi=prev_frame_checkpoint_buf.roi_full)
    #
    #     this_frame_flow_buf = oiio.ImageBuf(flow)
    #     this_frame_flow_np = this_frame_flow_buf.get_pixels(roi=this_frame_flow_buf.roi_full)
    #
    #     depth = re.sub(FRAME_REGEX, '.%s.' % self.this_frame, self.settings.depth)
    #     # print('depth filepath:', depth)
    #
    #     curr_frame_depth_buf = oiio.ImageBuf(depth)
    #     curr_frame_depth_np = curr_frame_depth_buf.get_pixels(roi=curr_frame_depth_buf.roi_full)
    #
    #     x, y, z = curr_frame_depth_np.shape
    #     prev_frame_warped_np = np.zeros((x, y, z), dtype=np.float32)
    #     boundary = 10
    #     # temporal_coherence.depth_warp(prev_frame_checkpoint_np, this_frame_flow_np, curr_frame_depth_np,
    #     #                               boundary, prev_frame_warped_np)
    #
    #     temporal_coherence.sample_back(prev_frame_checkpoint_np, this_frame_flow_np, boundary, prev_frame_warped_np,
    #                                    interpolation='nn')
    #
    #     curr_frame_mask_buf = oiio.ImageBuf(flow_weight)
    #     curr_frame_mask_np = curr_frame_mask_buf.get_pixels(roi=curr_frame_mask_buf.roi_full)
    #
    #     if self.settings.debug_output:
    #         a = '%s/%s_prev_frame_checkpoint.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(prev_frame_checkpoint_np, a, silent=True)
    #
    #         b = '%s/%s_prev_frame_warped.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(prev_frame_warped_np, b, silent=True)
    #
    #     # apply weights
    #     prev_frame_warped_np *= curr_frame_mask_np
    #
    #     if self.settings.debug_output:
    #         c = '%s/%s_prev_frame_warped_weighted.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(prev_frame_warped_np, c, silent=True)
    #
    #         d = '%s/%s_curr_frame_checkpoint.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(curr_frame_checkpoint_np, d, silent=True)
    #
    #     # # add to the current frame
    #     # curr_frame_checkpoint_np += (prev_frame_warped_np * curr_frame_mask_np)
    #     # curr_frame_checkpoint_np = prev_frame_warped_np
    #
    #     if self.settings.debug_output:
    #         e = '%s/%s_curr_frame_sum.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(curr_frame_checkpoint_np, e, silent=True)
    #
    #     # # # normalise
    #     # divisor = np.ones((x, y, z), dtype=np.float32)
    #     # divisor += curr_frame_mask_np
    #     # curr_frame_checkpoint_np /= divisor
    #
    #     if self.settings.debug_output:
    #         f = '%s/%s_curr_frame_normalised.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(curr_frame_checkpoint_np, f, silent=True)
    #
    #     # composite over the given opt image using the motion mask as an aplha channel
    #     orig_opt_fp = self.settings.opt_image.rgb_filepath
    #     orig_opt_buf = oiio.ImageBuf(orig_opt_fp)
    #     orig_opt_np = orig_opt_buf.get_pixels(roi=orig_opt_buf.roi_full)
    #     # b = orig_opt_np
    #     # alpha = curr_frame_mask_np
    #     # premult_b = b * alpha
    #     # a = curr_frame_checkpoint_np
    #     # a += b * alpha
    #
    #     temporal_coherence.a_over_b(curr_frame_checkpoint_np, orig_opt_np, curr_frame_mask_np, curr_frame_checkpoint_np)
    #
    #     if self.settings.debug_output:
    #         g = '%s/%s_a_over_b.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         utils.np_write(curr_frame_checkpoint_np, g, silent=True)
    #
    #     curr_frame_mask_np_inv = np.ones((x, y, z), dtype=np.float32) - curr_frame_mask_np
    #     # print(4.4, curr_frame_mask_np.max())
    #
    #     # this is kind of dodgy setting this here but ok while testing
    #     # self.nst.temporal_weight_mask = curr_frame_mask_np_inv
    #     # curr_frame_mask_np_inv = np.clip(curr_frame_mask_np_inv, 0.1, 1.0)
    #     temporal_weight_mask = utils.np_to_tensor(curr_frame_mask_np_inv, self.settings.core.cuda, raw=True)
    #     # self.nst.temporal_weight_mask = torch.clamp(temporal_weight_mask, 0.1, 1.0)
    #     self.nst.temporal_weight_mask = torch.clamp(temporal_weight_mask, 0.0, 1.0)
    #     # self.nst.temporal_weight_mask = utils.np_to_tensor(curr_frame_mask_np_inv, self.settings.core.cuda, raw=True)
    #
    #     if self.settings.debug_output:
    #         h = '%s/%s_mask.%04d.exr' % (self.outdir, self.pass_, self.this_frame)
    #         mpib = utils.tensor_to_buf(self.nst.temporal_weight_mask, raw=True)
    #         utils.write_exr(mpib, h)
    #
    #     # self.nst.temporal_weight_mask = utils.np_to_tensor(curr_frame_mask_np, self.settings.core.cuda, raw=True)
    #     # print(4.5, self.nst.temporal_weight_mask.max())
    #
    #     # move to imagenet space
    #     opt_tensor = utils.np_to_tensor(curr_frame_checkpoint_np, self.settings.core.cuda,
    #                                     colorspace=self.settings.out_colorspace)
    #
    #     # move to gpu
    #     if self.settings.core.cuda:
    #         device = core_utils.get_cuda_device()
    #         opt_tensor = opt_tensor.detach().to(torch.device(device))
    #     opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
    #
    #     return opt_tensor
    #

