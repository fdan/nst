import re
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
                 content: settings.Image=None):

        self.settings = settings.WriterSettings()
        self.settings.styles = styles
        self.settings.opt_image = opt_image
        self.settings.content = content
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
                    # print(2.1, self.settings.content.rgb_filepath)

            if self.settings.opt_image:
                if self.settings.opt_image.rgb_filepath:
                    self.settings.opt_image.rgb_filepath = re.sub(FRAME_REGEX, ".%s." % frame, self.settings.opt_image.rgb_filepath)
                    # print(2.2, self.settings.opt_image.rgb_filepath)

            if self.settings.out:
                self.settings.out = re.sub(FRAME_REGEX, ".%s." % frame, self.settings.out)

        out_dir = os.path.abspath(os.path.join(self.settings.out, os.path.pardir))
        os.makedirs(out_dir, exist_ok=True)

        self.prepare_model()

        if self.settings.content:
            self.nst.content = self.prepare_content()

        self.nst.opt_tensor = self.prepare_opt()

        self.nst.styles = self.prepare_styles()
        self.nst.prepare()
        assert self.settings.core == self.nst.settings
        # print(5.5, self.nst.settings)

        # set num cpus
        torch.set_num_threads = int(self.settings.core.cpu_threads)

        # do style transfer
        tensor = self.nst()

        out_format = self.settings.out.split('.')[-1]

        # if self.settings.output_format == 'exr':
        if out_format == 'exr':
            print('writing exr ', self.settings.out)
            buf = utils.tensor_to_buf(tensor)
            if self.settings.out_colorspace != 'srgb_texture':
                buf = oiio.ImageBufAlgo.colorconvert(buf, 'srgb_texture', self.settings.out_colorspace)
            buf.write(self.settings.out)

        # elif self.settings.output_format == 'pt':
        elif out_format == 'pt':
            print('writing pt %s \n' % self.settings.out)
            torch.save(tensor, self.settings.out)

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

    def prepare_opt(self):

        # if self.settings.core.temporal_weight:
        #     this_frame = int(self.settings.frame) + int(self.settings.pre)
        #     re.sub(FRAME_REGEX, ".%s." % this_frame)
        #     pass
        #     # 1.  check if prev frame output exists


        opt_filepath = self.settings.opt_image.rgb_filepath

        if opt_filepath.endswith('.exr'):
            opt_tensor = utils.image_to_tensor(opt_filepath, self.settings.core.cuda,
                                           colorspace=self.settings.opt_image.colorspace)

        elif opt_filepath.endswith('.pt'):
            opt_tensor = torch.load(opt_filepath)

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
        # self.settings.output_format = 'pt'
        self._opt_tensor = torch.zeros(1)

    def load_output(self, path):
        if path.endswith('.exr'):
            tensor = utils.image_to_tensor(path, self.settings.core.cuda,
                                           colorspace=self.settings.opt_image.colorspace)

        elif path.endswith('.pt'):
            tensor = torch.load(path)

        return tensor

    # def prepare_opt(self):
    #
    #     opt_filepath = self.settings.opt_image.rgb_filepath
    #     opt_tensor = self.load_output(opt_filepath)
    #
    #     if self.settings.core.temporal_weight:
    #         # prev frame can be +1 or -1 depending on starting pass
    #         prev_frame = int(self.settings.frame) - self.settings.starting_pass
    #         print('prev frame:', prev_frame)
    #         prev_output = re.sub(FRAME_REGEX, prev_frame, self.settings.out)
    #
    #         if os.path.isfile(prev_output):
    #             # slide to a numpy image
    #             prev_opt = self.load_output(prev_output)
    #             prev_opt_np = '' # to cpu, to numpy, slice, transpose?
    #
    #             # load flow
    #             if self.settings.starting_pass:
    #                 flow_path = self.settings.motion_fore
    #             else:
    #                 flow_path = self.settings.motion_back
    #             flow_path = re.sub(FRAME_REGEX, prev_frame, flow_path)
    #             flow_buf = oiio.ImageBuf(flow_path)
    #             flow_np = flow_buf.get_pixels(roi=flow_buf.roi_full)
    #
    #             # load depth
    #             depth_path = re.sub(FRAME_REGEX, prev_frame, self.settings.depth_map.rgb_filepath)
    #             depth_buf = oiio.ImageBuf(depth_path)
    #             depth_np = depth_buf.get_pixels(roi=depth_buf.roi_full)
    #
    #             x, y, z = flow_np.shape
    #             warp_output = np.zeros((x, y, z), dtype=np.float32)
    #
    #             temporal_coherence.depth_warp(prev_opt_np, flow_np, depth_np, warp_output)
    #
    #     # composite warp_output and opt_tensor, using motion mask as alpha
    #     # store motion mask for temporal guide
    #
    #     opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
    #     return opt_tensor

    def prepare_temporal_weight_mask(self):
        temporal_weight_tensor = utils.image_to_tensor(self.settings.temporal_weight_mask.rgb_filepath,
                                                       self.settings.core.cuda,
                                                       raw=True
                                                       )

        return temporal_weight_tensor

    def write(self) -> None:

        self.prepare_model()

        if self.settings.temporal_weight_mask:
            self.nst.temporal_weight_mask = self.prepare_temporal_weight_mask()

        super(AnimWriter, self).write()

    def run(self):
        if self.settings.interleaved:
            self._run_interleaved()
        else:
            self._run_sequential()

    def _run_sequential(self):
        """
        Process one frame at a time to full optimisation, from start to finish
        """
        pass

    def _run_interleaved(self):
        """
        Interleave results across the sequence, slowly optimise each frame and propagate results
        """
        for pass_ in range(self.settings.starting_pass, self.settings.passes+1):
            print('pass:', pass_)

            # if direction is 1, we are in a forward pass.  if 0, negative.
            self.settings.core.iterations = int(self.settings.core.iterations / self.settings.passes)

            direction = pass_ % 2
            direction_ = 'forward' if direction else 'backward'
            print('direction:', direction_)

            start = self.settings.last_frame if direction == 0 else self.settings.first_frame
            end = self.settings.first_frame if direction == 0 else self.settings.last_frame
            increment_by = -1 if direction == 0 else 1

            # if direction is backwards, use the forewards disocclusion mask

            for this_frame in range(start, end, increment_by):
                print('frame:', this_frame)
                self.settings.frame = this_frame

                # warp_from_frame is the temporal prior in the context of pass direction
                # i.e. the color image we are wanting to warp
                if this_frame == self.settings.first_frame and pass_ == 1:
                    warp_from_frame = -1
                elif this_frame == self.settings.first_frame:
                    warp_from_frame = (this_frame + 1)
                elif this_frame == self.settings.last_frame:
                    warp_from_frame = (this_frame - 1)
                else:
                    warp_from_frame = (this_frame + 1) if direction == 0 else (this_frame - 1)

                # backwards->forwards pass
                if direction == 1:
                    prev_frame = this_frame - 1
                    flow = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.motion_fore)
                    flow_weight = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.motion_fore_weight)

                # forwards->backwards pass
                elif direction == 0:
                    prev_frame = this_frame + 1
                    flow = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.motion_back)
                    flow_weight = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.motion_back_weight)
                else:
                    raise Exception('Direction must be 0 or 1')

                out_fp = re.sub(FRAME_REGEX, ".%s." % this_frame, self.settings.out)

                # check if a pt exists for the prior frame
                prev_out_fp = re.sub(FRAME_REGEX, ".%s." % prev_frame, self.settings.out)
                prev_out_fp_pt = prev_out_fp.replace('.exr', '.pt')
                if os.path.isfile(prev_out_fp_pt):
                    print('warping previous frame:', prev_out_fp_pt)
                    prev_out_fp_tensor = torch.load(prev_out_fp_pt)
                    print(3.1, prev_out_fp_tensor.size())

                # load the flow and flow weight into tensors

                out_fp_pt = out_fp.replace('.exr', '.pt')
                if os.path.isfile(out_fp_pt):
                    print('found previous .pt for frame:', out_fp_pt)
                    out_fp = out_fp_pt
                    self.settings.opt_image.rgb_filepath = out_fp

                # main nst call
                self.write()

        #todo:
        # for each frame, convert the .pt to .exr



