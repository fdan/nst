from typing import List
import json

import os
import yaml

import OpenImageIO as oiio
from torch.autograd import Variable

from . import utils

from nst.core import model, loss
from nst.core import utils as core_utils
import nst.settings as settings

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

    def load(self, fp):
        self.settings.load(fp)

    def prepare_model(self):
        self.vgg, self.settings.core.cuda = core_utils.get_vgg(self.settings.core.engine,
                                                               self.settings.core.model_path)
        self.nst = model.Nst()
        self.nst.settings = self.settings.core

    def get_output_dir(self):
        if not self.output_dir:
            self.output_dir = os.path.abspath(os.path.join(self.settings.out, (os.path.pardir)))
        return self.output_dir

    def write_exr(self, frame: int=None) -> None:
        if self.settings.content:
            if self.settings.content.rgb_filepath:
                if '####' in self.settings.content.rgb_filepath and frame:
                    self.settings.content.rgb_filepath = self.settings.content.rgb_filepath.replace('####', '%04d' % frame)
                else:
                    self.settings.content.rgb_filepath = self.settings.content.rgb_filepath

        if self.settings.out:
            if '####' in self.settings.out and frame:
                out_fp = self.settings.out.replace('####', '%04d' % frame)
            else:
                out_fp = self.settings.out

        out_dir = os.path.abspath(os.path.join(self.settings.out, os.path.pardir))
        os.makedirs(out_dir, exist_ok=True)

        self.prepare_model()

        if self.settings.content:
            self.nst.content = self.prepare_content()

        self.nst.opt_tensor = self.prepare_opt()

        self.nst.styles = self.prepare_styles()
        self.nst.prepare()
        assert self.settings.core == self.nst.settings

        # call the forward method of the model, i.e. run inference
        tensor = self.nst()

        buf = utils.tensor_to_buf(tensor)

        if self.settings.out_colorspace != 'srgb_texture':
            buf = oiio.ImageBufAlgo.colorconvert(buf, 'srgb_texture', self.settings.out_colorspace)

        buf.write(out_fp)

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
        opt_tensor = utils.image_to_tensor(self.settings.opt_image.rgb_filepath, self.settings.core.cuda,
                                       colorspace=self.settings.opt_image.colorspace)

        opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
        return opt_tensor

    def write_style_gram(self, ext='exr'):
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
