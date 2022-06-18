import os
import yaml

import OpenImageIO as oiio
from torch.autograd import Variable

from . import utils

from nst.core import model, loss
from nst.core import utils as core_utils


class Style(object):
    def __init__(self, image):
        self.image = image
        self.in_mask = ''
        self.target_map = None
        self.out_mask = ''
        self.scale = 1.0
        self.colorspace = 'srgb_texture'

    def __repr__(self):
        return yaml.dump(self)

class Content(object):
    def __init__(self, image):
        self.image = image
        self.colorspace = 'acescg'
        self.scale = 1.0

    def __repr__(self):
        return yaml.dump(self)


"""
from nst.oiio import Style, Content, StyleImager
style1 = Style('style.exr')
content = Content('content.exr')
si = StyleImager(styles=[style1], content=content, engine='gpu')
si
"""

class StyleImager(object):

    def __init__(self, styles=None, content=None, engine='cpu'):
        self.styles = styles
        self.engine = engine
        self.content = content
        self.from_content = True
        self.iterations = 500
        self.log_iterations = 50
        self.opt_image = None
        self.out = ''
        self.output_dir = ''
        self.scale = 1.0
        self.cuda_device = None
        self.pyramid_scale_factor = 0.63
        self.style_mips = 4
        self.style_layers = ['p1', 'p2', 'r31', 'r42']
        self.style_layer_weights = [1.0, 1.0, 1.0, 1.0]
        self.content_layer = 'r41'
        self.content_layer_weight = 1.0
        self.style_mip_weights = [1.0] * self.style_mips
        self.content_mips = 1
        self.do_cuda = False
        self.optimiser = 'lbfgs'
        self.learning_rate = 1.0
        self.progressive_output = False
        self.progressive_ext = 'jpg'
        self.write_gradients = False
        self.gradient_ext = 'jpg'
        self.write_style_pyramid = False
        self.write_style_activations = False
        self.write_content_activations = False
        self.write_gram = False
        self.opt_colorspace = 'srgb_texture'
        self.frame = ''
        self.out_colorspace = 'srgb_texture'
        self.model_filepath = os.getenv('NST_VGG_MODEL')

    def prepare_model(self):
        self.vgg, self.do_cuda = core_utils.get_vgg(self.engine, self.model_filepath)
        self.nst = model.Nst()
        self.nst.engine = self.engine
        self.nst.model_path = self.model_filepath
        self.nst.vgg = self.vgg
        self.nst.cuda = self.do_cuda
        self.nst.optimiser = self.optimiser
        self.nst.pyramid_scale_factor = self.pyramid_scale_factor
        self.nst.style_mips = self.style_mips
        self.nst.style_layers = self.style_layers
        self.nst.style_layer_weights = self.style_layer_weights
        self.nst.content_layer = self.content_layer
        self.nst.content_layer_weight = self.content_layer_weight
        self.nst.style_mip_weights = self.style_mip_weights
        self.nst.content_mips = self.content_mips
        self.nst.learning_rate = self.learning_rate
        self.nst.scale = self.scale
        self.nst.iterations = self.iterations
        self.nst.log_iterations = self.log_iterations

    def get_output_dir(self):
        return os.path.abspath(os.path.join(self.out, (os.path.pardir)))

    def save(self, fp):
        pardir = os.path.abspath(os.path.join(fp, os.path.pardir))
        os.makedirs(pardir, exist_ok=True)
        with open(fp, mode="wt", encoding="utf-8") as file:
            yaml.dump(self, file)

    @staticmethod
    def load(fp):
        with open(fp, mode="r", encoding='utf-8') as file:
            return yaml.load(file, yaml.Loader)

    def __str__(self):
        return yaml.dump(self)

    def __repr__(self):
        return yaml.dump(self)

    def write_exr(self, frame: int=None) -> None:
        if self.content:
            if self.content.image:
                if '####' in self.content.image and frame:
                    self.content.image = self.content.image.replace('####', '%04d' % frame)
                else:
                    self.content.image = self.content.image

        if self.out:
            if '####' in self.out and frame:
                self._out = self.out.replace('####', '%04d' % frame)
            else:
                self._out = self.out

        out_dir = os.path.abspath(os.path.join(self.out, os.path.pardir))
        os.makedirs(out_dir, exist_ok=True)

        self.prepare_model()

        if self.content:
            self.nst.content = self.prepare_content()

        if self.opt_image:
            self.nst.opt_tensor = self.prepare_opt(clone=self.opt_image)
        elif self.from_content:
            self.nst.opt_tensor = self.prepare_opt(clone=self.content.image)

        self.nst.styles = self.prepare_styles()
        self.nst.prepare()

        # call the forward method of the model, i.e. run inference
        tensor = self.nst()

        buf = utils.tensor_to_buf(tensor)

        if self.out_colorspace != 'srgb_texture':
            buf = oiio.ImageBufAlgo.colorconvert(buf, 'srgb_texture', self.out_colorspace)

        buf.write(self._out)

    def prepare_styles(self):
        styles = []

        for style in self.styles:
            style_tensor = utils.image_to_tensor(style.image, self.do_cuda, colorspace=style.colorspace)
            s = model.TorchStyle(style_tensor)
            if style.in_mask:
                s.in_mask = utils.image_to_tensor(style.in_mask, self.do_cuda, raw=True)
            if style.target_map:
                s.target_map = utils.image_to_tensor(style.target_map, self.do_cuda, raw=True)
            styles.append(s)

        return styles

    def prepare_content(self):
        content_tensor = utils.image_to_tensor(self.content.image, self.do_cuda, colorspace=self.content.colorspace)
        return content_tensor

    def prepare_opt(self, clone=None):
        if clone:
            if self.content:
                tensor = utils.image_to_tensor(clone, self.do_cuda, colorspace=self.content.colorspace)

            else:
                tensor = utils.image_to_tensor(clone, self.do_cuda, colorspace=self.opt_colorspace)

            opt_tensor = Variable(tensor.data.clone(), requires_grad=True)

        return opt_tensor

    def write_style_gram(self, ext='exr'):
        s_index = 0
        for s in self.styles:

            if s.in_mask:
                style_in_mask_tensor = utils.image_to_tensor(s.in_mask, self.do_cuda, raw=True)
            else:
                style_in_mask_tensor = None

            style_tensor = utils.image_to_tensor(s.image, self.do_cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.do_cuda,
                                                                mips=self.style_mips,
                                                                pyramid_scale_factor=self.pyramid_scale_factor)
            style_activations = []
            style_layer_names = [x.name for x in s.layers]
            for layer_activation_pyramid in self.vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            vgg_layer_index = 0
            for vgg_layer in style_activations:
                gram_pyramid = []
                mip_index = 0
                for mip_activations in vgg_layer:
                    layer_name = s.layers[vgg_layer_index].name
                    gram = loss.GramMatrix()(mip_activations).detach()
                    if self.write_gram:
                        fp = '%s/gram/style_%s/layer_%s/mip_%s.%s' % ( self.get_output_dir(), s_index, layer_name, mip_index, ext)
                        utils.write_gram(gram, fp)
                    gram_pyramid += [gram]
                    mip_index += 1
                vgg_layer_index += 1

            s_index += 1

    def write_style_pyramids(self, ext='jpg'):
        s_index = 0
        for s in self.styles:
            style_tensor = utils.image_to_tensor(s.image, self.do_cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.do_cuda, mips=self.style_mips, pyramid_scale_factor=self.pyramid_scale_factor)
            outdir = '%s/style_%02d/pyramid' % (self.get_output_dir(), s_index)
            utils.Pyramid.write_gaussian_pyramid(style_pyramid, outdir, ext=ext)
            s_index += 1

    def write_style_activations(self, layer_limit=10, ext='exr'):
        s_index = 0
        for s in self.styles:

            if s.in_mask:
                style_in_mask_tensor = utils.image_to_tensor(s.in_mask, self.do_cuda, raw=True)
            else:
                style_in_mask_tensor = None

            style_tensor = utils.image_to_tensor(s.image, self.do_cuda, colorspace=s.colorspace)
            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=self.do_cuda, mips=self.style_mips, pyramid_scale_factor=self.pyramid_scale_factor)
            style_activations = []
            style_layer_names = [x.name for x in s.layers]
            for layer_activation_pyramid in self.vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            vgg_layer_index = 0
            for vgg_layer in style_activations:
                layer_name = s.layers[vgg_layer_index].name
                mip_index = 0
                for mip_activations in vgg_layer:
                    _, c, w, h = mip_activations.size()
                    outdir = '%s/style_%02d/activations/layer_%s/mip_%s' % (self.get_output_dir(), s_index, layer_name, mip_index)
                    utils.write_activations(mip_activations, outdir, layer_limit=layer_limit, ext=ext)
                    mip_index += 1
                vgg_layer_index += 1

            s_index += 1

    def write_content_activations(self, layer_limit=10, ext='exr'):

        if self.content:
            content_tensor = self._prepare_content()
            content_layers = self.content.layers
            content_activations = []
            content_layer_names = [x.name for x in content_layers]
            for layer_activation_pyramid in self.vgg([content_tensor], content_layer_names):
                content_activations.append(layer_activation_pyramid)

            if self.write_content_activations:
                vgg_layer_index = 0
                for vgg_layer in content_activations:
                    layer_name = self.content.layers[vgg_layer_index].name
                    mip_index = 0
                    for mip_activations in vgg_layer:
                        _, c, w, h = mip_activations.size()
                        outdir = '%s/activations/content/layer_%s/mip_%s' % (
                        self.get_output_dir(), layer_name, mip_index)
                        utils.write_activations(mip_activations, outdir, layer_limit=layer_limit, ext=ext)
                        mip_index += 1
                    vgg_layer_index += 1
