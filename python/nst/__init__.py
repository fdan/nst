import memory_profiler
import yaml
import random
import os
from timeit import default_timer as timer

import OpenImageIO as oiio
from torch.autograd import Variable # deprecated - use Tensor
from torch import optim

from PIL import Image

from . import entities
from . import utils

import numpy as np

# https://pytorch.org/docs/stable/notes/randomness.html
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False

LOG = ''
MAX_GPU_RAM_USAGE = 90
DO_CUDA = False
TOTAL_GPU_MEMORY = 1
TOTAL_SYSTEM_MEMORY = 1


def log(msg):
    print(msg)


class VGGLayer(object):
    def __init__(self, name, weight=1.0, mip_weights=[1.0]*4):
        self.name = name
        self.weight = weight
        self.mip_weights = mip_weights

    def __repr__(self):
        return yaml.dump(self)


class Style(object):
    def __init__(self, image, layers):
        self.image = image
        self.in_mask = ''
        self.target_map = None
        self.out_mask = ''
        self.layers = layers
        self.scale = 1.0
        self.mips = 4
        self.colorspace = 'srgb_texture'

    def __repr__(self):
        return yaml.dump(self)


class Content(object):
    def __init__(self, image, layers, in_mask=None, out_mask=None, mips=1):
        self.image = image
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.layers = layers
        self.colorspace = 'acescg'
        self.scale = 1.0
        self.mips = mips

    def __repr__(self):
        return yaml.dump(self)


class StyleImager(object):

    def __init__(self, style1=None, style2=None, content=None, frame=0, render_out=None, engine='cpu'):
        self.style1 = style1
        self.style2 = style2
        self.style_zoom = None
        self.zoom_factor = 0.17
        self.gauss_scale_factor = 0.63
        self.style_rescale = None
        self.content = content
        self.iterations = 500
        self.log_iterations = 100
        self.random_style = False
        self.engine = engine
        self.from_content = True
        self.unsafe = False
        self.progressive = False
        self.max_loss = None
        self.lr = 1
        self.opt_x = 512
        self.opt_y = 512
        self.frame = frame
        self.render_out = render_out
        self.optimisation_image = None
        self.out = ''
        self.output_dir = ''
        self.cuda_device = None
        self.out_colorspace = 'srgb_texture'
        self.opt_colorspace = 'srgb_texture'
        self.write_gradients = False

        if engine == 'gpu':
            self.init_cuda()

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

    def init_cuda(self) -> None:
        self.cuda_device = utils.get_cuda_device()
        # log('cuda device:', self.cuda_device);

    def render_to_disk(self):
        img = self.generate_image()
        img_path = self.render_out

        t_ = img_path.split('/')
        t_.pop()
        d_ = ('/'.join(t_))
        if not os.path.isdir(d_):
            os.makedirs(d_)

        img.save(img_path)

    def generate_image(self, frame: int=None) -> Image:
        tensor = self.generate_tensor(frame=frame)
        return utils.tensor_to_image(tensor)

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

        tensor = self.generate_tensor(frame=frame)
        buf = utils.tensor_to_buf(tensor)

        if self.out_colorspace != 'srgb_texture':
            buf = oiio.ImageBufAlgo.colorconvert(buf, 'srgb_texture', self.out_colorspace)

        buf.write(self._out)

    def generate_tensor(self, frame: int=None) -> torch.Tensor:
        if self.content:
            if self.content.image:
                if '####' in self.content.image and frame:
                    self.content.image = self.content.image.replace('####', '%04d' % frame)
                else:
                    self.content.image = self.content.image

        start = timer()
        vgg = self._prepare_engine()

        loss_layers = []
        loss_fns = []
        weights = []
        targets = []
        masks = []
        mip_weights = []

        output_dir = self.get_output_dir()

        if self.content:
            content_layers = self.content.layers
            content_masks = [torch.Tensor(0)]
            masks += content_masks
            loss_layers += content_layers
            content_loss_fns = [entities.MipMSELoss()] * len(content_layers)
            loss_fns += content_loss_fns
            content_weights = [x.weight for x in self.content.layers]
            weights += content_weights
            content_tensor = self._prepare_content()

            content_activations = []
            content_layer_names = [x.name for x in content_layers]
            for layer_activation_pyramid in vgg([content_tensor], content_layer_names):
                content_activations.append(layer_activation_pyramid)

            content_targets = []
            for layer_activation_pyramid in content_activations:
                thing = []
                for tensor in layer_activation_pyramid:
                    thing += [tensor.detach()]
                content_targets += thing

            targets += content_targets

            for cl in self.content.layers:
                mip_weights += [cl.mip_weights]

        if self.optimisation_image:
            opt_tensor = self.prepare_opt(clone=self.optimisation_image)
        elif self.from_content:
            opt_tensor = self.prepare_opt(clone=self.content.image)
        else:
            opt_tensor = self.prepare_opt()

        if self.style1:
            loss_layers += self.style1.layers

            style_tensor = utils.image_to_tensor(self.style1.image, DO_CUDA, colorspace=self.style1.colorspace)

            if self.style1.in_mask:
                style_in_mask_tensor = utils.image_to_tensor(self.style1.in_mask, DO_CUDA, raw=True)
            else:
                style_in_mask_tensor = None

            if self.style1.target_map:
                style_target_map_tensor_1 = utils.image_to_tensor(self.style1.target_map, DO_CUDA, raw=True)
            else:
                style_target_map_tensor_1 = None

            if self.style_zoom or self.style_rescale:
                style_tensor = utils.zoom_image(style_tensor, self.style_zoom, self.style_rescale, zoom_factor=self.zoom_factor, cuda=DO_CUDA)

                if self.style1.in_mask:
                    style_in_mask_tensor = utils.zoom_image(style_in_mask_tensor, self.style_zoom, self.style_rescale, zoom_factor=self.zoom_factor, cuda=DO_CUDA)

            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=DO_CUDA, mips=self.style1.mips, scale_factor=self.gauss_scale_factor)

            style_activations = []
            style_layer_names = [x.name for x in self.style1.layers]
            for layer_activation_pyramid in vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            style_loss_fns = [entities.MipGramMSELoss01()] * len(self.style1.layers)
            loss_fns += style_loss_fns
            weights += [x.weight for x in self.style1.layers]

            style_targets = []
            for layer_activation_pyramid in style_activations:
                gram_pyramid = []
                for tensor in layer_activation_pyramid:
                    gram = entities.GramMatrix()(tensor).detach()
                    gram_pyramid += [gram]
                style_targets += [gram_pyramid]

            targets += style_targets

            for sl in self.style1.layers:
                mip_weights += [sl.mip_weights]

        if self.style2:
            loss_layers += self.style2.layers

            style_tensor = utils.image_to_tensor(self.style2.image, DO_CUDA, colorspace=self.style2.colorspace)

            if self.style2.in_mask:
                style_in_mask_tensor = utils.image_to_tensor(self.style2.in_mask, DO_CUDA, raw=True)
            else:
                style_in_mask_tensor = None

            if self.style2.target_map:
                style_target_map_tensor_2 = utils.image_to_tensor(self.style2.target_map, DO_CUDA, raw=True)
            else:
                style_target_map_tensor_2 = None

            if self.style_zoom or self.style_rescale:
                style_tensor = utils.zoom_image(style_tensor, self.style_zoom, self.style_rescale, zoom_factor=self.zoom_factor, cuda=DO_CUDA)

                if self.style2.in_mask:
                    style_in_mask_tensor = utils.zoom_image(style_in_mask_tensor, self.style_zoom, self.style_rescale, zoom_factor=self.zoom_factor, cuda=DO_CUDA)

            style_pyramid = utils.Pyramid.make_gaussian_pyramid(style_tensor, cuda=DO_CUDA, mips=self.style2.mips, scale_factor=self.gauss_scale_factor)

            style_activations = []
            style_layer_names = [x.name for x in self.style2.layers]
            for layer_activation_pyramid in vgg(style_pyramid, style_layer_names, mask=style_in_mask_tensor):
                style_activations.append(layer_activation_pyramid)

            style_loss_fns = [entities.MipGramMSELoss01()] * len(self.style2.layers)
            loss_fns += style_loss_fns
            weights += [x.weight for x in self.style2.layers]

            style_targets = []
            for layer_activation_pyramid in style_activations:
                gram_pyramid = []
                for tensor in layer_activation_pyramid:
                    gram = entities.GramMatrix()(tensor).detach()
                    gram_pyramid += [gram]
                style_targets += [gram_pyramid]

            targets += style_targets

            for sl in self.style2.layers:
                mip_weights += [sl.mip_weights]

        # if self.style.image:
        #     for mip in self.style.mips:
        #
        #         # todo: in adopting the gaussian pyramid approach, we need to package the data up differently.
        #         # currently in the closure, we index everything by loss layer, and below we have a loss layer for
        #         # each mip.
        #         #
        #         # we need to change that, such that the style_targets holds gram activations for all mips in a layer.
        #         # so that's a change to the entity relationsip: instead of mips holding layers, layers hold mips.
        #
        #         if mip.layers:
        #             style_layer_names = [x for x in mip.layers]
        #             style_layer_weights = [mip.layers[x] for x in mip.layers]
        #             # print(1.0, 'mip scale:', mip.scale, style_layer_names, style_layer_weights)
        #             loss_layers += style_layer_names
        #
        #             style_tensor = utils.image_to_tensor(self.style.image, DO_CUDA, resize=mip.scale, colorspace=self.style_colorspace)
        #
        #             style_activations = []
        #             for x in vgg(style_tensor, style_layer_names):
        #                 style_activations.append(x)
        #
        #             # style_loss_fns = [entities.GramMSELoss()] * len(style_layer_names)
        #             style_loss_fns = [entities.MipGramMSELoss()] * len(style_layer_names)
        #             loss_fns += style_loss_fns
        #             weights += style_layer_weights
        #
        #             # do not calculate the gram here - we need to know the mip dimensions in the loss fn
        #             # style_targets = style_activations
        #             style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
        #             targets += style_targets
        #
        #             scales += [float(mip.scale)]

        # # todo: do this for each mip
        # if self.style_image:
        #     loss_layers += style_layer_names
        #     style_tensor = utils.image_to_tensor(self.style_image, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)
        #
        #     style_activations = []
        #     for x in vgg(style_tensor, style_layer_names):
        #         style_activations.append(x)
        #
        #     style_loss_fns = [entities.GramMSELoss()] * len(style_layer_names)
        #     loss_fns += style_loss_fns
        #     weights += style_layer_weights
        #     style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
        #     targets += style_targets

        if DO_CUDA:
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

        show_iter = self.log_iterations

        # lbfgs
        optimizer = optim.LBFGS([opt_tensor], lr=self.lr, max_iter=int(self.iterations))

        # adam
        optimizer = optim.Adam([opt_tensor])

        n_iter = [1]
        current_loss = [9999999]

        layer_masks = []

        if self.content:
            layer_masks.append(self.content.out_mask)
        else:
            layer_masks.append(None)

        # for cl in self.content.out_mask:
        #     if not cl:
        #         layer_masks.append(None)
        #     else:
        #         opt_x, opt_y = opt_tensor.size()[2], opt_tensor.size()[3]
        #         mask = oiio.ImageBuf(cl)
        #
        #         # oiio axis are flipped:
        #         scaled_mask = oiio.ImageBufAlgo.resize(mask, roi=oiio.ROI(0, opt_y, 0, opt_x, 0, 1, 0, 3))
        #         mask_np = scaled_mask.get_pixels()
        #         x, y, z = mask_np.shape
        #         mask_np = mask_np[:, :, :1].reshape(x, y)
        #         mask_tensor = torch.Tensor(mask_np).detach().to(torch.device(self.cuda_device))
        #         layer_masks.append(mask_tensor)

        # todo: update to work with mips, one mask per mip (not per mip layer)
        # for sl in style_layer_names:
        #     has_mask = True if 'mask' in self.style_layers[sl] else False
        #
        #     if not has_mask:
        #         layer_masks.append(None)
        #         continue

            # mask_file = self.style.out_mask
            #
            # opt_x, opt_y = opt_tensor.size()[2], opt_tensor.size()[3]
            # mask = oiio.ImageBuf(mask_file)
            #
            # # oiio axis are flipped:
            # scaled_mask = oiio.ImageBufAlgo.resize(mask, roi=oiio.ROI(0, opt_y, 0, opt_x, 0, 1, 0, 3))
            # mask_np = scaled_mask.get_pixels()
            # x, y, z = mask_np.shape
            # mask_np = mask_np[:, :, :1].reshape(x, y)
            # mask_tensor = torch.Tensor(mask_np).detach().to(torch.device(self.cuda_device))
            # layer_masks.append(mask_tensor)

        layer_masks = 10 * [None]

        cuda_device = self.cuda_device

        def closure():
            opt_pyramid = utils.Pyramid.make_gaussian_pyramid(opt_tensor, cuda=DO_CUDA, mips=self.style1.mips,
                                                              scale_factor=self.gauss_scale_factor)
            opt_activations = []

            loss_layer_names = [x.name for x in loss_layers]

            for opt_layer_activation_pyramid in vgg(opt_pyramid, loss_layer_names):
                opt_activations.append(opt_layer_activation_pyramid)

            layer_gradients = []

            if cuda_device:
                loss = torch.zeros(1, requires_grad=False).to(torch.device(cuda_device))
            else:
                loss = torch.zeros(1, requires_grad=False)

            for index, opt_layer_activation_pyramid in enumerate(opt_activations):
                optimizer.zero_grad() # this is really significant - we're zeroing the grads for each layer so we can sum them separately after masking
                target_layer_activation_pyramid = targets[index]

                layer_loss = loss_fns[index](opt_layer_activation_pyramid, target_layer_activation_pyramid, mip_weights[index])
                layer_weight = weights[index]
                weighted_layer_loss = layer_weight * layer_loss
                weighted_layer_loss.backward(retain_graph=True)

                loss += layer_loss

                # # this may not work with multiple styles
                if torch.is_tensor(style_target_map_tensor_1):
                    b, c, w, h = opt_tensor.grad.size()
                    masked_grad = opt_tensor.grad.clone()
                    if self.write_gradients:
                        utils.write_gradient(masked_grad, '%s/grad1/%04d.jpg' % (output_dir, n_iter[0]))
                    for i in range(0, c):
                        masked_grad[0][i] *= style_target_map_tensor_1[0][0]
                    if self.write_gradients:
                        utils.write_gradient(masked_grad, '%s/grad2/%04d.jpg' % (output_dir, n_iter[0]))
                    layer_gradients.append(masked_grad)
                #
                # elif torch.is_tensor(style_target_map_tensor_2):
                #     b, c, w, h = opt_tensor.grad.size()
                #     masked_grad = opt_tensor.grad.clone()
                #     if self.write_gradients:
                #         utils.write_gradient(masked_grad, '%s/grad3/%04d.exr' % (output_dir, n_iter[0]))
                #     for i in range(0, c):
                #         masked_grad[0][i] *= style_target_map_tensor_2[0][0]
                #     if self.write_gradients:
                #         utils.write_gradient(masked_grad, '%s/grad4/%04d.exr' % (output_dir, n_iter[0]))
                #     layer_gradients.append(masked_grad)
                #
                # else:
                #     layer_gradients.append(opt_tensor.grad.clone())
                layer_gradients.append(opt_tensor.grad.clone())

            # normalise: ensure mean activation remains same
            # mask_normalisation = (x * y) / layer_mask.sum()
            # weighted_mask_tensor = torch.div(layer_mask, mask_normalisation)

            b, c, w, h = opt_tensor.grad.size()

            if self.engine == "gpu":
                output_layer_gradient = torch.zeros((b, c, w, h)).detach().to(torch.device(cuda_device))
            else:
                output_layer_gradient = torch.zeros((b, c, w, h)).detach()

            # sum all layer gradients
            for lg in layer_gradients:
                output_layer_gradient += lg
            opt_tensor.grad = output_layer_gradient

            if self.progressive:
                # iterative_output = '%s/iterations/render.%04d.exr' % (output_dir, n_iter[0])
                iterative_output = '%s/iterations/render.%04d.jpg' % (output_dir, n_iter[0])
                os.makedirs(os.path.realpath(os.path.join(iterative_output, os.pardir)), exist_ok=True)
                buf = utils.tensor_to_buf(torch.clone(opt_tensor))
                # utils.write_exr(buf, iterative_output)
                utils.write_jpg(buf, iterative_output)

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % show_iter == (show_iter - 1):

                if torch.__version__ == '1.1.0':
                    max_mem_cached = torch.cuda.max_memory_cached(0) / 1000000
                elif torch.__version__ == '1.7.1':
                    max_mem_cached = torch.cuda.max_memory_reserved(0) / 1000000

                msg = ''
                msg += 'Iteration: %d, ' % (n_iter[0])
                msg += 'loss: %s, ' % (nice_loss)
                if DO_CUDA:
                    msg += 'memory used: %s of %s' % (max_mem_cached, TOTAL_GPU_MEMORY)
                else:
                    mem_usage = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=0.1)
                    msg += 'memory used: %.02f of %s Gb' % (mem_usage[0]/1000, TOTAL_SYSTEM_MEMORY/1000000000)

                log(msg)
            return loss

        if self.iterations:
            max_iter = int(self.iterations)
            while n_iter[0] <= max_iter:
                optimizer.step(closure)

        end = timer()
        duration = "%.02f seconds" % float(end - start)
        log("duration: %s" % duration)

        return opt_tensor


    def _prepare_engine(self):
        return utils.get_vgg(self.engine)


    def _prepare_content(self):

        content_tensor = utils.image_to_tensor(self.content.image, DO_CUDA, resize=self.content.scale,
                                               colorspace=self.content.colorspace)
        return content_tensor

    def prepare_opt(self, clone=None):
        if clone:
            if self.content:
                tensor = utils.image_to_tensor(clone, DO_CUDA, resize=self.content.scale, colorspace=self.content.colorspace)
            else:
                tensor = utils.image_to_tensor(clone, DO_CUDA, colorspace=self.opt_colorspace)

            opt_tensor = Variable(tensor.data.clone(), requires_grad=True)

        else:
            o_width = self.opt_x
            o_height = self.opt_y
            opt_image = Image.new("RGB", (o_width, o_height), 255)
            random_grid = map(lambda x: (
                    int(random.random() * 256),
                    int(random.random() * 256),
                    int(random.random() * 256)
                ), [0] * o_width * o_height)

            random_grid = list(random_grid)

            opt_image.putdata(random_grid)
            opt_tensor = utils.PIL_to_tensor(opt_image, DO_CUDA)
            opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)

        return opt_tensor
