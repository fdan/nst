import sys
import random
import traceback
import uuid
import os
import json
import subprocess
from timeit import default_timer as timer

# import memory_profiler
import OpenImageIO as oiio
import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from PIL import Image

# import matplotlib
# matplotlib.use('Agg')

from . import entities
from . import utils

from nst_farm.singularity import NstFarm

import numpy as np

# import OpenImageIO as oiio
# from OpenImageIO import ImageBuf, ImageSpec, ROI

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
    global LOG
    LOG += msg + '\n'
    print(msg)


def doit(opts):
    try:
        _doit(opts)
    except:
        print(traceback.print_exc())
    finally:
        if opts.farm:
            env_cleanup = ['setup-conda-env', '-r']
            subprocess.check_output(env_cleanup)


def _doit(opts):
    start = timer()

    style = opts.style
    if style:
        style = utils.get_full_path(style)

    content = opts.content
    if content:
        content = utils.get_full_path(content)

    # user_style_layers = opts.style_layers
    # if user_style_layers:
    #     user_style_layers = [int(x) for x in user_style_layers.split(',')]

    render_name = opts.render_name

    output_dir = utils.get_full_path(opts.output_dir)
    if not output_dir.endswith('/'):
        output_dir += '/'

    temp_dir = '/tmp/nst/%s' % str(uuid.uuid4())[:8:]
    engine = opts.engine
    iterations = opts.iterations
    max_loss = opts.loss
    unsafe = bool(opts.unsafe)
    random_style = bool(opts.random_style)
    progressive = bool(opts.progressive)

    # if this fails, we want an exception:
    os.makedirs(temp_dir)

    # if this fails, dir probably exists already:
    try:
        os.makedirs(output_dir)
    except:
        pass

    log('\nstyle input: %s' % style)
    log('content input: %s' % content)
    log('output dir: %s' % output_dir)
    log('engine: %s' % engine)
    log('iterations: %s' % iterations)
    log('max_loss: %s' % max_loss)
    log('unsafe: %s' % unsafe)
    log('')

    style_imager = StyleImager(style_image=style, content_image=content)
    style_imager.output_dir = output_dir

    if iterations:
        style_imager.iterations = iterations

    opt_tensor = style_imager.generate_tensor()

    output_render_name = render_name or 'render.png'
    output_render = output_dir + output_render_name
    # output_render = output_dir + 'render.png'
    utils.render_image(opt_tensor, output_render)

    end = timer()

    utils.graph_loss(style_imager.loss_graph, output_dir)

    duration = "%.02f seconds" % float(end-start)
    log('completed\n')
    log("duration: %s" % duration)
    log_filepath = output_dir + '/log.txt'

    if progressive:
        utils.do_ffmpeg(output_dir, temp_dir)

    with open(log_filepath, 'w') as log_file:
        log_file.write(LOG)


class StyleMip(object):
    def __init__(self, scale, layers, out_mask=None):
        self.scale = scale
        self.out_mask = out_mask
        self.layers = layers

    def as_json(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        return {self.scale: self.layers}


class Style(object):
    def __init__(self, image, mips, in_mask=None, out_mask=None):
        self.image = image
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.mips = mips


class StyleImager(object):

    def __init__(self, style, content_image=None, frame=0, render_out=None, engine='gpu'):
        self.style = style
        self.iterations = 500
        self.log_iterations = 100
        self.content_image = content_image
        self.random_style = False
        self.output_dir = None
        self.engine = engine
        self.from_content = True
        self.unsafe = False
        self.progressive = False
        self.max_loss = None
        self.lr = 1
        self.loss_graph = ([], [])
        self.opt_x = 512
        self.opt_y = 512
        self.content_layers = ['r42']
        self.content_weights = [1.0]
        self.content_masks = [None]
        self.frame = frame
        self.render_out = render_out
        self.raw_weights = True
        self.optimisation_image = None
        self.output_dir = '%s/output' % os.getcwd()
        self.cuda_device = None
        self.out = None
        self.out_colorspace = 'acescg'
        self.content_scale = 1.0
        self.style_scale = 1.0
        self.content_colorspace = 'acescg'
        self.style_colorspace = 'srgb_texture'

        if engine == 'gpu':
            self.init_cuda()

    def init_cuda(self) -> None:
        self.cuda_device = utils.get_cuda_device()
        # log('cuda device:', self.cuda_device);

    # def send_to_farm(self, frames: str=None) -> None:
    #     nfm = NstFarm()
    #     nfm.from_content = self.from_content
    #     nfm.style = self.style.image
    #     nfm.content = self.content_image
    #     nfm.out = self.out
    #     nfm.engine = self.engine
    #     if frames:
    #         nfm.frames = frames
    #     nfm.slayers = [x for x in self.style_layers] # todo: requires change to nst bin
    #     nfm.sweights = [self.style_layers[x]['weight'] for x in self.style_layers]  # todo: requires change to nst bin
    #     nfm.send_to_farm()

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
        if self.content_image:
            if '####' in self.content_image and frame:
                self._content_image = self.content_image.replace('####', '%04d' % frame)
            else:
                self._content_image = self.content_image

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

        if self.content_image:
            if '####' in self.content_image and frame:
                self._content_image = self.content_image.replace('####', '%04d' % frame)
            else:
                self._content_image = self.content_image

        start = timer()
        vgg = self._prepare_engine()

        loss_layers = []
        loss_fns = []
        weights = []
        targets = []
        masks = []
        scales = []

        if self.content_image:
            content_layers = self.content_layers
            content_masks = [torch.Tensor(0)]
            masks += content_masks
            loss_layers += content_layers
            # content_loss_fns = [entities.MSELoss()] # not using mask, but need to handle extra arg...
            # content_loss_fns = [entities.MSELoss()] * len(content_layers)
            content_loss_fns = [entities.MipMSELoss()] * len(content_layers)
            loss_fns += content_loss_fns
            content_weights = self.content_weights
            weights += content_weights
            content_tensor = self._prepare_content()
            content_activations = []
            for x in vgg(content_tensor, content_layers):
                content_activations.append(x)

            content_targets = [A.detach() for A in content_activations]
            targets += content_targets

            scales += [1.0]

        if self.optimisation_image:
            opt_tensor = self.prepare_opt(clone=self.optimisation_image)
        elif self.from_content:
            opt_tensor = self.prepare_opt(clone=self._content_image)
        else:
            opt_tensor = self.prepare_opt()

        if self.style.image:
            for mip in self.style.mips:

                # todo: in adopting the gaussian pyramid approach, we need to package the data up differently.
                # currently in the closure, we index everything by loss layer, and below we have a loss layer for
                # each mip.
                #
                # we need to change that, such that the style_targets holds gram activations for all mips in a layer.
                # so that's a change to the entity relationsip: instead of mips holding layers, layers hold mips.

                if mip.layers:
                    style_layer_names = [x for x in mip.layers]
                    style_layer_weights = [mip.layers[x] for x in mip.layers]
                    # print(1.0, 'mip scale:', mip.scale, style_layer_names, style_layer_weights)
                    loss_layers += style_layer_names

                    style_tensor = utils.image_to_tensor(self.style.image, DO_CUDA, resize=mip.scale, colorspace=self.style_colorspace)

                    style_activations = []
                    for x in vgg(style_tensor, style_layer_names):
                        style_activations.append(x)

                    # style_loss_fns = [entities.GramMSELoss()] * len(style_layer_names)
                    style_loss_fns = [entities.MipGramMSELoss()] * len(style_layer_names)
                    loss_fns += style_loss_fns
                    weights += style_layer_weights

                    # do not calculate the gram here - we need to know the mip dimensions in the loss fn
                    # style_targets = style_activations
                    style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
                    targets += style_targets

                    scales += [float(mip.scale)]

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
        optimizer = optim.LBFGS([opt_tensor], lr=self.lr, max_iter=int(self.iterations))
        n_iter = [1]
        current_loss = [9999999]

        layer_masks = []

        for cl in self.content_masks:
            if not cl:
                layer_masks.append(None)
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
            # output_tensors = vgg(opt_tensor, loss_layers)
            layer_gradients = []

            if cuda_device:
                loss = torch.zeros(1, requires_grad=False).to(torch.device(cuda_device))
            else:
                loss = torch.zeros(1, requires_grad=False)

            # for counter, tensor in enumerate(output_tensors):
            for counter, layer in enumerate(loss_layers):
                optimizer.zero_grad()

                # # is it a problem that the loss function doesn't receive the opt tensor?  will that affect grad?
                # input_ = F.interpolate(input, scale_factor=scales[counter], mode='bilinear')
                # tensor = vgg(input_, [layer])
                # layer_loss = loss_fns[counter](tensor, targets[counter])

                layer_loss = loss_fns[counter](opt_tensor, targets[counter], scales[counter], layer, vgg)

                # layer_loss = loss_fns[counter](opt_tensor, targets[counter], scales[counter], layer, vgg)
                layer_weight = weights[counter]
                weighted_layer_loss = layer_weight * layer_loss
                weighted_layer_loss.backward(retain_graph=True)

                # don't apply mask for content loss
                if True == True: # silly way to make this run on any layer, witout adjusting indentation, for easy rollback

                    loss += layer_loss

                    # if this style layer has a mask
                    if layer_masks[counter] is not None:

                        layer_mask = layer_masks[counter]
                        b, c, w, h = opt_tensor.grad.size()
                        masked_grad = opt_tensor.grad.clone()

                        # output gradient to disk for first epoch
                        # if n_iter[0] == 3:
                        #     ni = masked_grad.cpu().numpy().transpose()
                        #
                        #     x = ni.shape[1]
                        #     y = ni.shape[0]
                        #     z = ni.shape[2]
                        #
                        #     ni = np.reshape(ni, (x, y, z))
                        #
                        #     buf = ImageBuf(ImageSpec(x, y, z, oiio.FLOAT))
                        #     buf.set_pixels(ROI(), ni.copy())
                        #
                        #     fp = 'grad_seq/%02d_%02d_grad.exr' % (n_iter[0], counter)
                        #     print('writing gradient:', fp)
                        #     buf.write(fp, oiio.FLOAT)
                        #     print('done')

                        # normalise: ensure mean activation remains same
                        # mask_normalisation = (x * y) / layer_mask.sum()
                        # weighted_mask_tensor = torch.div(layer_mask, mask_normalisation)

                        for i in range(0, c):
                            # masked_grad[0][i] *= weighted_mask_tensor
                            masked_grad[0][i] *= layer_mask

                        # if n_iter[0] == 3:
                        #     ni = masked_grad.cpu().numpy().transpose()
                        #
                        #     x = ni.shape[1]
                        #     y = ni.shape[0]
                        #     z = ni.shape[2]
                        #
                        #     ni = np.reshape(ni, (x, y, z))
                        #
                        #     buf = ImageBuf(ImageSpec(x, y, z, oiio.FLOAT))
                        #     buf.set_pixels(ROI(), ni.copy())
                        #
                        #     fp = 'grad_seq/%02d_%02d_grad_masked.exr' % (n_iter[0], counter)
                        #     print('writing gradient:', fp)
                        #     buf.write(fp, oiio.FLOAT)

                        layer_gradients.append(masked_grad)

                    # this style layer does not have a mask
                    else:
                        layer_gradients.append(opt_tensor.grad.clone())

                else:
                    loss += layer_loss
                    layer_gradients.append(opt_tensor.grad.clone())

            b, c, w, h = opt_tensor.grad.size() # not strictly necessary?

            # todo: this assumes the engine is gpu.
            if self.engine == "gpu":
                output_layer_gradient = torch.zeros((b, c, w, h)).detach().to(torch.device(cuda_device))
            else:
                output_layer_gradient = torch.zeros((b, c, w, h)).detach()
            for lg in layer_gradients:
                output_layer_gradient += lg

            # average?  why?
            opt_tensor.grad = output_layer_gradient

            # output opt_tensor as an image seq here

            nice_loss = '{:,.0f}'.format(loss.item())
            if self.progressive:
                output_render = self.output_dir + '/render.%04d.png' % n_iter[0]
                utils.render_image(opt_tensor, output_render, 'loss: %s\niteration: %s' % (nice_loss, n_iter[0]))

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
                # if DO_CUDA:
                #     msg += 'memory used: %s of %s' % (max_mem_cached, TOTAL_GPU_MEMORY)
                # else:
                #     mem_usage = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=0.1)
                #     msg += 'memory used: %.02f of %s Gb' % (mem_usage[0]/1000, TOTAL_SYSTEM_MEMORY/1000000000)

                log(msg)
            return loss

        if self.iterations:
            max_iter = int(self.iterations)
            while n_iter[0] <= max_iter:
                optimizer.step(closure)

        # if we did per-iteration gradient outputs, do an ffmpeg on the sequence:

        # if we did per-iteration opt outputs, do an ffmpeg on the sequence:

        end = timer()
        duration = "%.02f seconds" % float(end - start)
        log("duration: %s" % duration)

        return opt_tensor


    def _prepare_engine(self):

        vgg = entities.VGG()

        model_filepath = os.getenv('NST_VGG_MODEL')

        for param in vgg.parameters():
            param.requires_grad = False

        global DO_CUDA

        if self.engine == 'gpu':
            vgg.cuda()

            vgg.load_state_dict(torch.load(model_filepath))
            global TOTAL_GPU_MEMORY

            if torch.cuda.is_available():
                DO_CUDA = True
                smi_mem_total = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']
                TOTAL_GPU_MEMORY = float(subprocess.check_output(smi_mem_total).rstrip(b'\n'))
                vgg.cuda()
                # log("using cuda\navailable memory: %.0f Gb" % TOTAL_GPU_MEMORY)
            else:
                msg = "gpu mode was requested, but cuda is not available"
                raise RuntimeError(msg)

        elif self.engine == 'cpu':
            vgg.load_state_dict(torch.load(model_filepath))
            global TOTAL_SYSTEM_MEMORY

            DO_CUDA = False
            from psutil import virtual_memory
            TOTAL_SYSTEM_MEMORY = virtual_memory().total
            # log("using cpu\navailable memory: %s Gb" % (TOTAL_SYSTEM_MEMORY / 1000000000))

        else:
            msg = "invalid arg for engine: valid options are cpu, gpu"
            raise RuntimeError(msg)

        return vgg

    def _prepare_content(self):
        content_tensor = utils.image_to_tensor(self._content_image, DO_CUDA, resize=self.content_scale,
                                               colorspace=self.content_colorspace)
        return content_tensor

    def prepare_opt(self, clone=None):
        if clone:
            content_tensor = utils.image_to_tensor(clone, DO_CUDA, resize=self.content_scale, colorspace=self.content_colorspace)
            opt_tensor = Variable(content_tensor.data.clone(), requires_grad=True)

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
