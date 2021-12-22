import sys
import random
import traceback
import uuid
import os
import subprocess
from timeit import default_timer as timer

# import memory_profiler
import OpenImageIO as oiio
import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
from torch import optim

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


class StyleImager(object):

    def __init__(self, style_layers=None, style_image=None, content_image=None, style_importance_mask=None, grad_mask=None,
                 frame=0, render_out=None, denoise=False):
        self.denoise = denoise
        self.style_importance_mask = style_importance_mask
        self.grad_mask = grad_mask
        self.iterations = 500
        self.log_iterations = 100
        self.style_image = style_image
        self.content_image = content_image
        self.style_image_2 = None
        self.style_image_3 = None
        self.style_image_4 = None
        self.style_image_5 = None
        self.style_image_6 = None
        # self.resize_content = 1.0
        # self.resize_style = 1.0
        # self.content_colorspace = 'srgb_texture'
        # self.style_colorspace = 'srgb_texture'
        self.random_style = False
        self.output_dir = None
        self.engine = 'gpu'
        self.from_content = True
        self.unsafe = False
        self.progressive = False
        self.max_loss = None
        self.lr = 1
        self.loss_graph = ([], [])
        self.style_layers = style_layers
        self.style_layers_2 = None
        self.style_layers_3 = None
        self.style_layers_4 = None
        self.style_layers_5 = None
        self.style_layers_6 = None
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
        self.init_cuda()

    def init_cuda(self) -> None:
        self.cuda_device = utils.get_cuda_device()
        # log('cuda device:', self.cuda_device)

    def send_to_farm(self, frames: str) -> None:
        nfm = NstFarm()
        nfm.from_content = self.from_content
        nfm.style = self.style_image
        nfm.content = self.content_image
        nfm.out = self.out
        nfm.engine = self.engine
        nfm.frames = frames
        nfm.slayers = [x for x in self.style_layers]
        nfm.sweights = [self.style_layers[x]['weight'] for x in self.style_layers]
        nfm.send_to_farm()

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

        if not self.raw_weights:
            utils.normalise_weights(self.style_layers)

        loss_layers = []
        loss_fns = []
        weights = []
        targets = []
        masks = []

        if self.style_layers:
            style_layer_names = [x for x in self.style_layers]
            style_layer_weights = [self.style_layers[x]['weight'] for x in self.style_layers]

        if self.style_layers_2:
            style_layer_names_2 = [x for x in self.style_layers_2]
            style_layer_weights_2 = [self.style_layers_2[x]['weight'] for x in self.style_layers_2]

        if self.style_layers_3:
            style_layer_names_3 = [x for x in self.style_layers_3]
            style_layer_weights_3 = [self.style_layers_3[x]['weight'] for x in self.style_layers_3]

        if self.style_layers_4:
            style_layer_names_4 = [x for x in self.style_layers_4]
            style_layer_weights_4 = [self.style_layers_4[x]['weight'] for x in self.style_layers_4]

        if self.style_layers_5:
            style_layer_names_5 = [x for x in self.style_layers_5]
            style_layer_weights_5 = [self.style_layers_5[x]['weight'] for x in self.style_layers_5]

        if self.style_layers_6:
            style_layer_names_6 = [x for x in self.style_layers_6]
            style_layer_weights_6 = [self.style_layers_6[x]['weight'] for x in self.style_layers_6]

        if self.content_image:
            content_layers = self.content_layers
            content_masks = [torch.Tensor(0)]
            masks += content_masks
            loss_layers += content_layers
            # content_loss_fns = [entities.MSELoss()] # not using mask, but need to handle extra arg...
            content_loss_fns = [entities.MSELoss()] * len(content_layers)
            loss_fns += content_loss_fns
            content_weights = self.content_weights
            weights += content_weights
            content_tensor = self._prepare_content()
            content_activations = []
            for x in vgg(content_tensor, content_layers):
                content_activations.append(x)

            content_targets = [A.detach() for A in content_activations]
            targets += content_targets

        # if self.content_image_pil:
        #
        #     content_layers = self.content_layers
        #     content_masks = [torch.Tensor(0)]
        #     masks += content_masks
        #     loss_layers += content_layers
        #     # content_loss_fns = [entities.MSELoss()] # not using mask, but need to handle extra arg...
        #     content_loss_fns = [entities.MSELoss()] * len(content_layers)
        #     loss_fns += content_loss_fns
        #     content_weights = self.content_weights
        #     weights += content_weights
        #     content_tensor = prepare_content(self._content_image, self.content_scale, self.content_colorspace)
        #     content_activations = []
        #     for x in vgg(content_tensor, content_layers):
        #         content_activations.append(x)
        #     # content_activations = vgg(content_tensor, content_layers)
        #
        #     content_targets = [A.detach() for A in content_activations]
        #     targets += content_targets

        if self.optimisation_image:
            opt_tensor = self.prepare_opt(clone=self.optimisation_image)
        elif self.from_content:
            opt_tensor = self.prepare_opt(clone=self._content_image)
        else:
            opt_tensor = self.prepare_opt()

        # how would we handle multiple style images?
        if self.style_image:
            loss_layers += style_layer_names
            style_tensor = utils.image_to_tensor(self.style_image, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations = []
            for x in vgg(style_tensor, style_layer_names):
                style_activations.append(x)

            style_loss_fns = [entities.GramMSELoss()] * len(style_layer_names)
            loss_fns += style_loss_fns
            weights += style_layer_weights
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
            targets += style_targets

        if self.style_image_2:
            loss_layers += style_layer_names_2
            style_tensor_2 = utils.image_to_tensor(self.style_image_2, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations_2 = []
            for x in vgg(style_tensor_2, style_layer_names_2):
                style_activations_2.append(x)

            style_loss_fns_2 = [entities.GramMSELoss()] * len(style_layer_names_2)
            loss_fns += style_loss_fns_2
            weights += style_layer_weights_2
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations_2]
            targets += style_targets

        if self.style_image_3:
            loss_layers += style_layer_names_3
            style_tensor_3 = utils.image_to_tensor(self.style_image_3, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations_3 = []
            for x in vgg(style_tensor_3, style_layer_names_3):
                style_activations_3.append(x)

            style_loss_fns_3 = [entities.GramMSELoss()] * len(style_layer_names_3)
            loss_fns += style_loss_fns_3
            weights += style_layer_weights_3
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations_3]
            targets += style_targets

        if self.style_image_4:
            loss_layers += style_layer_names_4
            style_tensor_4 = utils.image_to_tensor(self.style_image_4, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations_4 = []
            for x in vgg(style_tensor_4, style_layer_names_4):
                style_activations_4.append(x)

            style_loss_fns_4 = [entities.GramMSELoss()] * len(style_layer_names_4)
            loss_fns += style_loss_fns_4
            weights += style_layer_weights_4
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations_4]
            targets += style_targets

        if self.style_image_5:
            loss_layers += style_layer_names_5
            style_tensor_5 = utils.image_to_tensor(self.style_image_5, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations_5 = []
            for x in vgg(style_tensor_5, style_layer_names_5):
                style_activations_5.append(x)

            style_loss_fns_5 = [entities.GramMSELoss()] * len(style_layer_names_5)
            loss_fns += style_loss_fns_5
            weights += style_layer_weights_5
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations_5]
            targets += style_targets

        if self.style_image_6:
            loss_layers += style_layer_names_6
            style_tensor_6 = utils.image_to_tensor(self.style_image_6, DO_CUDA, resize=self.style_scale, colorspace=self.style_colorspace)

            style_activations_6 = []
            for x in vgg(style_tensor_6, style_layer_names_6):
                style_activations_6.append(x)

            style_loss_fns_6 = [entities.GramMSELoss()] * len(style_layer_names_6)
            loss_fns += style_loss_fns_6
            weights += style_layer_weights_6
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations_6]
            targets += style_targets

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

        for sl in style_layer_names:
            has_mask = True if 'mask' in self.style_layers[sl] else False

            if not has_mask:
                layer_masks.append(None)
                continue

        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)
        layer_masks.append(None)

            # mask_file = self.style_layers[sl]['mask']
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

        cuda_device = self.cuda_device

        def closure():
            output_tensors = vgg(opt_tensor, loss_layers)
            layer_gradients = []

            loss = torch.zeros(1, requires_grad=False).to(torch.device(cuda_device))

            for counter, tensor in enumerate(output_tensors):
                optimizer.zero_grad()
                layer_loss = loss_fns[counter](tensor, targets[counter])
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

            nice_loss = '{:,.0f}'.format(loss.item())
            if self.progressive:
                output_render = self.output_dir + '/render.%04d.png' % n_iter[0]
                # print('doing render:', output_render)
                # print(os.path.isdir(self.output_dir))
                utils.render_image(opt_tensor, output_render, 'loss: %s\niteration: %s' % (nice_loss, n_iter[0]))
                # print('finished render')

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % show_iter == (show_iter - 1):

                # to do: ideally find the actual version where this changed:
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
            # log('')
            while n_iter[0] <= max_iter:
                optimizer.step(closure)
            # log('')

        if self.max_loss:
            while current_loss[0] > int(self.max_loss):
                optimizer.step(closure)

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