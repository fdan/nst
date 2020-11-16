import sys
import random
import traceback
import uuid
import os
import subprocess
from timeit import default_timer as timer

import memory_profiler

import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
from torch import optim

from PIL import Image

import matplotlib
matplotlib.use('Agg')

from . import entities
from . import utils

import numpy as np

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ROI

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

    def __init__(self, style_layers, style_image=None, content_image=None, style_importance_mask=None, grad_mask=None,
                 frame=0, render_out=None, denoise=False):
        self.denoise = denoise
        self.style_imprtance_mask = style_importance_mask
        self.grad_mask = grad_mask
        self.iterations = 500
        self.log_iterations = 20
        self.style_image = style_image
        self.content_image = content_image
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
        self.content_layers = ['r42']
        self.content_weights = [1.0]
        self.frame = frame
        self.render_out = render_out

        utils.normalise_weights(self.style_layers)

        if self.content_image:
            self.content_image_pil = Image.open(self.content_image)

    def generate_image(self):
        tensor = self.generate_tensor()
        return utils.tensor_to_image(tensor)

    def render_to_disk(self):
        img = self.generate_image()
        img_path = self.render_out

        t_ = img_path.split('/')
        t_.pop()
        d_ = ('/'.join(t_))
        if not os.path.isdir(d_):
            os.makedirs(d_)

        img.save(img_path)

    def generate_tensor(self):
        start = timer()

        vgg = prepare_engine(self.engine)

        loss_layers = []
        loss_fns = []
        weights = []
        targets = []
        masks = []

        style_layer_names = [x for x in self.style_layers]
        style_layer_weights = [self.style_layers[x]['weight'] for x in self.style_layers]

        if self.content_image_pil:

            content_layers = self.content_layers
            content_masks = [torch.Tensor(0)]
            masks += content_masks
            loss_layers += content_layers
            content_loss_fns = [entities.MSELoss()] # not using mask, but need to handle extra arg...
            loss_fns += content_loss_fns
            content_weights = [1.0]
            weights += content_weights
            content_tensor = prepare_content(self.content_image)
            content_activations = vgg(content_tensor, content_layers)
            content_targets = [A.detach() for A in content_activations]
            targets += content_targets

        opt_tensor = prepare_opt(clone=self.content_image)

        if self.style_image:
            loss_layers += style_layer_names
            style_tensor = prepare_style(self.style_image, self.random_style, self.output_dir)
            style_activations = []
            for x in vgg(style_tensor, style_layer_names):
                style_activations.append(x)
            style_loss_fns = [entities.GramMSELoss()] * len(style_layer_names)
            loss_fns += style_loss_fns
            weights += style_layer_weights
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
            targets += style_targets

        if DO_CUDA:
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

        show_iter = self.log_iterations
        optimizer = optim.LBFGS([opt_tensor], lr=self.lr, max_iter=int(self.iterations))
        n_iter = [1]
        current_loss = [9999999]

        layer_masks = []

        for sl in style_layer_names:
            has_mask = True if 'mask' in self.style_layers[sl] else False

            if not has_mask:
                layer_masks.append(None)
                continue

            mask_file = self.style_layers[sl]['mask']

            opt_x, opt_y = opt_tensor.size()[2], opt_tensor.size()[3]
            mask = oiio.ImageBuf(mask_file)

            # oiio axis are flipped:
            scaled_mask = oiio.ImageBufAlgo.resize(mask, roi=oiio.ROI(0, opt_y, 0, opt_x, 0, 1, 0, 3))
            mask_np = scaled_mask.get_pixels()
            x, y, z = mask_np.shape
            mask_np = mask_np[:, :, :1].reshape(x, y)
            mask_tensor = torch.Tensor(mask_np).detach().to(torch.device("cuda:0"))
            layer_masks.append(mask_tensor)

        def closure():
            output_tensors = vgg(opt_tensor, loss_layers)
            layer_gradients = []

            loss = torch.zeros(1, requires_grad=False)

            for counter, tensor in enumerate(output_tensors):
                optimizer.zero_grad()
                layer_loss = loss_fns[counter](tensor, targets[counter])
                layer_weight = weights[counter]
                weighted_layer_loss = layer_weight * layer_loss
                weighted_layer_loss.backward(retain_graph=True)

                # don't apply mask for content loss
                if counter != 0:
                    loss += layer_loss

                    # if this style layer has a mask
                    if layer_masks[counter-1] is not None:
                        layer_mask = layer_masks[counter-1]
                        b, c, w, h = opt_tensor.grad.size()
                        masked_grad = opt_tensor.grad.clone()

                        for i in range(0, c):
                            masked_grad[0][i] *= layer_mask

                        layer_gradients.append(masked_grad)

                    # this style layer does not have a mask
                    else:
                        layer_gradients.append(opt_tensor.grad.clone())

                else:
                    loss += layer_loss
                    layer_gradients.append(opt_tensor.grad.clone())

            b, c, w, h = opt_tensor.grad.size() # not strictly necessary?
            output_layer_gradient = torch.zeros((b, c, w, h)).detach().to(torch.device("cuda:0"))
            for lg in layer_gradients:
                output_layer_gradient += lg

            # average?  why?
            # output_layer_gradient = (layer_gradients[0] + layer_gradients[1]) / 2.0
            opt_tensor.grad = output_layer_gradient

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % show_iter == (show_iter - 1):
                max_mem_cached = torch.cuda.max_memory_cached(0) / 1000000
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
            log('')
            while n_iter[0] <= max_iter:
                optimizer.step(closure)
            log('')

        if self.max_loss:
            while current_loss[0] > int(self.max_loss):
                optimizer.step(closure)

        end = timer()
        duration = "%.02f seconds" % float(end - start)
        log("duration: %s" % duration)

        return opt_tensor


def prepare_engine(engine):
    vgg = entities.VGG()
    model_filepath = os.getenv('NST_VGG_MODEL')
    vgg.load_state_dict(torch.load(model_filepath))

    for param in vgg.parameters():
        param.requires_grad = False

    global DO_CUDA

    if engine == 'gpu':
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

    elif engine == 'cpu':
        global TOTAL_SYSTEM_MEMORY
        
        DO_CUDA = False
        from psutil import virtual_memory
        TOTAL_SYSTEM_MEMORY = virtual_memory().total
        # log("using cpu\navailable memory: %s Gb" % (TOTAL_SYSTEM_MEMORY / 1000000000))

    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg) 
    
    return vgg


def prepare_style(style, random_style, output_dir):
    style_image = Image.open(style)

    if random_style:
        style_image = utils.random_crop_image(style_image)
        style_image.save('%s/style.png' % output_dir)

    style_tensor = utils.image_to_tensor(style_image, DO_CUDA)
    return style_tensor


def prepare_content(content):
    content_image = Image.open(content)
    content_tensor = utils.image_to_tensor(content_image, DO_CUDA)
    return content_tensor


def prepare_opt(clone=None, width=500, height=500):
    if clone:
        content_image = Image.open(clone)
        content_tensor = utils.image_to_tensor(content_image, DO_CUDA)
        opt_tensor = Variable(content_tensor.data.clone(), requires_grad=True)
    else:
        o_width = width
        o_height = height
        opt_image = Image.new("RGB", (o_width, o_height), 255)
        random_grid = map(lambda x: (
                int(random.random() * 256),
                int(random.random() * 256),
                int(random.random() * 256)
            ), [0] * o_width * o_height)

        # handle different map behaviour for python3
        if sys.version[0] == '3':
            random_grid = list(random_grid)

        opt_image.putdata(random_grid)
        opt_tensor = utils.image_to_tensor(opt_image, DO_CUDA)
        opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)

    return opt_tensor

