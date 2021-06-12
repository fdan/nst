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

    def __init__(self, style_image=None, content_image=None, masks={}, style_importance_mask=None, grad_mask=None, frame=0, render_out=None, denoise=False):
        print(1, style_image)
        self.denoise = denoise
        self.masks = masks
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
        # self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        # self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.style_weights = []
        self.content_layers = ['r42']
        self.frame = frame
        self.render_out = render_out

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

        if self.content_image_pil:

            if not self.denoise:
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

            if self.denoise:
                content_layers = self.content_layers
                content_masks = [torch.Tensor(0)]
                masks += content_masks
                loss_layers += content_layers
                content_loss_fns = [entities.MSELoss()] # not using mask, but need to handle extra arg...
                loss_fns += content_loss_fns
                content_weights = [1.0]
                weights += content_weights
                content_targets = []

                for layer in content_layers:
                    layer_size = entities.VGG.layers[layer]['channels']

                    b = 1
                    c = layer_size
                    w = entities.VGG.layers[layer]['x']
                    h = entities.VGG.layers[layer]['x']

                    new_np_array = np.zeros((b, c, w, h))
                    for i in range(0, c):
                        exr = 'cv_075/%s/v003/%04d/%s_v003_%04d_denoised.%04d.exr' % (layer, i+1, layer, i+1, self.frame)
                        buf = oiio.ImageBuf(exr)
                        ni = buf.get_pixels()

                        for j in range(0, w):
                            for k in range(0, h):
                                new_np_array[0][i][j][k] = ni[j][k][0]

                    new_tensor = torch.Tensor(new_np_array)
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    new_tensor = new_tensor.detach().to(device)
                    # content_targets += [new_tensor]
                    content_targets = [new_tensor]

                targets += content_targets

        # use a clone of the content image as the optimisation image
        if self.from_content:
            opt_tensor = prepare_opt(clone=self.content_image)

        # use a random noise image as the optimisation image
        else:
            # use the x and y dimensions of the content image
            if self.content_image:
                content_image = Image.open(self.content_image)
                opt_tensor = prepare_opt(width=content_image.size[0], height=content_image.size[1])
            # use default x and y dimensions
            elif self.style_image:
                opt_tensor = prepare_opt(height=743, width=1356)
            else:
                raise Exception("Style, content or both must be specified")

        if self.style_image:
            loss_layers += self.style_layers

            style_tensor = prepare_style(self.style_image, self.random_style, self.output_dir)

            style_activations = []

            if self.style_imprtance_mask:
                style_importance_masks = []
                style_importance_mask = oiio.ImageBuf(self.style_imprtance_mask)

                # style_acvitations = []

                for sl in self.style_layers:
                    # resize to gram for the layer
                    # x = entities.VGG.layers[sl]['channels']
                    # y = x

                    # oiio axis are flipped:
                    # scaled_style_importance_mask = oiio.ImageBufAlgo.resize(style_importance_mask, roi=oiio.ROI(0, y, 0, x, 0, 1, 0, 3))
                    # style_importance_mask_np = scaled_style_importance_mask.get_pixels()

                    for x in vgg(style_tensor, self.style_layers):
                        b, c, w, h = x.shape

                        # print(style_importance_mask_np.shape)
                        # print(b, c, w, h)

                        # oiio axis are flipped:
                        scaled_style_importance_mask = oiio.ImageBufAlgo.resize(style_importance_mask, roi=oiio.ROI(0, w, 0, h, 0, 1, 0, 3))
                        style_importance_mask_np = scaled_style_importance_mask.get_pixels()
                        style_importance_mask_np = style_importance_mask_np[:, :, :1].reshape(w, h)
                        stye_importance_mask_tensor = torch.Tensor(style_importance_mask_np).detach().to(torch.device("cuda:0"))

                        # normalise the mask by dividing by mean squarea
                        style_importance_mask_mean_square = torch.sqrt(stye_importance_mask_tensor.mean())
                        normalised_style_importance_mask_tensor = torch.div(stye_importance_mask_tensor, style_importance_mask_mean_square)

                        masked_x = x.clone()
                        for i in range(0, c):
                            masked_x[0][i] *= normalised_style_importance_mask_tensor

                        # print(x.shape)
                        # style_acvitations.append(masked_x)
                        style_activations.append(masked_x)


            else:
                # style_activations = []
                for x in vgg(style_tensor, self.style_layers):
                    style_activations.append(x)
                # style_activations = [x for x in vgg(style_tensor, self.style_layers)]

            if self.masks:
                style_loss_fns = [entities.MaskedGramMSELoss()] * len(self.style_layers)
                style_masks = []

                for sl in self.style_layers:

                    # determine the target dimensions of the layer mask and resize
                    opt_x, opt_y = opt_tensor.size()[2], opt_tensor.size()[3]
                    opt_ratio = opt_x / opt_y

                    x = entities.VGG.layers[sl]['x']
                    y = int(x / opt_ratio)

                    mask = oiio.ImageBuf(self.masks[sl])

                    # oiio axis are flipped:
                    scaled_mask = oiio.ImageBufAlgo.resize(mask, roi=oiio.ROI(0, y, 0, x, 0, 1, 0, 3))
                    mask_np = scaled_mask.get_pixels()

                    # get mask tensor
                    x, y, z = mask_np.shape # this is pointless?
                    mask_np = mask_np[:, :, :1].reshape(x, y)
                    mask_tensor = torch.Tensor(mask_np).detach().to(torch.device("cuda:0"))

                    self.style_weights.append(mask_tensor.max())

                    # print(2.1, mask_tensor)
                    # print(2.2, torch.max(mask_tensor))
                    # print(2.3, torch.mean(mask_tensor))

                    # normalise the mask to avoid over stimulation of perceptrons
                    # don't think the mean squareroot thing made sense?  this seems to work better

                    # hardcode a particular weight:
                    mask_normalisation = (x*y) / mask_tensor.sum()
                    weighted_mask_tensor = torch.mul(mask_tensor, mask_normalisation)

                    # mean_square = torch.sqrt(mask_tensor.mean())
                    # mean_square = torch.pow(mask_tensor, 2).mean()
                    # weighted_mask_tensor = torch.div(mask_tensor, mean_square)

                    # todo: resize to a multidimensinal tensor matching vgg layer (?)
                    style_masks.append(weighted_mask_tensor)

                    # don't do anything to mask, pass in as raw data - for debugging only
                    # style_masks.append(mask_tensor)

            else:
                # testing out style_importance_mask with new loss function:
                # style_loss_fns = [entities.GramMSELoss2()] * len(self.style_layers)

                style_loss_fns = [entities.GramMSELoss()] * len(self.style_layers)
                style_masks = [None for x in self.style_layers]

            masks += style_masks
            loss_fns += style_loss_fns
            style_weights = self.style_weights
            weights += style_weights
            style_targets = [entities.GramMatrix()(A).detach() for A in style_activations]
            targets += style_targets

        if DO_CUDA:
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

        show_iter = self.log_iterations
        optimizer = optim.LBFGS([opt_tensor], lr=self.lr, max_iter=int(self.iterations))
        n_iter = [1]
        current_loss = [9999999]
        loss_graph = ([], [])
        layer_tensors = [0]

        def closure():

            # if self.engine == 'gpu':
            #     if not self.unsafe:
            #         # pytorch may not be the only process using GPU ram.  Be a good GPU memory citizen
            #         # by checking, and abort if a treshold is met.  Opt out via --unsafe flag.
            #         smi_mem_used = ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
            #         used_gpu_memory = float(subprocess.check_output(smi_mem_used).rstrip(b'\n'))
            #
            #         percent_gpu_usage = used_gpu_memory / TOTAL_GPU_MEMORY * 100
            #         if percent_gpu_usage > MAX_GPU_RAM_USAGE:
            #             raise RuntimeError("Ran out of GPU memory")

            # clear gradients:
            optimizer.zero_grad()

            output_tensors = vgg(opt_tensor, loss_layers)
            layer_tensors[0] = []
            layer_losses = []

            for counter, tensor in enumerate(output_tensors):
                # this is where you would output the image for each layer...
                # layer_image = utils.postp(tensor)
                # layer_image.save()
                # layer_output_path = output_dir + ''
                # layer_tensors[0].append(tensor)
                # print('')
                # print('opt_tensor', opt_tensor.size(), opt_tensor.dim())
                # print(counter, 'output_tensor', tensor.size(), tensor.dim())
                # if n_iter[0] == int(iterations)-1:
                #     layer_output = output_dir + 'layer_%s.png' % counter
                #     print('rendering')
                #     utils.render_image(opt_tensor, layer_output, 'layer: %s' % counter)

                # l = loss_fns[counter](tensor, targets[counter], masks[counter], style_importance_masks[counter])

                l = loss_fns[counter](tensor, targets[counter], masks[counter])
                if len(weights) > 1:
                    w = weights[counter]
                    weighted_loss = w * l
                    layer_losses.append(weighted_loss)
                else:
                    layer_losses.append(l)

            loss = sum(layer_losses)

            # this is where you could apply a mask to the loss, because it's telling the image
            # what has to change
            # print(opt_tensor.grad)

            # r = opt_tensor[0][0][0][0].tolist()
            # g = opt_tensor[0][1][0][0].tolist()
            # b = opt_tensor[0][2][0][0].tolist()
            # print(1, '|{:16s}|{:16s}|{:16s}|'.format(format(r, '6.3f'), format(g, '6.3f'), format(b, '6.3f')))
            #
            loss.backward()

            # r = opt_tensor.grad[0][0][0][0].tolist()
            # g = opt_tensor.grad[0][1][0][0].tolist()
            # b = opt_tensor.grad[0][2][0][0].tolist()
            # print(2, '|{:16s}|{:16s}|{:16s}|'.format(format(r, '6.3f'), format(g, '6.3f'), format(b, '6.3f')))

            # r = opt_tensor[0][0][0][0].tolist()
            # g = opt_tensor[0][1][0][0].tolist()
            # b = opt_tensor[0][2][0][0].tolist()
            # print(3, '|{:16s}|{:16s}|{:16s}|'.format(format(r, '6.3f'), format(g, '6.3f'), format(b, '6.3f')))
            # print('\n')

            # print(loss, opt_tensor.grad.__class__, opt_tensor.grad.shape)

            # apply a grad mask if given

            if self.grad_mask:
                grad_mask = oiio.ImageBuf(self.grad_mask)
                mask_x = grad_mask.oriented_width
                mask_y = grad_mask.oriented_height

                # oiio axis are flipped:
                scaled_grad_mask = oiio.ImageBufAlgo.resize(grad_mask, roi=oiio.ROI(0, mask_y, 0, mask_x, 0, 1, 0, 3))
                grad_mask_np = scaled_grad_mask.get_pixels()
                x_, y_, z_ = grad_mask_np.shape
                grad_mask_np = grad_mask_np[:, :, :1].reshape(x_, y_)
                grad_mask_tensor = torch.Tensor(grad_mask_np).detach().to(torch.device("cuda:0"))

                grad_b, grad_c, grad_w, grad_h = opt_tensor.grad.size()
                masked_grad = opt_tensor.grad.clone()
                for i in range(0, grad_c):
                    masked_grad[0][i] *= grad_mask_tensor

                opt_tensor.grad = masked_grad

            # if n_iter[0] == max_iter:
            #     print 111, opt_tensor.grad_fn
            #     print 222, loss.grad_fn, loss.grad
            grad_img = utils.tensor_to_image(opt_tensor.grad)
            # print grad_img.__class__;;
            # print dir(grad_img)
            grad_render = '%s/grad.%04d.png' % (self.output_dir, n_iter[0])
            utils.render_image(opt_tensor.grad, grad_render)
            # utils.do_ffmpeg('grad', 'grad', 'grad')

            nice_loss = '{:,.0f}'.format(loss.item())

            if self.progressive:
                output_render = self.output_dir + '/render.%04d.png' % n_iter[0]
                utils.render_image(opt_tensor, output_render, 'loss: %s\niteration: %s' % (nice_loss, n_iter[0]))

            current_loss[0] = loss.item()
            n_iter[0] += 1

            loss_graph[0].append(n_iter[0])
            loss_graph[1].append(loss.item())

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

        # render to disk...disabled for now
        # for layer in layer_tensors[0]:
        #     # print(dir(layer))
        #     layer_ind = layer_tensors[0].index(layer)
            # print('layer %s size: %s dim: %s' % (layer_ind, layer.size(), layer.dim()))
            # output_layer = output_dir + 'layer_%s.png' % layer_tensors[0].index(layer)
            # output_layer = 'opt/layer_%s.png' % layer_tensors[0].index(layer)
            # print('rendering:', output_layer)
            # utils.render_image(layer, output_layer)

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

