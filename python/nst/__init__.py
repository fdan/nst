"""
experiments:

* loss for multiple style images
* from a content and output, calculate style
* vary style/content balance across image
* logarithmic graphs
* tiling / scaling of style
* style reconstruction
* content reconstruction
* style atlases


tech things to do:
* optimise a noise image rather than cloning content
* perform content only reconstruction on noise image
* perform style only reconstruction on noise image
* output optimised image at various layers of cnn

"""
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


LOG = ''
MAX_GPU_RAM_USAGE = 90
DO_CUDA = False
TOTAL_GPU_MEMORY = 1
TOTAL_SYSTEM_MEMORY = 1


def log(msg):
    global LOG
    LOG += msg + '\n'
    print msg


def doit(opts):
    try:
        _doit(opts)
    except:
        print traceback.print_exc()
    finally:
        if opts.farm:
            env_cleanup = ['setup-conda-env', '-r']
            subprocess.check_output(env_cleanup)


def _doit(opts):
    start = timer()
    style = utils.get_full_path(opts.style)
    content = utils.get_full_path(opts.content)
    output_dir = utils.get_full_path(opts.output_dir)
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

    vgg = prepare_engine(engine)
    content_tensor, style_tensor, opt_tensor = prepare_images(style, random_style, content, output_dir)

    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    # loss_layers = style_layers

    loss_fns = [entities.GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    # loss_fns = [entities.GramMSELoss()] * len(style_layers)
    if DO_CUDA:
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    # weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # compute optimization targets
    style_targets = [entities.GramMatrix()(A).detach() for A in vgg(style_tensor, style_layers)]
    content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets
    # targets = style_targets

    # run style transfer
    show_iter = 20
    optimizer = optim.LBFGS([opt_tensor])
    n_iter = [0]
    current_loss = [9999999]
    loss_graph = ([], [])

    def closure():
        if engine == 'gpu':
            if not unsafe:
                # pytorch may not be the only process using GPU ram.  Be a good GPU memory citizen
                # by checking, and abort if a treshold is met.  Opt out via --unsafe flag.
                smi_mem_used = ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
                used_gpu_memory = float(subprocess.check_output(smi_mem_used).rstrip('\n'))

                percent_gpu_usage = used_gpu_memory / TOTAL_GPU_MEMORY * 100
                if percent_gpu_usage > MAX_GPU_RAM_USAGE:
                    raise RuntimeError("Ran out of GPU memory")

        optimizer.zero_grad()

        # The __call__ method on the class seems to actually execute the foward method
        # The args given to vgg.__call__() are passed to vgg.forward()
        # The output is a list of tensors [torch.Tensor].
        output_tensors = vgg(opt_tensor, loss_layers)

        layer_losses = []
        for counter, tensor in enumerate(output_tensors):
            w = weights[counter]
            l = loss_fns[counter](tensor, targets[counter])
            layer_losses.append(w * l)

        loss = sum(layer_losses)
        loss.backward()
        nice_loss = '{:,.0f}'.format(loss.item())

        if progressive:
            output_render = temp_dir + '/render.%04d.png' % n_iter[0]
            utils.render_image(opt_tensor, output_render, 'loss: %s\niteration: %s' % (nice_loss, n_iter[0]))

        current_loss[0] = loss.item()
        n_iter[0] += 1

        loss_graph[0].append(n_iter[0])
        loss_graph[1].append(loss.item())

        if n_iter[0] % show_iter == (show_iter - 1):
            max_mem_cached = torch.cuda.max_memory_cached(0) / 1000000
            msg = ''
            msg += 'Iteration: %d, ' % (n_iter[0] + 1)
            msg += 'loss: %s, ' % (nice_loss)

            if DO_CUDA:
                msg += 'memory used: %s of %s' % (max_mem_cached, TOTAL_GPU_MEMORY)
            else:
                mem_usage = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=0.1)
                msg += 'memory used: %.02f of %s Gb' % (mem_usage[0]/1000, TOTAL_SYSTEM_MEMORY/1000000000)

            log(msg)
        return loss

    if iterations:
        max_iter = int(iterations)
        log('')
        while n_iter[0] <= max_iter:
            optimizer.step(closure)
        log('')

    if max_loss:
        while current_loss[0] > int(max_loss):
            optimizer.step(closure)

    output_render = output_dir + '/render.png'
    utils.render_image(opt_tensor, output_render)

    end = timer()

    utils.graph_loss(loss_graph, output_dir)

    duration = "%.02f seconds" % float(end-start)
    log('completed\n')
    log("duration: %s" % duration)
    log_filepath = output_dir + '/log.txt'

    if progressive:
        utils.do_ffmpeg(output_dir, temp_dir)

    with open(log_filepath, 'w') as log_file:
        log_file.write(LOG)


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
            TOTAL_GPU_MEMORY = float(subprocess.check_output(smi_mem_total).rstrip('\n'))
            vgg.cuda()

            log("using cuda\navailable memory: %.0f Gb" % TOTAL_GPU_MEMORY)
        else:
            msg = "gpu mode was requested, but cuda is not available"
            raise RuntimeError(msg)

    elif engine == 'cpu':
        global TOTAL_SYSTEM_MEMORY
        
        DO_CUDA = False
        from psutil import virtual_memory
        TOTAL_SYSTEM_MEMORY = virtual_memory().total
        log("using cpu\navailable memory: %s Gb" % (TOTAL_SYSTEM_MEMORY / 1000000000))

    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg) 
    
    return vgg


def prepare_images(style, random_style, content, output_dir):
    # list of PIL images
    style_image = Image.open(style)

    if random_style:
        style_image = utils.random_crop_image(style_image)
        style_image.save('%s/style.png' % output_dir)

    content_image = Image.open(content)

    style_tensor = utils.image_to_tensor(style_image, DO_CUDA)
    content_tensor = utils.image_to_tensor(content_image, DO_CUDA)

    o_width = content_image.size[0]
    o_height = content_image.size[1]
    opt_image = Image.new("RGB", (o_width, o_height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * o_width * o_height)
    opt_image.putdata(random_grid)
    opt_tensor = utils.image_to_tensor(opt_image, DO_CUDA)
    opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)

    # opt_tensor = Variable(content_tensor.data.clone(), requires_grad=True)

    return content_tensor, style_tensor, opt_tensor
