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
from matplotlib import pyplot

from . import entities
from . import utils


LOG = ''
MAX_GPU_RAM_USAGE = 90


def log(msg):
    global LOG
    LOG += msg + '\n'
    print msg


def render_image(tensor, filepath):
    out_img = utils.postp(tensor.data[0].cpu().squeeze())
    out_img.save(filepath)


def doit(opts):

    start = timer()

    style = opts.style
    content = opts.content
    output_dir = opts.output_dir
    engine = opts.engine
    iterations = opts.iterations
    max_loss = opts.loss
    unsafe = opts.unsafe

    try:
        os.makedirs(output_dir)
    except:
        pass

    log('style input: %s' % style)
    log('content input: %s' % content)
    log('output dir: %s' % output_dir)
    log('engine: %s' % engine)
    log('iterations: %s' % iterations)
    log('max_loss: %s' % max_loss)

    if unsafe:
        log('unsafe: %s heroes explore to give us hope' % unsafe)
    else:
        log('unsafe: %s cutting edge is for people who want to bleed' % unsafe)

    log('')

    if engine == 'gpu':
        if torch.cuda.is_available():
            do_cuda = True
            smi_mem_total = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']
            total_gpu_memory = float(subprocess.check_output(smi_mem_total).rstrip('\n'))

            log("using cuda\navailable memory: %.0f Gb" % total_gpu_memory)
        else:
            msg = "gpu mode was requested, but cuda is not available"
            raise RuntimeError(msg)

    elif engine == 'cpu':
        do_cuda = False
        from psutil import virtual_memory
        mem = virtual_memory()
        # mem.total
        log("using cpu\navailable memory: %s Gb" % (mem.total / 1000000000))

    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg)

    # get network
    vgg = entities.VGG()

    model_filepath = os.getenv('NST_VGG_MODEL')
    vgg.load_state_dict(torch.load(model_filepath))

    for param in vgg.parameters():
        param.requires_grad = False

    if do_cuda:
        vgg.cuda()

    # list of PIL images
    input_images = [Image.open(style), Image.open(content)]

    style_image = Image.open(style)
    content_image = Image.open(content)

    style_tensor = utils.image_to_tensor(style_image, do_cuda)
    content_tensor = utils.image_to_tensor(content_image, do_cuda)

    # variable is dperecated
    opt_img = Variable(content_tensor.data.clone(), requires_grad=True)

    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [entities.GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    if do_cuda:
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    # compute optimization targets
    style_targets = [entities.GramMatrix()(A).detach() for A in vgg(style_tensor, style_layers)]
    content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets

    # run style transfer

    show_iter = 20
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]
    current_loss = [9999999]

    loss_graph = ([], [])

    def closure():

        if engine == 'gpu':

            if not unsafe:
                # pytorch may not be the only process using GPU ram.  Be a good GPU memory citizen
                # by checking, and abort if a treshold is met.
                # Opt out by running with --safe False.
                smi_mem_used = ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
                used_gpu_memory = float(subprocess.check_output(smi_mem_used).rstrip('\n'))

                percent_gpu_usage = used_gpu_memory / total_gpu_memory * 100
                if percent_gpu_usage > MAX_GPU_RAM_USAGE:
                    raise RuntimeError("Ran out of GPU memory")


        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        current_loss[0] = loss.item()
        n_iter[0] += 1

        loss_graph[0].append(n_iter[0])
        loss_graph[1].append(loss.item())

        if n_iter[0] % show_iter == (show_iter - 1):
            nice_loss = '{:,.0f}'.format(loss.item())
            max_mem_cached = torch.cuda.max_memory_cached(0) / 1000000
            msg = ''
            msg += 'Iteration: %d, ' % (n_iter[0] + 1)
            msg += 'loss: %s, ' % (nice_loss)

            if do_cuda:
                msg += 'memory used: %s of %s' % (max_mem_cached, total_gpu_memory)
            else:
                mem_usage = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=0.1)
                msg += 'memory used: %.02f of %s Gb' % (mem_usage[0]/1000, mem.total/1000000000)

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
    render_image(opt_img, output_render)

    pyplot.plot(loss_graph[0], loss_graph[1])
    pyplot.xlabel('iterations')
    pyplot.ylabel('loss')
    loss_graph_filepath = output_dir + '/loss.png'
    pyplot.savefig(loss_graph_filepath)

    end = timer()
    duration = "%.02f seconds" % float(end-start)
    log('completed\n')
    log("duration: %s" % duration)

    log_filepath = output_dir + '/log.txt'

    with open(log_filepath, 'w') as log_file:
        log_file.write(LOG)
