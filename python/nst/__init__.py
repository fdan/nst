import os
import subprocess
from timeit import default_timer as timer

import memory_profiler

import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
from torch import optim

from PIL import Image

from matplotlib import pyplot

from . import entities
from . import utils


LOG = ''


def log(msg):
    global LOG
    LOG += msg + '\n'
    print msg


def doit(opts):

    start = timer()

    style = opts.style
    content = opts.content
    output = opts.output
    engine = opts.engine
    iterations = opts.iterations
    max_loss = opts.loss

    if iterations and max_loss:
        log("iterations and max_loss canot both be specified")
        return

    if engine == 'gpu':
        if torch.cuda.is_available():
            do_cuda = True
            smi_cmd = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader']
            total_gpu_memory = subprocess.check_output(smi_cmd).rstrip('\n')
            log("using cuda\navailable memory: %s" % total_gpu_memory)
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

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        current_loss[0] = loss.item()
        n_iter[0] += 1
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

    out_img = utils.postp(opt_img.data[0].cpu().squeeze())

    if output:
        out_img.save(output)
    else:
        pyplot.imshow(out_img)

    end = timer()
    duration = "%.02f seconds" % float(end-start)
    log('completed\n')
    log("duration: %s" % duration)

    log_filepath = output.split('.')[0] + '.log'

    with open(log_filepath, 'w') as log_file:
        log_file.write(LOG)
