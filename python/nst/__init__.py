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


"""



import traceback
import uuid
import random
import os
import subprocess
from timeit import default_timer as timer
import shutil

import memory_profiler

import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
from torch import optim

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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


def render_image(tensor, filepath, text=None):
    out_img = utils.postp(tensor.data[0].cpu().squeeze())

    if text:
        draw = ImageDraw.Draw(out_img)

        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 30)
        draw.text((0, 0), text, (255, 255, 255), font=font)

    out_img.save(filepath)


def get_full_path(filename):
    if not filename.startswith('/'):
        return os.getcwd() + '/' + filename
    return filename


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
    style = get_full_path(opts.style)
    content = get_full_path(opts.content)
    output_dir = get_full_path(opts.output_dir)
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
    style_image = Image.open(style)

    if random_style:
        style_image = random_crop_image(style_image)
        style_image.save('%s/style.png' % output_dir)

    content_image = Image.open(content)

    style_tensor = utils.image_to_tensor(style_image, do_cuda)
    # style_tensor_1 = utils.image_to_tensor(style_image_1, do_cuda)
    # style_tensor_2 = utils.image_to_tensor(style_image_2, do_cuda)
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
    # style_targets_1 = [entities.GramMatrix()(A).detach() for A in vgg(style_tensor_1, style_layers)]
    # style_targets_2 = [entities.GramMatrix()(A).detach() for A in vgg(style_tensor_2, style_layers)]
    content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets
    # targets = (style_targets_1 + style_targets_2)/2 + content_targets

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
                # by checking, and abort if a treshold is met.  Opt out via --unsafe flag.
                smi_mem_used = ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
                used_gpu_memory = float(subprocess.check_output(smi_mem_used).rstrip('\n'))

                percent_gpu_usage = used_gpu_memory / total_gpu_memory * 100
                if percent_gpu_usage > MAX_GPU_RAM_USAGE:
                    raise RuntimeError("Ran out of GPU memory")

        optimizer.zero_grad()

        # The __call__ method on the class seems to actually execute the foward method
        # The args given to vgg.__call__() are passed to vgg.forward()
        # The output is a list of tensors [torch.Tensor].
        output_tensors = vgg(opt_img, loss_layers)

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
            render_image(opt_img, output_render, 'loss: %s\niteration: %s' % (nice_loss, n_iter[0]))

        current_loss[0] = loss.item()
        n_iter[0] += 1

        loss_graph[0].append(n_iter[0])
        loss_graph[1].append(loss.item())

        if n_iter[0] % show_iter == (show_iter - 1):
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

    end = timer()

    pyplot.plot(loss_graph[0], loss_graph[1])
    pyplot.xlabel('iterations')
    pyplot.ylabel('loss')
    loss_graph_filepath = output_dir + '/loss.png'
    pyplot.savefig(loss_graph_filepath)

    duration = "%.02f seconds" % float(end-start)
    log('completed\n')
    log("duration: %s" % duration)

    log_filepath = output_dir + '/log.txt'

    if progressive:
        ffmpeg_cmd = []
        ffmpeg_cmd += ['ffmpeg', '-i', '%s/render.%%04d.png' % temp_dir]
        ffmpeg_cmd += ['-c:v', 'libx264', '-crf', '15', '-y']
        ffmpeg_cmd += ['%s/prog.mp4' % output_dir]
        subprocess.check_output(ffmpeg_cmd)
        shutil.rmtree(temp_dir)

    with open(log_filepath, 'w') as log_file:
        log_file.write(LOG)


def random_crop_image(image):
    """
    Given a PIL.Image, crop it with a bbox of random location and size
    """
    x_size, y_size = image.size
    min_size = min(x_size, y_size)

    bbox_min_ratio = 0.2
    bbox_max_ratio = 0.7

    bbox_min = int(min_size * bbox_min_ratio)
    bbox_max = int(min_size * bbox_max_ratio)

    # bbox_min, bbox_max = 80, 400
    bbox_size = random.randrange(bbox_min, bbox_max)

    # the bbox_size determins where the center can be placed
    x_range = (0+(bbox_size/2), x_size-(bbox_size/2))
    y_range = (0 + (bbox_size / 2), y_size - (bbox_size / 2))

    bbox_ctr_x = random.randrange(x_range[0], x_range[1])
    bbox_ctr_y = random.randrange(y_range[0], y_range[1])

    bbox_left = bbox_ctr_x - (bbox_size/2)
    bbox_upper = bbox_ctr_y - (bbox_size/2)
    bbox_right = bbox_ctr_x + (bbox_size/2)
    bbox_lower = bbox_ctr_y + (bbox_size/2)

    return image.crop((bbox_left, bbox_upper, bbox_right, bbox_lower))



