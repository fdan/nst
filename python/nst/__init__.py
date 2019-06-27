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


tech things done:
* optimise a noise image rather than cloning content
* perform content only reconstruction on noise image

tech things to do:
* perform style only reconstruction on noise image
* output optimised image at various layers of cnn



some notes and discoveries

1.  The magic-number weights Gatys provides, are selected for cases where the
optimisation image is a duplicate of the content image.  If you want to perform
gradient descent on a random noise image, you will need to significantly increase
the "strength" of the content weight.

2.  Running for 5000 iterations seems to give quite different results for a nosie
image.  When optimising the content image, after a very high number of iterations
the image begins to degrade significantly.  The same is not true of a randm noise
image.

3.  I assume the reason for using a clone of the content image as the optimisation
image is performance - there is less transformation required, it's a primed image.

4.  Playing with the manually specified weights is a pretty interesting way to
significantly affect the outcome.  However a major pain is that it's not easy to
affect the style/content ratio overall, because there's a list of 5 style weights.
Need to find out if the 1.0 content weight can go above 1.0 (I doubt it), because
if so it would be a very easy way to say "more or less content".

This is definitely something you would want to expose to the artist as a control.
Be good to do some wedge tests with these values changing over time.
Also worth noting the magic numbers given are all proportinate to the size / rank
of each layer in dimensions.  So the user input should probably be tied to this
still.

5.  While I suspect playing with the style weights could have the same effect,
it would be nice to be able to output a snapshot of the optimised image at each
layer.  This would presumably have to be done in the forward function of the VGG
class?







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

    style = opts.style
    if style:
        style = utils.get_full_path(style)

    content = opts.content
    if content:
        content = utils.get_full_path(content)

    user_style_layers = opts.style_layers
    if user_style_layers:
        user_style_layers = [int(x) for x in user_style_layers.split(',')]

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

    vgg = prepare_engine(engine)

    loss_layers = []
    loss_fns = []
    weights = []
    targets = []

    if style:
        potential_style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

        if user_style_layers:
            style_layers = [potential_style_layers[x-1] for x in user_style_layers]
        else:
            style_layers = potential_style_layers

        loss_layers += style_layers
        style_loss_fns = [entities.GramMSELoss()] * len(style_layers)
        loss_fns += style_loss_fns
        style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        weights += style_weights
        style_tensor = prepare_style(style, random_style, output_dir)
        style_targets = [entities.GramMatrix()(A).detach() for A in vgg(style_tensor, style_layers)]
        targets += style_targets

    if content:
        content_layers = ['r42']
        loss_layers += content_layers
        content_loss_fns = [nn.MSELoss()]
        loss_fns += content_loss_fns
        content_weights = [1.0]
        weights += content_weights
        content_tensor = prepare_content(content)
        content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
        targets += content_targets

    # use a clone of the content image as the optimisation image
    if style and content:
        opt_tensor = prepare_opt(clone=content)

    # use a random noise image as the optimisation image
    else:
        # use the x and y dimensions of the content image
        if content:
            content_image = Image.open(content)
            opt_tensor = prepare_opt(width=content_image.size[0], height=content_image.size[1])
        # use default x and y dimensions
        if style:
            opt_tensor = prepare_opt(height=743, width=1356)
        else:
            raise Exception("Style, content or both must be specified")

    if DO_CUDA:
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    show_iter = 20
    optimizer = optim.LBFGS([opt_tensor], max_iter=int(iterations))
    n_iter = [1]
    current_loss = [9999999]
    loss_graph = ([], [])
    layer_tensors = [0]

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

        # at this point, opt_tensor has not been changed...

        layer_tensors[0] = []

        layer_losses = []

        for counter, tensor in enumerate(output_tensors):
            # this is where you would output the image for each layer btw...
            # layer_image = utils.postp(tensor)
            # layer_image.save()
            # layer_output_path = output_dir + ''
            # layer_tensors[0].append(tensor)
            # print ''
            # print 'opt_tensor', opt_tensor.size(), opt_tensor.dim()
            # print counter, 'output_tensor', tensor.size(), tensor.dim()

            # if n_iter[0] == int(iterations)-1:
            #     layer_output = output_dir + 'layer_%s.png' % counter
            #     print 'rendering'
            #     utils.render_image(opt_tensor, layer_output, 'layer: %s' % counter)

            w = weights[counter]
            # print 1, w
            l = loss_fns[counter](tensor, targets[counter])
            # print 2, l
            # l is a tensor, and here we just multiply it by a float:
            weighted_loss = w * l
            # to make weight variable, we should use an image instead.
            layer_losses.append(weighted_loss)

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
            msg += 'Iteration: %d, ' % (n_iter[0])
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

    # print len(layer_tensors[0])
    # print layer_tensors[0]
    # return
    # for layer in layer_tensors[0]:
    #     # print dir(layer)
    #     layer_ind = layer_tensors[0].index(layer)
    #     print 'layer %s size: %s dim: %s' % (layer_ind, layer.size(), layer.dim())
    #     output_layer = output_dir + 'layer_%s.png' % layer_tensors[0].index(layer)
    #     print 'rendering:', output_layer
    #     utils.render_image(layer, output_layer)

    output_render_name = render_name or 'render.png'
    output_render = output_dir + output_render_name
    print 2, output_render
    # output_render = output_dir + 'render.png'
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
        opt_image.putdata(random_grid)
        opt_tensor = utils.image_to_tensor(opt_image, DO_CUDA)
        opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)

    return opt_tensor

