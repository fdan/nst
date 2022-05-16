"""
given three images, A, B and C, optimise A such that it looks like a
combination of B and C
"""
import random
import os
import subprocess
from timeit import default_timer as timer

import OpenImageIO as oiio
from torch.autograd import Variable # deprecated - use Tensor
from torch import optim
import torch.nn as nn

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


class DeepSmooth(object):

    def __init__(self):
        self.current_frame = ''
        self.prior_key = ''
        self.next_key = ''
        self.prior_key_weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.next_key_weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        # self.fore_mvec = ''
        # self.back_mvec = ''
        self.output = ''
        self.iterations = 500
        self.log_iterations = 100
        self.resize = 1.0
        self.content_colorspace = 'srgb_texture'
        self.engine = 'gpu'
        self.content_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.out_colorspace = 'acescg'
        self.lr = 1
        self.init_cuda()

    def init_cuda(self) -> None:
        self.cuda_device = utils.get_cuda_device()

    def write_exr(self) -> None:
        t_ = self.output.split('/')
        t_.pop()
        d_ = ('/'.join(t_))
        try:
            os.makedirs(d_)
        except:
            pass

        tensor = self.generate_tensor()
        buf = utils.tensor_to_buf(tensor)
        buf.write(self.output, oiio.FLOAT)

    def generate_tensor(self) -> torch.Tensor:
        start = timer()
        vgg = prepare_engine(self.engine)

        loss_layers = []
        loss_fns = []
        weights = []
        targets = []

        # current_frame
        # curret_frame_layers = self.content_layers
        # loss_layers += curret_frame_layers
        # current_frame_loss_fns = [nn.MSELoss()] * len(curret_frame_layers)
        # loss_fns += current_frame_loss_fns
        # current_frame_weight = [1.0]
        # weights += current_frame_weight
        # current_frame_tensor = prepare_image(self.current_frame, self.resize, self.content_colorspace)
        # current_frame_activations = []
        #
        # for x in vgg(current_frame_tensor, curret_frame_layers):
        #     current_frame_activations.append(x)
        #
        # current_frame_targets = [A.detach() for A in current_frame_activations]
        # targets += current_frame_targets

        # prior key
        prior_key_layers = self.content_layers
        loss_layers += prior_key_layers
        prior_key_loss_fns = [nn.MSELoss()] * len(prior_key_layers)
        loss_fns += prior_key_loss_fns
        weights += self.prior_key_weights
        prior_key_tensor = prepare_image(self.prior_key, self.resize, self.content_colorspace)
        prior_key_activations = []

        for x in vgg(prior_key_tensor, prior_key_layers):
            prior_key_activations.append(x)

        prior_key_targets = [A.detach() for A in prior_key_activations]
        targets += prior_key_targets

        # next key
        next_key_layers = self.content_layers
        loss_layers += next_key_layers
        next_key_loss_fns = [nn.MSELoss()] * len(next_key_layers)
        loss_fns += next_key_loss_fns
        weights += self.next_key_weights
        next_key_tensor = prepare_image(self.next_key, self.resize, self.content_colorspace)
        next_key_activations = []

        for x in vgg(next_key_tensor, next_key_layers):
            next_key_activations.append(x)

        next_key_targets = [A.detach() for A in next_key_activations]
        targets += next_key_targets

        ###

        opt_tensor = prepare_opt(self.current_frame, self.resize, self.content_colorspace)

        if DO_CUDA:
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

        show_iter = self.log_iterations
        optimizer = optim.LBFGS([opt_tensor], lr=self.lr, max_iter=int(self.iterations))
        n_iter = [1]
        current_loss = [9999999]

        def closure():
            optimizer.zero_grad()
            output_tensors = vgg(opt_tensor, loss_layers)
            layer_losses = []

            for counter, tensor in enumerate(output_tensors):
                w = weights[counter]
                l = loss_fns[counter](tensor, targets[counter])
                weighted_loss = w * l
                layer_losses.append(weighted_loss)

            loss = sum(layer_losses)
            loss.backward()

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % show_iter == (show_iter - 1):

                msg = ''
                msg += 'Iteration: %d, ' % (n_iter[0])
                msg += 'loss: %s, ' % (nice_loss)
                print(msg)

            return loss

        if self.iterations:
            max_iter = int(self.iterations)
            while n_iter[0] <= max_iter:
                optimizer.step(closure)

        end = timer()
        duration = "%.02f seconds" % float(end - start)
        print("duration: %s" % duration)

        return opt_tensor


def prepare_engine(engine: str) -> entities.VGG:
    vgg = entities.VGG()
    model_filepath = os.getenv('NST_VGG_MODEL')

    for param in vgg.parameters():
        param.requires_grad = False

    global DO_CUDA

    if engine == 'gpu':
        vgg.cuda()

        vgg.load_state_dict(torch.load(model_filepath))
        global TOTAL_GPU_MEMORY

        if torch.cuda.is_available():
            DO_CUDA = True
            smi_mem_total = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']
            TOTAL_GPU_MEMORY = float(subprocess.check_output(smi_mem_total).rstrip(b'\n'))
            vgg.cuda()
            print("using cuda\navailable memory: %.0f Gb" % TOTAL_GPU_MEMORY)
        else:
            msg = "gpu mode was requested, but cuda is not available"
            raise RuntimeError(msg)

    elif engine == 'cpu':
        vgg.load_state_dict(torch.load(model_filepath))
        global TOTAL_SYSTEM_MEMORY

        DO_CUDA = False
        from psutil import virtual_memory
        TOTAL_SYSTEM_MEMORY = virtual_memory().total

    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg)

    return vgg


def prepare_image(image: str, rescale: float, colorspace: str) -> torch.Tensor:
    content_tensor = utils.image_to_tensor(image, DO_CUDA, resize=rescale, colorspace=colorspace)
    return content_tensor


def prepare_opt(clone: str, scale: float, colorspace: str) -> torch.Tensor:
    if clone:
        content_tensor = utils.image_to_tensor(clone, DO_CUDA, resize=scale, colorspace=colorspace)
        opt_tensor = Variable(content_tensor.data.clone(), requires_grad=True)
        return opt_tensor
