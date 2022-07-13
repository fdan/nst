import json
import os

import torch
from torch import optim

from . import guides
from . import vgg
from . import utils
import nst.settings as settings

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False


import torch


class TorchStyle(object):
    def __init__(self, tensor, alpha=torch.zeros(0), target_map=torch.zeros(0)):
        self.tensor = tensor
        self.alpha = alpha
        self.target_map = target_map
        self.scale = 1.0


class Nst(torch.nn.Module):
    def __init__(self):
        super(Nst, self).__init__()
        self.vgg = vgg.VGG()
        self.content = torch.zeros(0)
        self.content_scale = 1.0
        self.styles = []
        self.opt_tensor = torch.zeros(0)
        self.opt_guides = []
        self.optimiser = None
        self.settings = settings.NstSettings()

    def prepare(self):
        b, c, w, h = self.opt_tensor.size()
        self.settings.opt_orig_scale = [w, h]

        if self.settings.cuda:
            self.settings.cuda_device = utils.get_cuda_device()

        for param in self.vgg.parameters():
            param.requires_grad = False

        if self.settings.engine == 'gpu':
            self.vgg.cuda()
            self.settings.cuda = True
            self.vgg.load_state_dict(torch.load(self.settings.model_path))

        elif self.settings.engine == 'cpu':
            self.vgg.load_state_dict(torch.load(self.settings.model_path))
            self.settings.cuda = False

        # handle rescale of tensors here
        if self.settings.scale != 1.0:
            self.content = utils.rescale_tensor(self.content, self.settings.scale)

            self.opt_tensor = utils.rescale_tensor(self.opt_tensor, self.settings.scale, requires_grad=True)

            for style in self.styles:
                i = self.styles.index(style)
                self.styles[i].tensor = utils.rescale_tensor(style.tensor, self.settings.scale)

                if self.styles[i].alpha.numel() != 0:
                    self.styles[i].alpha = utils.rescale_tensor(style.alpha, self.settings.scale)

                if self.styles[i].target_map.numel() != 0:
                    self.styles[i].target_map = utils.rescale_tensor(style.target_map, self.settings.scale)

        if self.settings.optimiser == 'lbfgs':
            print('optimiser is lbfgs')
            self.optimiser = optim.LBFGS([self.opt_tensor], lr=self.settings.learning_rate)
        elif self.settings.optimiser == 'adam':
            print('optimiser is adam')
            self.optimiser = optim.Adam([self.opt_tensor], lr=self.settings.learning_rate)
        else:
            raise Exception("unsupported optimiser:", self.settings.optimiser)

        content_guide = guides.ContentGuide(self.content, self.vgg, self.settings.content_layer,
                                            self.settings.content_layer_weight, self.settings.cuda_device)

        content_guide.prepare()
        self.opt_guides.append(content_guide)

        style_guide = guides.StyleGuide(self.styles, self.vgg, self.settings.style_mips,
                                        self.settings.pyramid_scale_factor, self.settings.style_mip_weights,
                                        self.settings.style_layers, self.settings.style_layer_weights,
                                        self.settings.cuda_device)

        style_guide.prepare()
        self.opt_guides.append(style_guide)

    def forward(self):
        n_iter = [1]
        current_loss = [9999999]

        if self.settings.cuda:
            max_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000000

        def closure():
            if self.settings.cuda_device:
                loss = torch.zeros(1, requires_grad=False).to(torch.device(self.settings.cuda_device))
            else:
                loss = torch.zeros(1, requires_grad=False)

            gradients = []

            for guide in self.opt_guides:
                gradients += guide(self.optimiser, self.opt_tensor, loss, n_iter[0])

            b, c, w, h = self.opt_tensor.grad.size()
            if self.settings.cuda:
                sum_gradients = torch.zeros((b, c, w, h)).detach().to(torch.device(self.settings.cuda_device))
            else:
                sum_gradients = torch.zeros((b, c, w, h)).detach()

            for grad in gradients:
                sum_gradients += grad

            self.opt_tensor.grad = sum_gradients

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % self.settings.log_iterations == (self.settings.log_iterations - 1):
                max_mem_cached = torch.cuda.max_memory_reserved(0) / 1000000
                msg = ''
                msg += 'Iteration: %d, ' % (n_iter[0])
                msg += 'loss: %s, ' % (nice_loss)
                if self.settings.cuda:
                    msg += 'memory used: %s of %s' % (max_mem_cached, max_memory)
                print(msg)

            return loss

        if self.settings.iterations:
            max_iter = int(self.settings.iterations)
            while n_iter[0] <= max_iter:
                self.optimiser.step(closure)

        if self.settings.scale != 1.0:
            if self.settings.rescale_output == True:
                self.opt_tensor = torch.nn.functional.interpolate(self.opt_tensor, size=self.settings.opt_orig_scale, mode='bilinear')
                # tensor = Variable(tensor.data.clone(), requires_grad=True)

        return self.opt_tensor

    # def save(self, fp):
    #     torch.save(self.state_dict(), fp)
    #
    # def load(self, fp):
    #     self.load_state_dict(torch.load(fp))
    #     self.eval()
