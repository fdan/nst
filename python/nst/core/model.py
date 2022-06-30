import json
import os

import torch
from torch import optim

from . import guides
from . import vgg
from . import utils
import nst.core as core

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False





class Nst(torch.nn.Module):
    def __init__(self):
        super(Nst, self).__init__()
        self.vgg = vgg.VGG()
        self.content = None
        self.content_scale = 1.0
        self.styles = []
        self.opt_tensor = None
        self.opt_guides = []
        self.optimiser = None
        self.settings = core.NstSettings()

    def prepare(self):
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

        if self.settings.optimiser_name == 'lbfgs':
            self.optimiser = optim.LBFGS([self.opt_tensor], lr=self.settings.learning_rate)
        elif self.settings.optimiser_name == 'adam':
            self.optimiser = optim.Adam([self.opt_tensor], lr=self.settings.learning_rate)

        # handle rescale of tensors here
        if self.settings.scale != 1:
            self.content = utils.rescale_tensor(self.content, self.settings.scale)
            self.opt_tensor = utils.rescale_tensor(self.opt_tensor, self.settings.scale)

            for style in self.styles:
                i = self.styles.index(style)
                self.styles[i].tensor = utils.rescale_tensor(style.tensor, self.settings.scale)

        content_guide = guides.ContentGuide(self.content, self.vgg, self.settings.content_layer,
                                            self.settings.content_layer_weight, self.settings.cuda_device)

        content_guide.prepare()
        self.opt_guides.append(content_guide)

        style_guide = guides.StyleGuide(self.styles, self.vgg, self.settings.style_mips,
                                        self.settings.pyramid_scale_factor, self.settings.style_mip_weights,
                                        self.settings.style_layers, self.settings.style_layer_weights,
                                        self.settings.cuda_device, scale=1.0)

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

                # print(1.1, self.opt_tensor[0][0][10][10])
            return loss

        if self.settings.iterations:
            max_iter = int(self.settings.iterations)
            while n_iter[0] <= max_iter:
                self.optimiser.step(closure)

        return self.opt_tensor

    # def save(self, fp):
    #     torch.save(self.state_dict(), fp)
    #
    # def load(self, fp):
    #     self.load_state_dict(torch.load(fp))
    #     self.eval()
