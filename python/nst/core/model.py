import torch
from torch import optim

from . import guides
from . import vgg
from . import utils

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False


class TorchStyle(object):
    def __init__(self, tensor, in_mask, target_map):
        self.tensor = tensor
        self.in_mask = in_mask
        self.target_map = target_map
        self.scale = 1.0


class Nst(torch.nn.Module):
    def __init__(self):
        super(Nst, self).__init__()

        # core data
        self.vgg = vgg.VGG()
        self.content = None
        self.content_scale = 1.0
        self.styles = []
        self.opt_tensor = None
        self.opt_guides = []

        # options
        self.engine = 'gpu'
        self.cuda = True
        self.model_path = ''
        self.optimiser_name = 'adam'
        self.optimiser = None
        self.cuda_device = 0
        self.pyramid_scale_factor = 0.63
        self.style_mips = 4
        self.style_layers = ['p1', 'p2', 'r31', 'r42']
        self.style_layer_weights = [1.0, 1.0, 1.0, 1.0]
        self.content_layer = 'r41'
        self.content_layer_weight = 1.0
        self.style_mip_weights = [1.0] * self.style_mips
        self.content_mips = 1
        self.optimiser = 'lbfgs'
        self.learning_rate = 1.0
        self.scale = 1.0
        self.iterations = 500
        self.log_iterations = 20

        # self.prepare()

    def prepare(self):
        if self.cuda:
            self.cuda_device = utils.get_cuda_device()

        for param in self.vgg.parameters():
            param.requires_grad = False
        if self.engine == 'gpu':
            self.vgg.cuda()
            self.cuda = True
            self.vgg.load_state_dict(torch.load(self.model_path))

        elif self.engine == 'cpu':
            self.vgg.load_state_dict(torch.load(self.model_path))
            self.cuda = False

        if self.optimiser_name == 'lbfgs':
            self.optimiser = optim.LBFGS([self.opt_tensor], lr=self.learning_rate)
        elif self.optimiser_name == 'adam':
            self.optimiser = optim.Adam([self.opt_tensor], lr=self.learning_rate)

        # handle rescale of tensors here
        if self.scale != 1:
            self.content = utils.rescale_tensor(self.content, self.scale)
            self.opt_tensor = utils.rescale_tensor(self.opt_tensor, self.scale)

            for style in self.styles:
                i = self.styles.index(style)
                self.styles[i].tensor = utils.rescale_tensor(style.tensor, self.scale)


        content_guide = guides.ContentGuide(self.content, self.vgg, self.content_layer, self.content_layer_weight,
                                            self.cuda_device)
        content_guide.prepare()
        self.opt_guides.append(content_guide)

        style_guide = guides.StyleGuide(self.styles, self.vgg, self.style_mips, self.pyramid_scale_factor,
                                        self.style_mip_weights, self.style_layers, self.style_layer_weights,
                                        self.cuda_device, scale=1.0)
        style_guide.prepare()
        self.opt_guides.append(style_guide)

    def forward(self):
        n_iter = [1]
        current_loss = [9999999]

        if self.cuda:
            max_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000000

        def closure():
            if self.cuda_device:
                loss = torch.zeros(1, requires_grad=False).to(torch.device(self.cuda_device))
            else:
                loss = torch.zeros(1, requires_grad=False)

            gradients = []

            for guide in self.opt_guides:
                gradients += guide(self.optimiser, self.opt_tensor, loss, n_iter[0])

            b, c, w, h = self.opt_tensor.grad.size()
            if self.cuda:
                sum_gradients = torch.zeros((b, c, w, h)).detach().to(torch.device(self.cuda_device))
            else:
                sum_gradients = torch.zeros((b, c, w, h)).detach()

            for grad in gradients:
                sum_gradients += grad

            self.opt_tensor.grad = sum_gradients

            nice_loss = '{:,.0f}'.format(loss.item())
            current_loss[0] = loss.item()
            n_iter[0] += 1
            if n_iter[0] % self.log_iterations == (self.log_iterations - 1):
                max_mem_cached = torch.cuda.max_memory_reserved(0) / 1000000
                msg = ''
                msg += 'Iteration: %d, ' % (n_iter[0])
                msg += 'loss: %s, ' % (nice_loss)
                if self.cuda:
                    msg += 'memory used: %s of %s' % (max_mem_cached, max_memory)
                print(msg)
            return loss

        if self.iterations:
            max_iter = int(self.iterations)
            while n_iter[0] <= max_iter:
                self.optimiser.step(closure)

        return self.opt_tensor

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        self.load_state_dict(torch.load(fp))
        self.eval()
