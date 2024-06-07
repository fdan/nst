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


class TorchMaskedImage(object):
    def __init__(self, tensor, alpha=torch.zeros(0)):
        self.tensor = tensor
        self.alpha = alpha


class TorchStyle(object):
    def __init__(self, tensor, alpha=torch.zeros(0), target_map=torch.zeros(0)):
        self.tensor = tensor
        self.alpha = alpha
        self.target_map = target_map
        # self.scale = 1.0
        ## to do
        # self.mips = 4
        # self.mip_weights = [1.0]*4
        # self.layers = ['r31', 'r42']
        # self.layer_weights = [0.01, 0.005]
        # self.pyramid_span = 0.5
        # self.zoom = 1.0
        ##


class Nst(torch.nn.Module):
    def __init__(self):
        super(Nst, self).__init__()
        self.vgg = vgg.VGG(pool='max')
        self.content = torch.zeros(0)
        self.temporal_content = torch.zeros(0)
        self.temporal_weight_mask = torch.zeros(0)
        self.content_scale = 1.0
        self.styles = []
        self.opt_tensor = torch.zeros(0)
        self.opt_masked = torch.zeros(0)
        self.opt_guides = []
        self.optimiser = None
        self.settings = settings.NstSettings()
        self.start_iter = 1

    def prepare(self):
        if self.settings.cuda:
            self.settings.cuda_device = utils.get_cuda_device()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # clone the opt tensor, doesn't require grad
        self.opt_masked = self.opt_tensor.clone()

        if self.settings.engine == 'gpu':
            self.vgg.cuda()
            self.settings.cuda = True
            self.vgg.load_state_dict(torch.load(self.settings.model_path))

        elif self.settings.engine == 'cpu':
            self.vgg.load_state_dict(torch.load(self.settings.model_path))
            self.settings.cuda = False

        # optimiser
        if self.settings.optimiser == 'lbfgs':
            # print('optimiser is lbfgs')
            self.optimiser = optim.LBFGS([self.opt_tensor],
                                         lr=self.settings.learning_rate)

        elif self.settings.optimiser == 'adam':
            # print('optimiser is adam')
            self.optimiser = optim.Adam([self.opt_tensor], lr=self.settings.learning_rate, amsgrad=True)

        elif self.settings.optimiser == 'adamw':
            # print('optimiser is adamw')
            self.optimiser = optim.AdamW([self.opt_tensor], lr=self.settings.learning_rate)

        elif self.settings.optimiser == 'adagrad':
            # print('optimiser is adagrad')
            self.optimiser = optim.Adagrad([self.opt_tensor], lr=self.settings.learning_rate)

        elif self.settings.optimiser == 'rmsprop':
            # print('optimiser is rmsprop')
            self.optimiser = optim.RMSprop([self.opt_tensor], lr=self.settings.learning_rate)

        elif self.settings.optimiser == 'asgd':
            # print('optimiser is asgd')
            self.optimiser = optim.ASGD([self.opt_tensor], lr=self.settings.learning_rate)

        else:
            raise Exception("unsupported optimiser:", self.settings.optimiser)

        # content can be null
        if self.content.numel() != 0:
            content_guide = guides.ContentGuide(
                self.content,
                self.vgg,
                self.settings.content_layer,
                self.settings.content_layer_weight,
                self.settings.content_loss_type,
                # self.temporal_weight_mask,
                self.styles[0].target_map,
                self.settings.cuda_device

            )

            content_guide.prepare()
            self.opt_guides.append(content_guide)
        else:
            print('not using content guide')

        # # temporal content can be null
        # if self.temporal_content.numel() != 0:
        #     temporal_content_guide = guides.TemporalContentGuide(
        #         self.temporal_content,
        #         self.vgg,
        #         self.settings.content_layer,
        #         self.settings.temporal_weight,
        #         self.settings.content_loss_type,
        #         self.temporal_weight_mask,
        #         self.settings.cuda_device
        #     )
        #
        #     temporal_content_guide.prepare()
        #     self.opt_guides.append(temporal_content_guide)
        # else:
        #     print('not using temporal content guide')

        if self.settings.laplacian_weight:
            laplacian_guide = guides.LaplacianGuide(
                self.vgg,
                self.content,
                self.settings.laplacian_weight,
                self.settings.laplacian_loss_layer,
                self.settings.laplacian_filter_kernel,
                self.settings.laplacian_blur_kernel,
                self.settings.laplacian_blur_sigma,
                self.settings.laplacian_loss_type,
                self.settings.cuda_device,
            )

            laplacian_guide.prepare()
            self.opt_guides.append(laplacian_guide)

        if self.settings.gram_weight:
            style_gram_guide = guides.StyleGramGuide(
                self.styles,
                self.vgg,
                self.settings.style_mips,
                self.settings.mip_weights,
                self.settings.style_layers,
                self.settings.style_layer_weights,
                self.settings.style_pyramid_span,
                self.settings.style_zoom,
                self.settings.gram_weight,
                self.settings.gram_loss_type,
                outdir=self.settings.outdir,
                write_pyramids=self.settings.write_pyramids,
                write_gradients=self.settings.write_gradients,
                cuda_device=self.settings.cuda_device,
                mask_layers=self.settings.mask_layers
            )

            style_gram_guide.prepare()
            self.opt_guides.append(style_gram_guide)

        if self.settings.histogram_weight:
            style_histogram_guide = guides.StyleHistogramGuide(
                self.styles,
                self.vgg,
                self.settings.style_mips,
                self.settings.mip_weights,
                self.settings.style_layers,
                self.settings.style_layer_weights,
                self.settings.style_pyramid_span,
                self.settings.style_zoom,
                self.settings.histogram_weight,
                self.settings.histogram_bins,
                self.settings.histogram_loss_type,
                self.settings.random_rotate_mode,
                self.settings.random_rotate,
                self.settings.random_crop,
                self.temporal_weight_mask,
                outdir=self.settings.outdir,
                write_pyramids=self.settings.write_pyramids,
                write_gradients=self.settings.write_gradients,
                cuda_device=self.settings.cuda_device,
                mask_layers=self.settings.mask_layers
            )

            style_histogram_guide.prepare()
            self.opt_guides.append(style_histogram_guide)

        if self.settings.tv_weight:
            tv_guide = guides.TVGuide(
                self.settings.tv_weight,
                None,
                self.settings.cuda
            )

            self.opt_guides.append(tv_guide)

        # if self.settings.temporal_weight and self.temporal_weight_mask.numel() != 0 and self.temporal_content.numel() != 0:
        #     temporal_guide = guides.TemporalContentGuide(
        #         self.content,
        #         self.vgg,
        #         self.settings.content_layer,
        #         self.settings.temporal_weight,
        #         self.settings.content_loss_type,
        #         self.temporal_weight_mask,
        #         self.settings.cuda_device
        #     )
        #
        #     temporal_guide.prepare()
        #     self.opt_guides.append(temporal_guide)
        # # else:
        # #     print(11.2, self.settings.temporal_weight, self.temporal_weight_mask)

    def forward(self):
        n_iter = [self.start_iter]
        current_loss = [9999999]

        if self.settings.cuda:
            max_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000000

        # self.style_index = 0
        # self.style_switch_counter = 0

        def closure():
            if self.settings.cuda_device:
                loss = torch.zeros(1, requires_grad=False).to(torch.device(self.settings.cuda_device))
            else:
                loss = torch.zeros(1, requires_grad=False)

            gradients = []

            # print(1.1, self.opt_masked.sum())

            for guide in self.opt_guides:
                # gradients += guide(self.optimiser, self.opt_tensor, loss, n_iter[0], self.style_index)
                gradients += guide(self.optimiser, self.opt_tensor, loss, n_iter[0])
                # gradients += guide(self.optimiser, o, loss, n_iter[0])

            b, c, w, h = self.opt_tensor.grad.size()
            if self.settings.cuda:
                sum_gradients = torch.zeros((b, c, w, h)).detach().to(torch.device(self.settings.cuda_device))
            else:
                sum_gradients = torch.zeros((b, c, w, h)).detach()

            for grad in gradients:
                sum_gradients += grad

            if self.settings.write_gradients:
                utils.write_tensor(sum_gradients, '%s/grad/%04d.pt' % (self.settings.outdir, n_iter[0]))

            # print(1.1, sum_gradients.sum())
            # print(1.2, self.opt_masked.sum())
            # self.opt_tensor.grad = sum_gradients
            # self.opt_masked -= (sum_gradients * 200000)
            # self.opt_masked.data.sub_(sum_gradients.data / self.settings.learning_rate)

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

            if self.settings.progressive_output:
                if n_iter[0] % self.settings.progressive_interval == 0:
                    fp = self.settings.outdir + '/prog/%04d.pt' % n_iter[0]
                    utils.write_tensor(self.opt_tensor, fp)

            return loss

        if self.settings.iterations:
            max_iter = int(self.settings.iterations)
            while n_iter[0] <= max_iter:
                self.optimiser.step(closure)

        return self.opt_tensor
        # return self.opt_masked

    # def save(self, fp):
    #     torch.save(self.state_dict(), fp)
    #
    # def load(self, fp):
    #     self.load_state_dict(torch.load(fp))
    #     self.eval()
