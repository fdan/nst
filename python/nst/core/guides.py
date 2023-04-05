import kornia
import torch
import torch.nn as nn

import histogram

from . import utils
from . import loss


class OptGuide(nn.Module):
    def __init__(self, cuda=False):
        super(OptGuide, self).__init__()
        self.name = ""
        self.cuda = cuda
        self.target_map = None
        self.loss_layers = []
        self.loss_function = None
        self.weight = 1.0
        self.target = None

    def prepare(self, *args):
        raise NotImplementedError


class ContentGuide(OptGuide):
    def __init__(self, tensor, vgg, layer, layer_weight, cuda_device=None):
        super(ContentGuide, self).__init__()
        self.name = "content guide"
        self.tensor = tensor
        self.target = None
        self.weight = layer_weight
        self.vgg = vgg
        self.layer = layer
        self.cuda_device = cuda_device

    def prepare(self):
        self.target = self.vgg([self.tensor], [self.layer])[0]

    def loss(self, opt, target):
        loss_fn = loss.MipMSELoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target)

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, 1, 1, cuda=cuda)
        opt_activation = self.vgg(opt_pyramid, [self.layer])[0]

        optimiser.zero_grad()
        weighted_loss = self.loss(opt_activation, self.target) * self.weight

        # backpropagate our weighted loss to calculate gradients
        weighted_loss.backward(retain_graph=True)
        # from here on, opt_tensor.grad contains the derived gradients

        loss += weighted_loss
        return opt_tensor.grad.clone()


# class HistogramGuide(OptGuide):
#     def __init__(self, tensor, weight, cuda_device=None):
#         super(HistogramGuide, self).__init__()
#         self.tensor = tensor
#         self.weight = weight
#         self.cuda_device = cuda_device
#         self.target = None
#
#     def prepare(self):
#         pass
#
#     def loss(self, opt, target, weight):
#         loss_fn = loss.MSELoss()
#         return loss_fn(opt, target, weight)
#
#     def forward(self, optimiser, opt_tensor, loss, iteration):
#         optimiser.zero_grad()
#         opt_hist = torch.histc(opt_tensor, bins=100, min=0, max=1)
#         loss += self.loss(opt_hist, self.target, self.weight)
#         return opt_tensor.grad.clone()


class TVGuide(OptGuide):
    def __init__(self, weight, cuda_device):
        super(TVGuide, self).__init__()
        self.name = "tv guide"
        self.width = 0
        self.height = 0
        self.weight = weight
        self.cuda_device = cuda_device

    def loss(self, opt_tensor):
        loss_fn = loss.TVLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt_tensor) * self.weight

    def forward(self, optimiser, opt_tensor, loss, iteration):
        optimiser.zero_grad()
        weighted_loss = self.loss(opt_tensor) * self.weight
        weighted_loss.backward(retain_graph=True)
        loss += weighted_loss
        return opt_tensor.grad.clone()


class DerivativeGuide(OptGuide):
    def __init__(self, tensor, weight, cuda_device=None):
        super(DerivativeGuide, self).__init__()
        self.tensor = tensor
        self.weight = weight
        self.cuda_device = cuda_device
        self.target = None

    def prepare(self):
        self.target = self._get_derivative(self.tensor)

    def _get_derivative(self, t):
        # deriv = kornia.filters.gaussian_blur2d(tensor, (15, 15), (15, 15))
        # deriv = kornia.filters.spatial_gradient(deriv, mode='diff')
        # deriv = deriv.transpose(0, 2).transpose(1, 2)[0] + deriv.transpose(0, 2).transpose(1, 2)[1]
        # return deriv
        t = kornia.filters.gaussian_blur2d(t, (9, 9), (8, 8))
        t = kornia.filters.laplacian(t, 21)
        t = kornia.filters.laplacian(t, 21)
        return t

    def loss(self, opt, target, weight):
        loss_fn = loss.MSELoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, weight)

    def forward(self, optimiser, opt_tensor, loss, iteration):
        optimiser.zero_grad()
        opt_deriv = self._get_derivative(opt_tensor)
        loss += self.loss(opt_deriv, self.target, self.weight)
        return opt_tensor.grad.clone()


class StyleGramGuide(OptGuide):

    def __init__(self,
                 styles,
                 vgg,
                 style_mips,
                 mip_weights,
                 layers,
                 layer_weights,
                 style_pyramid_span,
                 style_zoom,
                 weight,
                 cuda_device=None,
                 write_gradients=False,
                 write_pyramids = False,
                 outdir='',
                 gradient_ext='jpg'):
        super(StyleGramGuide, self).__init__()
        self.name = "style gram guide"
        self.styles = styles
        self.target = []
        self.target_maps = []
        self.vgg = vgg
        self.cuda_device = cuda_device
        self.style_mips = style_mips
        self.mip_weights = mip_weights
        self.layers = layers
        self.layer_weight = layer_weights
        self.style_pyramid_span = style_pyramid_span
        self.zoom = style_zoom
        self.weight = weight
        self.write_gradients = write_gradients
        self.write_pyramids = write_pyramids
        self.outdir = outdir
        self.gradient_ext = gradient_ext

    def prepare(self, *args):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        for style in self.styles:
            tensor = style.tensor

            if style.target_map.numel() != 0:
                self.target_maps += [style.target_map] * len(self.layers)
            else:
                self.target_maps += [None] * len(self.layers)

            if self.zoom != 1.0:
                tensor = utils.zoom_image(tensor, self.zoom)
                style.alpha = utils.zoom_image(style.alpha, self.zoom)

            style_pyramid = utils.make_gaussian_pyramid(tensor, self.style_pyramid_span, self.style_mips,
                                                        cuda=cuda)

            if self.write_pyramids:
                utils.write_pyramid(style_pyramid, self.outdir)

            style_activations = []

            for layer_activation_pyramid in self.vgg(style_pyramid, self.layers, mask=style.alpha):
                style_activations.append(layer_activation_pyramid)

            vgg_layer_index = 0
            for vgg_layer in style_activations:
                gram_pyramid = []
                mip_index = 0
                for mip_activations in vgg_layer:
                    gram = loss.GramMatrix()(mip_activations).detach()
                    gram_pyramid += [gram]
                    mip_index += 1

                vgg_layer_index += 1

                # we have a list of gram pyramids, one for each style image:
                self.target += [gram_pyramid]

    def loss(self, opt, target, weight):
        loss_fn = loss.MipGramMSELoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, weight) * self.weight

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, self.style_pyramid_span, self.style_mips, cuda=cuda)
        opt_activations = []
        for opt_layer_activation_pyramid in self.vgg(opt_pyramid, self.layers):
            opt_activations.append(opt_layer_activation_pyramid)

        # define an array to store gradients for each vgg layer
        style_gradients = []

        for index, opt_layer_activation_pyramid in enumerate(opt_activations):

            # clear gradients, so we can calculate per layer
            optimiser.zero_grad()
            target_layer_activation_pyramid = self.target[index]
            layer_loss = self.loss(opt_layer_activation_pyramid, target_layer_activation_pyramid, self.mip_weights)
            layer_weight = self.layer_weight[index]
            weighted_layer_loss = layer_weight * layer_loss

            # backpropagate our weighted loss to calculate gradients
            weighted_layer_loss.backward(retain_graph=True)
            # from here on, opt_tensor.grad contains the derived gradients

            loss += weighted_layer_loss

            # to do - write to .pt and convert after
            # # write gradients
            # if self.write_gradients:
            #     utils.write_gradient(opt_tensor, '%s/style/grad/%04d.%s' % (self.outdir, iteration, self.gradient_ext))

            # clone the gradients to a new tensor
            layer_gradients = opt_tensor.grad.clone()

            # if a target map is provided, apply it to the gradients
            if torch.is_tensor(self.target_maps[index]):
                b, c, w, h = opt_tensor.grad.size()

                for i in range(0, c):
                    layer_gradients[0][i] *= self.target_maps[index][0][0]

                # # write the masked gradient
                # if self.write_gradients:
                #     utils.write_gradient(layer_gradients, '%s/style/masked_grad/%04d.%s' % (self.outdir, iteration, self.gradient_ext))

            style_gradients.append(layer_gradients)

        return style_gradients


class StyleHistogramGuide(OptGuide):

    def __init__(self,
                 styles,
                 vgg,
                 style_mips,
                 mip_weights,
                 layers,
                 layer_weights,
                 style_pyramid_span,
                 style_zoom,
                 weight,
                 cuda_device=None,
                 write_gradients=False,
                 write_pyramids = False,
                 outdir='',
                 gradient_ext='jpg'):
        super(StyleHistogramGuide, self).__init__()
        self.name = "style histogram guide"
        self.styles = styles
        self.target = []
        self.target_maps = []
        self.vgg = vgg
        self.cuda_device = cuda_device
        self.style_mips = style_mips
        self.mip_weights = mip_weights
        self.layers = layers
        self.layer_weight = layer_weights
        self.style_pyramid_span = style_pyramid_span
        self.zoom = style_zoom
        self.weight = weight
        self.write_gradients = write_gradients
        self.write_pyramids = write_pyramids
        self.outdir = outdir
        self.gradient_ext = gradient_ext

    def prepare(self, *args):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        for style in self.styles:
            tensor = style.tensor

            if style.target_map.numel() != 0:
                self.target_maps += [style.target_map] * len(self.layers)
            else:
                self.target_maps += [None] * len(self.layers)

            if self.zoom != 1.0:
                tensor = utils.zoom_image(tensor, self.zoom)
                style.alpha = utils.zoom_image(style.alpha, self.zoom)

            style_pyramid = utils.make_gaussian_pyramid(tensor, self.style_pyramid_span, self.style_mips,
                                                        cuda=cuda)

            if self.write_pyramids:
                utils.write_pyramid(style_pyramid, self.outdir)

            style_pyramid_activations = self.vgg(style_pyramid, self.layers, mask=style.alpha)

            bins = 256

            for vgg_layer in style_pyramid_activations:
                histogram_pyramid = []
                for mip_activations in vgg_layer:

                    hist = histogram.computeHistogram(mip_activations[0], bins)
                    min_ = torch.min(mip_activations[0].view(mip_activations.shape[1], -1), 1)[0].data.clone()
                    max_ = torch.max(mip_activations[0].view(mip_activations.shape[1], -1), 1)[0].data.clone()
                    histogram_pyramid.append((hist, min_, max_))

                self.target += [histogram_pyramid]

    def loss(self, opt, target, weight):
        loss_fn = loss.MipHistogramMSELoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, weight) * self.weight

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, self.style_pyramid_span, self.style_mips, cuda=cuda)
        opt_activations = []
        for opt_layer_activation_pyramid in self.vgg(opt_pyramid, self.layers):
            opt_activations.append(opt_layer_activation_pyramid)

        # define an array to store gradients for each vgg layer
        style_gradients = []

        for index, opt_layer_activation_pyramid in enumerate(opt_activations):

            # clear gradients, so we can calculate per layer
            optimiser.zero_grad()
            target_layer_activation_pyramid = self.target[index]
            layer_loss = self.loss(opt_layer_activation_pyramid, target_layer_activation_pyramid, self.mip_weights)
            layer_weight = self.layer_weight[index]
            weighted_layer_loss = layer_weight * layer_loss

            # backpropagate our weighted loss to calculate gradients
            weighted_layer_loss.backward(retain_graph=True)
            # from here on, opt_tensor.grad contains the derived gradients

            loss += weighted_layer_loss

            # to do - write to .pt and convert after
            # # write gradients
            # if self.write_gradients:
            #     utils.write_gradient(opt_tensor, '%s/style/grad/%04d.%s' % (self.outdir, iteration, self.gradient_ext))

            # clone the gradients to a new tensor
            layer_gradients = opt_tensor.grad.clone()

            # if a target map is provided, apply it to the gradients
            if torch.is_tensor(self.target_maps[index]):
                b, c, w, h = opt_tensor.grad.size()

                for i in range(0, c):
                    layer_gradients[0][i] *= self.target_maps[index][0][0]

                # # write the masked gradient
                # if self.write_gradients:
                #     utils.write_gradient(layer_gradients, '%s/style/masked_grad/%04d.%s' % (self.outdir, iteration, self.gradient_ext))

            style_gradients.append(layer_gradients)

        return style_gradients


