import os

import kornia
import torch
import torch.nn as nn

import histogram

from . import utils
from . import loss


class Histogram(nn.Module):
    def __init__(self, histogram, min, max):
        super(Histogram, self).__init__()
        self.histogram = histogram
        self.min = min
        self.max = max


class OptGuide(nn.Module):
    def __init__(self, cuda=False):
        super(OptGuide, self).__init__()
        self.name = ""
        self.cuda = cuda
        self.target_map = torch.Tensor(0)
        self.loss_layers = []
        self.loss_function = None
        self.weight = 1.0
        self.target = None

    def prepare(self, *args):
        raise NotImplementedError


class ContentGuide(OptGuide):
    def __init__(self, tensor, vgg, layer, layer_weight, loss_type, temporal_mask, cuda_device=None):
        super(ContentGuide, self).__init__()
        self.name = "content guide"
        self.tensor = tensor
        self.loss_type = loss_type
        self.target = None
        self.weight = layer_weight
        self.vgg = vgg
        self.layer = layer
        self.cuda_device = cuda_device

    def prepare(self):
        self.target = self.vgg([self.tensor], [self.layer])[0]
        # self.target = self.vgg([self.tensor], [self.layer], mask=self.temporal_mask)[0]

    def loss(self, opt, target):
        loss_fn = loss.MipLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, self.loss_type)

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, 1, 1, cuda=cuda)
        opt_activation = self.vgg(opt_pyramid, [self.layer])[0]
        # opt_activation = self.vgg(opt_pyramid, [self.layer], mask=self.temporal_mask)[0]

        optimiser.zero_grad()
        weighted_loss = self.loss(opt_activation, self.target) * self.weight

        # backpropagate our weighted loss to calculate gradients
        weighted_loss.backward(retain_graph=True)
        # from here on, opt_tensor.grad contains the derived gradients

        # manually apply gradients:
        # x.data.sub_(lr * x.grad.data())
        # x.grad.data.zero_()

        loss += weighted_loss

        layer_gradients = opt_tensor.grad.clone()

        # if self.temporal_mask.numel() != 0:
        #     b, c, w, h = opt_tensor.grad.size()
        #     for i in range(0, c):
        #         layer_gradients[0][i] *= self.temporal_mask[0][0]

        return layer_gradients


class TemporalContentGuide(OptGuide):
    def __init__(self, tensor, vgg, layer, weight, loss_type, mask, cuda_device=None):
        super(TemporalContentGuide, self).__init__()
        self.name = "temporal content guide"
        self.tensor = tensor
        self.loss_type = loss_type
        self.mask = mask
        self.resized_mask = None
        self.target = None
        self.weight = weight
        self.vgg = vgg
        self.layer = layer
        self.cuda_device = cuda_device

    def prepare(self):
        self.target = self.vgg([self.tensor], [self.layer])[0]

        # apply the mask
        b, c, h, w = self.target[0].size()
        self.resized_mask = utils.rescale_tensor_by_tensor(self.mask, self.target[0])
        for i in range(0, c):
            self.target[0][0][i] *= self.resized_mask[0][0]


    def loss(self, opt, target):
        loss_fn = loss.MipLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, self.loss_type)

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, 1, 1, cuda=cuda)
        opt_activation = self.vgg(opt_pyramid, [self.layer])[0]

        # need to apply the mask to the opt activations
        b, c, h, w = opt_activation[0].size()
        for i in range(0, c):
            opt_activation[0][0][i] *= self.resized_mask[0][0]

        optimiser.zero_grad()
        weighted_loss = self.loss(opt_activation, self.target) * self.weight

        # backpropagate our weighted loss to calculate gradients
        weighted_loss.backward(retain_graph=True)
        # from here on, opt_tensor.grad contains the derived gradients

        loss += weighted_loss
        return opt_tensor.grad.clone()


class TVGuide(OptGuide):
    def __init__(self, weight, loss_type, cuda_device):
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

        return loss_fn(opt_tensor)

    def forward(self, optimiser, opt_tensor, loss, iteration):
        optimiser.zero_grad()
        weighted_loss = self.loss(opt_tensor) * self.weight
        weighted_loss.backward(retain_graph=True)
        loss += weighted_loss
        return opt_tensor.grad.clone()


class LaplacianGuide(OptGuide):
    def __init__(self,
                 vgg,
                 tensor,
                 weight,
                 layer,
                 laplacian_kernel,
                 blur_kernel,
                 blur_sigma,
                 loss_type,
                 cuda_device=None):
        super(LaplacianGuide, self).__init__()
        self.vgg = vgg
        self.tensor = tensor
        self.weight = weight
        self.layer = layer
        self.laplacian_kernel = laplacian_kernel
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.loss_type = loss_type
        self.cuda_device = cuda_device
        self.target = None

    def prepare(self):
        target_deriv = self._get_laplacian(self.tensor)
        self.target = self.vgg([target_deriv], [self.layer])[0]

    def _get_laplacian(self, t):
        t = kornia.filters.gaussian_blur2d(t, (self.blur_kernel, self.blur_kernel), (self.blur_sigma, self.blur_sigma))
        t = kornia.filters.laplacian(t, self.laplacian_kernel)
        t = kornia.filters.laplacian(t, self.laplacian_kernel)
        return t

    def loss(self, opt, target):
        loss_fn = loss.MipLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, self.loss_type)

    def forward(self, optimiser, opt_tensor, loss, iteration):

        opt_deriv = self._get_laplacian(opt_tensor)
        opt_activation = self.vgg([opt_deriv], [self.layer])[0]
        optimiser.zero_grad()
        weighted_loss = self.loss(opt_activation, self.target) * self.weight
        weighted_loss.backward(retain_graph=True)
        loss += weighted_loss
        return opt_tensor.grad.clone()

        # alternate code for if we allowed multiple layers for deriv guide
        # deriv_gradients = []
        # for index, layer_activation in enumerate(opt_activations):
        #     optimiser.zero_grad()
        #     loss += self.loss(layer_activation, self.target[index], self.weight)
        #     deriv_gradients.append(opt_tensor.grad.clone())
        # return deriv_gradients


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
                 loss_type,
                 cuda_device=None,
                 write_gradients=False,
                 write_pyramids = False,
                 outdir='',
                 mask_layers = [],
                 gradient_ext='jpg'):
        super(StyleGramGuide, self).__init__()
        self.name = "style gram guide"
        self.styles = styles
        self.target_maps = []
        self.loss_type = loss_type

        for s in styles:
            if s.target_map.numel() != 0:
                self.target_maps.append(s.target_map)

        self.target = []

        self.vgg = vgg
        self.cuda_device = cuda_device
        self.style_mips = style_mips
        self.mip_weights = mip_weights
        self.layers = layers
        self.mask_layers = mask_layers
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
                # normalise the map
                style.target_map *= 1000

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
        loss_fn = loss.MipGramLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, weight, self.loss_type) * self.weight

    def forward(self, optimiser, opt_tensor, loss, iteration):
        # print(5.0)

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

            # clone the gradients to a new tensor
            layer_gradients = opt_tensor.grad.clone()

            # get the layer name
            layer_name = self.layers[index]

            # if a target map is provided, apply it to the gradients
            if torch.is_tensor(self.target_maps[index]) and layer_name in self.mask_layers:
                # print(5.5)
                b, c, w, h = opt_tensor.grad.size()

                for i in range(0, c):
                    layer_gradients[0][i] *= self.target_maps[index][0][0]
            # else:
            #     print(5.6)

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
                 bins,
                 loss_type,
                 random_rotate_mode,
                 random_rotate,
                 random_crop,
                 cuda_device=None,
                 write_gradients=False,
                 write_pyramids = False,
                 outdir='',
                 mask_layers = [],
                 gradient_ext='jpg'):
        super(StyleHistogramGuide, self).__init__()
        self.name = "style histogram guide"
        self.styles = styles
        self.loss_type = loss_type
        self.random_rotate_mode = random_rotate_mode
        self.random_rotate = random_rotate
        self.random_crop = random_crop
        self.target = []
        self.target_maps = []
        self.vgg = vgg
        self.cuda_device = cuda_device
        self.style_mips = style_mips
        self.mip_weights = mip_weights
        self.layers = layers
        self.mask_layers = mask_layers
        self.layer_weight = layer_weights
        self.style_pyramid_span = style_pyramid_span
        self.zoom = style_zoom
        self.weight = weight
        self.bins = bins
        self.write_gradients = write_gradients
        self.write_pyramids = write_pyramids
        self.outdir = outdir
        self.gradient_ext = gradient_ext

    def prepare(self, *args):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        # todo: resize the temporal mask to the dims of the opt activations

        for style in self.styles:
            tensor = style.tensor

            # if self.temporal_mask.numel() != 0:
            #     self.target_maps += [self.temporal_mask] * len(self.layers)
            # else:
            #     self.target_maps += [None] * len(self.layers)

            if style.target_map.numel() != 0:
                self.target_maps += [style.target_map] * len(self.layers)
            else:
                self.target_maps += [None] * len(self.layers)

            if self.random_rotate_mode != 'none':
                tensor = utils.random_rotate_crop(tensor, self.random_rotate, self.random_crop, self.random_rotate_mode)

            if self.zoom != 1.0:
                tensor = utils.zoom_image(tensor, self.zoom)
                style.alpha = utils.zoom_image(style.alpha, self.zoom)

            style_pyramid = utils.make_gaussian_pyramid(tensor, self.style_pyramid_span, self.style_mips,
                                                        cuda=cuda)

            if self.write_pyramids:
                utils.write_pyramid(style_pyramid, self.outdir)

            style_layer_pyramid_activations = self.vgg(style_pyramid, self.layers, mask=style.alpha)

            for vgg_layer in style_layer_pyramid_activations:

                # why are we calling it a histogram pyramid?
                histogram_pyramid = []
                for mip_activations in vgg_layer: # would this be better named something like pyramid_level?

                    # print(100.1, mip_activations.size()) # tensor
                    # p1 m1: [1, 64, 907, 930]
                    # p1 m2: [1, 64, 510, 522]
                    # p1 m3: [1, 64, 286, 293]
                    # p1 m4: [1, 64, 161, 165]
                    # p1 m5: [1, 64, 90, 92]
                    #
                    # p2 m1: [1, 128, 453, 465]
                    # p2 m2: [1, 128, 255, 261]
                    # p2 m3: [1, 128, 143, 146]
                    # p2 m4: [1, 128, 80, 82]
                    # p2 m5: [1, 128, 45, 46]

                    hist = histogram.computeHistogram(mip_activations[0], self.bins)
                    # print(100.2, hist.size()) # (64, 256), (128, 256), (256, 256), (512, 256) - for 256 bins
                    min_ = torch.min(mip_activations[0].view(mip_activations.shape[1], -1), 1)[0].data.clone()
                    # print(100.3, min_.size()) # zeros, size:
                    max_ = torch.max(mip_activations[0].view(mip_activations.shape[1], -1), 1)[0].data.clone()
                    # print(100.4, max_.size()) # thousands, size:

                    h = Histogram(hist, min_, max_)
                    # histogram_pyramid.append((hist, min_, max_))
                    histogram_pyramid.append(h)

                self.target += [histogram_pyramid]

    def loss(self, opt, target, weight):
        loss_fn = loss.MipHistogramLoss()

        if self.cuda_device:
            loss_fn = loss_fn.cuda()

        return loss_fn(opt, target, weight, self.bins, self.loss_type) * self.weight

    def forward(self, optimiser, opt_tensor, loss, iteration):
        if self.cuda_device:
            cuda = True
        else:
            cuda = False

        opt_pyramid = utils.make_gaussian_pyramid(opt_tensor, self.style_pyramid_span, self.style_mips, cuda=cuda)
        # mask_pyramid = utils.make_gaussian_pyramid(self.temporal_mask, self.style_pyramid_span, self.style_mips, cuda=cuda)

        opt_activations = []
        # for opt_layer_activation_pyramid in self.vgg(opt_pyramid, self.layers, mask=mask_pyramid):
        # for opt_layer_activation_pyramid in self.vgg(opt_pyramid, self.layers, mask=self.temporal_mask):
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

            # clone the gradients to a new tensor
            layer_gradients = opt_tensor.grad.clone()
            # layer_gradients = o.grad.clone()

            # get the layer name
            layer_name = self.layers[index]

            if torch.is_tensor(self.target_maps[index]) and layer_name in self.mask_layers:
                b, c, w, h = opt_tensor.grad.size()
                for i in range(0, c):
                    layer_gradients[0][i] *= (self.target_maps[index][0][0] * 0.1)

            style_gradients.append(layer_gradients)

        return style_gradients


