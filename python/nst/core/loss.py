import torch
import torch.nn as nn

import histogram

class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class TVLoss(nn.Module):
    def forward(self, input, target=None, mip_weights=None):
        height = input.shape[1]
        width = input.shape[2]

        dx = input[:, :height - 1, :width - 1, :] - input[:, 1:, :width - 1, :]
        dy = input[:, :height - 1, :width - 1, :] - input[:, :height - 1, 1:, :]
        loss_ = (dx ** 2 + dy ** 2).sum() ** 0.5
        # return loss normalized by image and batch size
        return loss_ / (width * height * input.shape[0])


# class HistogramLoss(nn.Module):
#     def forward(self, input, target, mip_weights=None):
#         # target is a multidimensional tensor, with a channel for each feature histogram
#         pass


class MipMSELoss(nn.Module):
    def forward(self, input, target, mip_weights=None):
        """
        input and target are both layer activation pyramids, i.e. a list of mip tensors
        """
        opt_layer_activation_pyramid = input
        target_layer_activation_pyramid = target
        loss = 0

        # the target may be smaller length than input, i.e. content loss is not pyramid.  iterate through target
        # or just assume single layer
        for index, target_activations in enumerate(target_layer_activation_pyramid):
            opt_activations = opt_layer_activation_pyramid[index]
            a_ = torch.sub(opt_activations, target_activations)
            b_ = torch.pow(a_, 2)
            c_ = torch.mean(b_)
            loss += c_

        return loss


class MipGramMSELoss(nn.Module):
    def forward(self, input, target, mip_weights):
        opt_layer_activation_pyramid = input
        target_layer_gram_pyramid = target
        loss = 0

        for index, target_gram in enumerate(target_layer_gram_pyramid):
            mip_weight = mip_weights[index]
            opt_activations = opt_layer_activation_pyramid[index]
            opt_gram = GramMatrix()(opt_activations)
            # calculate MSE
            a_ = torch.sub(opt_gram, target_gram)
            b_ = torch.pow(a_, 2)
            c_ = torch.mean(b_)
            c_ *= mip_weight

            loss += c_

        return loss


class MipHistogramMSELoss(nn.Module):

    def computeHistogramMatchedActivation(self, input, target):
        target_ = target[0]
        minv = target[1]
        maxv = target[2]
        assert(len(input.shape) == 3)
        assert(len(minv.shape) == 1)
        assert(len(maxv.shape) == 1)
        assert(target_.shape[0] == input.shape[0])
        assert(minv.shape[0] == input.shape[0])
        assert(maxv.shape[0] == input.shape[0])
        assert(target_.shape[1] == 256)
        res = input.data.clone() # Clone, we don'input want to change the values of features map or target histogram
        histogram.matchHistogram(res, target_.clone())
        for c in range(res.size(0)):
            res[c].mul_(maxv[c] - minv[c]) # Values in range [0, max - min]
            res[c].add_(minv[c])           # Values in range [min, max]
        return res.data.unsqueeze(0)

    def forward(self, input, target, mip_weights):
        opt_layer_activation_pyramid = input
        target_layer_histogram_pyramid = target
        loss = 0

        for index, target_histogram in enumerate(target_layer_histogram_pyramid):
            mip_weight = mip_weights[index]
            opt_activation = opt_layer_activation_pyramid[index]

            histogramCorrectedTarget = self.computeHistogramMatchedActivation(opt_activation[0],
                                                                              target_histogram)

            a_ = torch.sub(opt_activation, histogramCorrectedTarget)
            b_ = torch.pow(a_, 2)
            c_ = torch.mean(b_)
            c_ *= mip_weight

            loss += c_

        return loss


class MSELoss(nn.Module):
    def forward(self, input, target, mip_weight):
        a_ = torch.sub(input, target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        c_ *= mip_weight
        return c_