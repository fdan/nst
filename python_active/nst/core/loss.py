import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.masked import masked_tensor

import histogram

class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


# https://notebook.community/zklgame/CatEyeNets/test/StyleTransfer-PyTorch
class TVLoss(nn.Module):
    def forward(self, img, target=None, mip_weights=None):
        N, C, H, W = img.size()
        loss = torch.sum(torch.pow(img[:, :, :H - 1, :] - img[:, :, 1:, :], 2))
        loss += torch.sum(torch.pow(img[:, :, :, :W - 1] - img[:, :, :, 1:], 2))
        return loss


class MipLoss(nn.Module):
    def forward(self, input, target, loss_type, mip_weights=None):
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

            if loss_type == 'mse':
                c_ = F.mse_loss(opt_activations, target_activations)
            elif loss_type == 'mae':
                c_ = F.l1_loss(opt_activations, target_activations)
            elif loss_type == 'huber':
                c_ = F.huber_loss(opt_activations, target_activations)
            else:
                raise RuntimeError('unknown loss type: %s' % loss_type)

            loss += c_

        return loss


class MipGramLoss(nn.Module):
    def forward(self, input, target, mip_weights, loss_type):
        opt_layer_activation_pyramid = input
        target_layer_gram_pyramid = target
        loss = 0

        for index, target_gram in enumerate(target_layer_gram_pyramid):
            mip_weight = mip_weights[index]
            opt_activations = opt_layer_activation_pyramid[index]
            opt_gram = GramMatrix()(opt_activations)

            # calculate MSE
            # a_ = torch.sub(opt_gram, target_gram)
            # b_ = torch.pow(a_, 2)
            # c_ = torch.mean(b_)
            # c_ *= mip_weight

            if loss_type == 'mse':
                c_ = F.mse_loss(opt_gram, target_gram)
            elif loss_type == 'mae':
                c_ = F.l1_loss(opt_gram, target_gram)
            elif loss_type == 'huber':
                c_ = F.huber_loss(opt_gram, target_gram)
            else:
                raise RuntimeError('unknown loss type: %s' % loss_type)

            loss += c_

        return loss


class MipHistogramLoss(nn.Module):

    def computeHistogramMatchedActivation(self, input, target, bins):
        assert(len(input.shape) == 3)
        assert(len(target.min.shape) == 1)
        assert(len(target.max.shape) == 1)
        assert(target.histogram.shape[0] == input.shape[0])
        assert(target.min.shape[0] == input.shape[0])
        assert(target.max.shape[0] == input.shape[0])
        # todo: get n. bins from settings
        assert(target.histogram.shape[1] == bins)
        res = input.data.clone() # Clone, we don'input want to change the values of features map or target histogram
        histogram.matchHistogram(res, target.histogram.clone())
        for c in range(res.size(0)):
            res[c].mul_(target.max[c] - target.min[c]) # Values in range [0, max - min]
            res[c].add_(target.min[c])           # Values in range [min, max]
        return res.data.unsqueeze(0)

    def forward(self,
                opt_layer_activation_pyramid,
                target_layer_histogram_pyramid,
                mip_weights,
                bins,
                loss_type):
        """
        target is a guides.Histogram
        """
        loss = 0

        for index, target_histogram in enumerate(target_layer_histogram_pyramid):
            mip_weight = mip_weights[index]
            opt_activation = opt_layer_activation_pyramid[index]

            histogramCorrectedTarget = self.computeHistogramMatchedActivation(opt_activation[0],
                                                                              target_histogram,
                                                                              bins)

            # manual mse loss:
            # a_ = torch.sub(opt_activation, histogramCorrectedTarget)
            # b_ = torch.pow(a_, 2)
            # c_ = torch.mean(b_)

            if loss_type == 'mse':
                c_ = F.mse_loss(opt_activation, histogramCorrectedTarget)
            elif loss_type == 'mae':
                c_ = F.l1_loss(opt_activation, histogramCorrectedTarget)
            elif loss_type == 'huber':
                c_ = F.huber_loss(opt_activation, histogramCorrectedTarget)
            else:
                raise RuntimeError('unknown loss type: %s' % loss_type)

            c_ *= mip_weight

            loss += c_

        return loss


class MipHistogramLossMasked(nn.Module):

    def computeHistogramMatchedActivation(self, input, target, bins):
        target_ = target[0]
        minv = target[1]
        target.max = target[2]
        assert(len(input.shape) == 3)
        assert(len(minv.shape) == 1)
        assert(len(target.max.shape) == 1)
        assert(target_.shape[0] == input.shape[0])
        assert(minv.shape[0] == input.shape[0])
        assert(target.max.shape[0] == input.shape[0])
        # todo: get n. bins from settings
        assert(target_.shape[1] == bins)
        res = input.data.clone() # Clone, we don'input want to change the values of features map or target histogram
        histogram.matchHistogram(res, target_.clone())
        for c in range(res.size(0)):
            res[c].mul_(target.max[c] - minv[c]) # Values in range [0, max - min]
            res[c].add_(minv[c])           # Values in range [min, max]
        return res.data.unsqueeze(0)

    def forward(self,
                opt_layer_activation_pyramid,
                target_layer_histogram_pyramid,
                target,
                mip_weights,
                bins,
                mask,
                loss_type):

        loss = 0

        for index, target_histogram in enumerate(target_layer_histogram_pyramid):
            mip_weight = mip_weights[index]
            opt_activation = opt_layer_activation_pyramid[index]

            histogramCorrectedTarget = self.computeHistogramMatchedActivation(opt_activation[0],
                                                                              target_histogram,
                                                                              bins)

            # todo: only subtract values given by the mask
            # mask and data need to be same dimensions

            opt_activation_mt = masked_tensor(opt_activation.float(), mask)
            histogramCorrectedTarget_mt = masked_tensor(histogramCorrectedTarget.float(), mask)

            # c_ = F.l1_loss(opt_activation_mt, histogramCorrectedTarget_mt)

            # MAE
            a_ = torch.sub(opt_activation_mt, histogramCorrectedTarget_mt)
            c_ = torch.mean(a_)
            c_ *= mip_weight
            loss += c_

        return loss


# class TemporalLoss(nn.Module):
#     pass


# class MSELoss(nn.Module):
#     def forward(self, input, target, mip_weight):
#         a_ = torch.sub(input, target)
#         b_ = torch.pow(a_, 2)
#         c_ = torch.mean(b_)
#         c_ *= mip_weight
#         return c_


# class MaskedMSELoss(nn.Module):
#     def forward(self, input, target, mip_weight):
#         a_ = torch.sub(input, target)
#         b_ = torch.pow(a_, 2)
#         c_ = torch.mean(b_)
#         c_ *= mip_weight
#         return c_
#
#
