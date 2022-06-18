import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


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


class MSELoss(nn.Module):
    def forward(self, input, target, mip_weight):
        a_ = torch.sub(input, target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        c_ *= mip_weight
        return c_