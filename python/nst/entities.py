import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


# gatys weights:
gw = {}
gw['r11'] = 0.244140625
gw['r12'] = 0.244140625
gw['r21'] = 0.06103515625
gw['r22'] = 0.06103515625
gw['r31'] = 0.0152587890625
gw['r32'] = 0.0152587890625
gw['r34'] = 0.0152587890625
gw['r41'] = 0.003814697265625
gw['r42'] = 0.003814697265625
gw['r43'] = 0.003814697265625
gw['r44'] = 0.003814697265625
gw['r51'] = 0.003814697265625
gw['r52'] = 0.003814697265625
gw['r53'] = 0.003814697265625
gw['r54'] = 0.003814697265625


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):

    # layers = {}
    # layers['r11'] = {'channels': 64, 'x': 512}
    # layers['r12'] = {'channels': 64, 'x': 512}
    # layers['r21'] = {'channels': 128, 'x': 256}
    # layers['r22'] = {'channels': 128, 'x': 256}
    # layers['r31'] = {'channels': 256, 'x': 128}
    # layers['r32'] = {'channels': 256, 'x': 128}
    # layers['r34'] = {'channels': 256, 'x': 128}
    # layers['r41'] = {'channels': 512, 'x': 64}
    # layers['r42'] = {'channels': 512, 'x': 64}
    # layers['r43'] = {'channels': 512, 'x': 64}
    # layers['r44'] = {'channels': 512, 'x': 64}
    # layers['r51'] = {'channels': 512, 'x': 32}
    # layers['r52'] = {'channels': 512, 'x': 32}
    # layers['r53'] = {'channels': 512, 'x': 32}
    # layers['r54'] = {'channels': 512, 'x': 32}

    def __init__(self, pool='max', conv_kernel_size=3, conv_kernel_padding=1, pool_kernel_size=2, pool_stride=2):

        super(VGG, self).__init__()
        # vgg modules

        # note: first two args of Conv2d are in channels, out channels
        # where is the x and y dimensions of each filter defined?
        # a: it's not defined, they come from the input tensor size
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding=conv_kernel_padding)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride) #stride is how much it jumps
            self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool4 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool5 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool2 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool3 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool4 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.pool5 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, tensor_pyramid, out_keys, mask=None):
        """
        :param out_keys: [str]
        :return: [torch.Tensor]
        """
        out_path = '/mnt/ala/research/danielf/nst/experiments/output/temp14/'
        out = {}
        out['r11'] = []
        out['r12'] = []
        out['p1'] = []
        out['r21'] = []
        out['r22'] = []
        out['p2'] = []
        out['r31'] = []
        out['r32'] = []
        out['r33'] = []
        out['r34'] = []
        out['p3'] = []
        out['r41'] = []
        out['r42'] = []
        out['r43'] = []
        out['r44'] = []
        out['p4'] = []
        out['r51'] = []
        out['r52'] = []
        out['r53'] = []
        out['r54'] = []
        out['p5'] = []

        result = []

        def process(layer_activations, layer_name):
            b, c, w, h = layer_activations.size()

            if layer_name not in out_keys:
                return layer_activations

            if not torch.is_tensor(mask):
                return layer_activations

            mask_scaled = torch.nn.functional.interpolate(mask, size=(w, h))

            # normalise: ensure mean activation remains same
            # mask_normalisation = (w * h) / mask.sum()
            # mask_normalised = torch.div(mask_scaled, mask_normalisation)

            masked_activations = layer_activations.clone()

            for i in range(0, c):
                masked_activations[0][i] *= mask_scaled[0][0]
                # masked_activations[0][i] *= mask_normalised[0][0]

            return masked_activations

        #  for each pyramid in the list, i.e. for each mip of the image
        for tensor_index in range(0, len(tensor_pyramid)):

            # for a given mip of the image, get the activations
            tensor = tensor_pyramid[tensor_index]

            # out['r11'] += [process(F.relu(self.conv1_1(tensor)), 'r11')]
            # out['r12'] += [process(F.relu(self.conv1_2(out['r11'][tensor_index])), 'r12')]
            # out['p1']  += [process(self.pool1(out['r12'][tensor_index]), 'p1')]
            # out['r21'] += [process(F.relu(self.conv2_1(out['p1'][tensor_index])), 'r21')]
            # out['r22'] += [process(F.relu(self.conv2_2(out['r21'][tensor_index])), 'r22')]
            # out['p2']  += [process(self.pool2(out['r22'][tensor_index]), 'p2')]
            # out['r31'] += [process(F.relu(self.conv3_1(out['p2'][tensor_index])), 'r31')]
            # out['r32'] += [process(F.relu(self.conv3_2(out['r31'][tensor_index])), 'r32')]
            # out['r33'] += [process(F.relu(self.conv3_3(out['r32'][tensor_index])), 'r33')]
            # out['r34'] += [process(F.relu(self.conv3_4(out['r33'][tensor_index])), 'r34')]
            # out['p3']  += [process(self.pool3(out['r34'][tensor_index]), 'p3')]
            # out['r41'] += [process(F.relu(self.conv4_1(out['p3'][tensor_index])), 'r41')]
            # out['r42'] += [process(F.relu(self.conv4_2(out['r41'][tensor_index])), 'r42')]
            # out['r43'] += [process(F.relu(self.conv4_3(out['r42'][tensor_index])), 'r43')]
            # out['r44'] += [process(F.relu(self.conv4_4(out['r43'][tensor_index])), 'r44')]
            # out['p4']  += [process(self.pool4(out['r44'][tensor_index]), 'p4')]
            # out['r51'] += [process(F.relu(self.conv5_1(out['p4'][tensor_index])), 'r51')]
            # out['r52'] += [process(F.relu(self.conv5_2(out['r51'][tensor_index])), 'r52')]
            # out['r53'] += [process(F.relu(self.conv5_3(out['r52'][tensor_index])), 'r53')]
            # out['r54'] += [process(F.relu(self.conv5_4(out['r53'][tensor_index])), 'r54')]
            # out['p5']  += [process(self.pool5(out['r54'][tensor_index]), 'p5')]

            r11 = F.relu(self.conv1_1(tensor))
            r12 = F.relu(self.conv1_2(r11))
            p1  = self.pool1(r12)
            r21 = F.relu(self.conv2_1(p1))
            r22 = F.relu(self.conv2_2(r21))
            p2  = self.pool2(r22)
            r31 = F.relu(self.conv3_1(p2))
            r32 = F.relu(self.conv3_2(r31))
            r33 = F.relu(self.conv3_3(r32))
            r34 = F.relu(self.conv3_4(r33))
            p3  = self.pool3(r34)
            r41 = F.relu(self.conv4_1(p3))
            r42 = F.relu(self.conv4_2(r41))
            r43 = F.relu(self.conv4_3(r42))
            r44 = F.relu(self.conv4_4(r43))
            p4  = self.pool4(r44)
            r51 = F.relu(self.conv5_1(p4))
            r52 = F.relu(self.conv5_2(r51))
            r53 = F.relu(self.conv5_3(r52))
            r54 = F.relu(self.conv5_4(r53))
            p5  = self.pool5(r54)

            out['r11'] += [process(r11, 'r11')]
            out['r12'] += [process(r12, 'r12')]
            out['p1'] += [process(p1, 'p1')]
            out['r21'] += [process(r21, 'r21')]
            out['r22'] += [process(r22, 'r22')]
            out['p2'] += [process(p2, 'p2')]
            out['r31'] += [process(r31, 'r31')]
            out['r32'] += [process(r32, 'r32')]
            out['r33'] += [process(r33, 'r33')]
            out['r34'] += [process(r34, 'r34')]
            out['p3'] += [process(p3, 'p3')]
            out['r41'] += [process(r41, 'r41')]
            out['r42'] += [process(r42, 'r42')]
            out['r43'] += [process(r43, 'r43')]
            out['r44'] += [process(r44, 'r44')]
            out['p4'] += [process(p4, 'p4')]
            out['r51'] += [process(r51, 'r51')]
            out['r52'] += [process(r52, 'r52')]
            out['r53'] += [process(r53, 'r53')]
            out['r54'] += [process(r54, 'r54')]
            out['p5'] += [process(p5, 'p5')]

            # out['r11'] += [F.relu(self.conv1_1(tensor))]
            # out['r12'] += [F.relu(self.conv1_2(out['r11'][tensor_index]))]
            # out['p1'] += [self.pool1(out['r12'][tensor_index])]
            # out['r21'] += [F.relu(self.conv2_1(out['p1'][tensor_index]))]
            # out['r22'] += [F.relu(self.conv2_2(out['r21'][tensor_index]))]
            # out['p2'] += [self.pool2(out['r22'][tensor_index])]
            # out['r31'] += [F.relu(self.conv3_1(out['p2'][tensor_index]))]
            # out['r32'] += [F.relu(self.conv3_2(out['r31'][tensor_index]))]
            # out['r33'] += [F.relu(self.conv3_3(out['r32'][tensor_index]))]
            # out['r34'] += [F.relu(self.conv3_4(out['r33'][tensor_index]))]
            # out['p3'] += [self.pool3(out['r34'][tensor_index])]
            # out['r41'] += [F.relu(self.conv4_1(out['p3'][tensor_index]))]
            # out['r42'] += [F.relu(self.conv4_2(out['r41'][tensor_index]))]
            # out['r43'] += [F.relu(self.conv4_3(out['r42'][tensor_index]))]
            # out['r44'] += [F.relu(self.conv4_4(out['r43'][tensor_index]))]
            # out['p4'] += [self.pool4(out['r44'][tensor_index])]
            # out['r51'] += [F.relu(self.conv5_1(out['p4'][tensor_index]))]
            # out['r52'] += [F.relu(self.conv5_2(out['r51'][tensor_index]))]
            # out['r53'] += [F.relu(self.conv5_3(out['r52'][tensor_index]))]
            # out['r54'] += [F.relu(self.conv5_4(out['r53'][tensor_index]))]
            # out['p5'] += [self.pool5(out['r54'][tensor_index])]

        # a list of activation pyramids indexed by layer
        result = [out[key] for key in out_keys]

        return result

        # mask
        # for each mip in the pyramid
        # for each vgg layer in the mip





# gram matrix and loss
class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        # print(2.1, F.size(), F.transpose(1,2).size())
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class RegionGramMSELoss(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    def forward(self, input, target, mask):
        # note: this doesn't yield good results.  I made some wrong assumptions here.

        # cut input into sectors, and take gram MSE loss of each.
        # to avoid hotspots of high loss

        chonk = torch.chunk(input, 2, dim=2)
        s1, s2 = torch.chunk(chonk[0], 2, dim=3)
        s3, s4 = torch.chunk(chonk[1], 2, dim=3)

        s1 = s1.contiguous()
        s2 = s2.contiguous()
        s3 = s3.contiguous()
        s4 = s4.contiguous()

        s1_a = torch.sub(GramMatrix()(s1), target)
        s1_c = torch.pow(s1_a, 2)
        s1_d = torch.mean(s1_c)

        s2_a = torch.sub(GramMatrix()(s2), target)
        s2_c = torch.pow(s2_a, 2)
        s2_d = torch.mean(s2_c)

        s3_a = torch.sub(GramMatrix()(s3), target)
        s3_c = torch.pow(s3_a, 2)
        s3_d = torch.mean(s3_c)

        s4_a = torch.sub(GramMatrix()(s4), target)
        s4_c = torch.pow(s4_a, 2)
        s4_d = torch.mean(s4_c)

        # # idea 6: combine region of highest loss with global.  idea is to mostly use global,
        # # but avoid hotspots of unresolved high loss.
        # s1_d_f = s1_d.tolist()
        # s2_d_f = s2_d.tolist()
        # s3_d_f = s3_d.tolist()
        # s4_d_f = s4_d.tolist()
        # d = {s1_d_f:s1_d, s2_d_f:s2_d, s3_d_f:s3_d, s4_d_f:s4_d}
        # max_d_f = max(s1_d_f, s2_d_f, s3_d_f, s4_d_f)
        # a_ = torch.sub(GramMatrix()(input), target)
        # b_ = torch.pow(a_, 2)
        # c_ = torch.mean(b_)
        # return torch.mean(torch.stack((d[max_d_f], c_)))

        # # idea 5: combine with region gramMSELoss with global gramMSELoss?
        # a_ = torch.sub(GramMatrix()(input), target)
        # b_ = torch.pow(a_, 2)
        # c_ = torch.mean(b_)
        # __ = torch.mean(torch.stack((s1_d, s2_d, s3_d, s4_d)))
        # return torch.mean(torch.stack((c_, __)))

        # idea 4: ?
        # return torch.mean(torch.stack((s1_d, s2_d, s3_d, s4_d)))
        # same result as 3

        # idea 3:
        # stitch regions back together before doing MS - does this match original GramMSE?
        # a = torch.cat((s1_a, s2_a), dim=1)
        # b = torch.cat((s3_a, s4_a), dim=2)
        # _, x_, y_ = a.size()
        # a_ = a.reshape(1, x_ * y_)
        # b_ = b.reshape(1, x_ * y_)
        # ab = torch.cat((a_, b_), dim=1)
        # ab_c = torch.pow(ab, 2)
        # ab_d = torch.mean(ab_c)
        # return ab_d

        # ab = torch.stack((a, b), dim=1)
        # print(1.7, ab.size)

        # idea 1: return region of max loss?
        # s1_d_f = s1_d.tolist()
        # s2_d_f = s2_d.tolist()
        # s3_d_f = s3_d.tolist()
        # s4_d_f = s4_d.tolist()
        # d = {s1_d_f:s1_d, s2_d_f:s2_d, s3_d_f:s3_d, s4_d_f:s4_d}
        # max_d_f = max(s1_d_f, s2_d_f, s3_d_f, s4_d_f)
        # return d[max_d_f]

        # idea 2:
        # make a 2x2 tensor of these values, return mean square (i.e. penalise greater loss)
        # lt = torch.zeros((2, 2))

        # cuda_device = 'cuda:%s' % torch.cuda.current_device()
        # device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

        # lt = lt.detach().to(device)
        # # lt[0][0] = s1_d
        # # lt[0][1] = s2_d
        # # lt[1][0] = s3_d
        # # lt[1][1] = s4_d
        #
        # lt[0][0] = torch.mean(s1_a)
        # lt[0][1] = torch.mean(s2_a)
        # lt[1][0] = torch.mean(s3_a)
        # lt[1][1] = torch.mean(s4_a)
        #
        # _ = torch.pow(lt, 2)
        # __ = torch.mean(_)
        # # __ = torch.mean(lt)
        # return __


class MipMSELoss(nn.Module):
    def forward(self, input, target, mip_weights):
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


class MipGramMSELoss01(nn.Module):
    def forward(self, input, target, mip_weights):
        opt_layer_activation_pyramid = input
        target_layer_gram_pyramid = target
        loss = 0

        for index, target_gram in enumerate(target_layer_gram_pyramid):
            mip_weight = mip_weights[index]
            opt_activations = opt_layer_activation_pyramid[index]
            opt_gram = GramMatrix()(opt_activations)
            # utils.write_gram(opt_gram)
            a_ = torch.sub(opt_gram, target_gram)
            b_ = torch.pow(a_, 2)
            c_ = torch.mean(b_)
            c_ *= mip_weight
            loss += c_

        return loss


class MipGramMSELoss(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    def forward(self, input, target, scale, layer, vgg):
        # print(input.size(), target.size(), scale, layer)

        # randomly create scale between values of 1.0 and 1./6.
        # scale = float(torch.FloatTensor(1).uniform_(0.25, 1.0)[0])

        # to do: apply a gaussian blur before downsampling [3] https://wxs.ca/research/multiscale-neural-synthesis/Snelgrove-multiscale-texture-synthesis.pdf
        # if I'm understanding this correctly, this involve the target not being the activations for a single mip,
        # but rather a tensor holding the activations for all the mips for a layer (blurring them before downsampling).
        # we would then also have to create blurred downsampled mips of the optimisation image to calculate loss against.

        # todo: randomly jitter scale within a range
        variation = 0.001
        scale = float(torch.FloatTensor(1).uniform_(scale-variation, scale+variation)[0])

        # todo: try the same, but upscaling the target instead of downsampling the input
        # this would mean the loss fn gets both the opt and target tensors without
        # any mip level or rescaling applied.
        # target_ = F.interpolate(target, scale_factor=1./scale, mode='nearest')

        # todo: try interpolating the layer activations themselves]]

        # note: doesn't matter if the dimensions of the input and ungrammed target
        # match - as long as they are close.  once grammed they are the same size.
        input_ = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)
        # input_ = F.interpolate(input, scale_factor=scale, mode='nearest')
        input_activations = vgg(input_, [layer])[0]
        # input_activations = vgg(input, [layer])[0]

        # the principle of what we're doing here, is to apply the same scale factor
        # to the optimisation image as we are to the style image.  i.e. we are scaling
        # the style into various mips to exploit how the various vgg layers will be activated
        # from this scaling.  eg: r41 is not well activated by 4k images, but is well
        # activated by 512sq images.
        #
        # if we don't scale the opt tensor, we have a problem in that the 2k opt tensor
        # itself will not be able to activate higher layers well.

        # g = GramMatrix()(input_activations)
        # print(g.size())

        # a_ = torch.sub(GramMatrix()(input_activations), target_)
        a_ = torch.sub(GramMatrix()(input_activations), target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        return c_


class GramMSELoss(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    # def forward(self, input, target, mask):
    def forward(self, input, target):

        # homebrew MSE loss, matches nn.MSELoss():
        a_ = torch.sub(GramMatrix()(input), target)

        # print(input.size(), target.size(), a_.size())

        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        return c_

        # black box MSE loss from api (identical results):
        # out = nn.MSELoss()(GramMatrix()(input), target)
        # return (out)


class GramMSELoss2(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    def forward(self, input, target, mask, mask2):
        # print(input.shape)
        # print(target.shape)
        print(5, input.shape)

        b_, c_, w_, h_ = target.size()
        masked_target = target.clone()

        for i in range(0, c_):
            masked_target[0][i] *= mask2

        out = nn.MSELoss()(GramMatrix()(input), masked_target)
        return (out)


# class MaskedGramMSELoss(nn.Module):
#     """
#     MSE = Mean Squared Error
#     https://pytorch.org/docs/stable/nn.html#mseloss
#     """
#     def forward(self, input, target, mask):
#
#         print(3, input.shape)
#
#         b, c, w, h = input.size()
#         masked_input = input.clone()
#
#         # doing this every iteration is quite performance intensive:
#         for i in range(0, c):
#             masked_input[0][i] *= mask
#
#         input_gram = GramMatrix()(masked_input)
#         out = nn.MSELoss()(input_gram, target)
#         return (out)


class MaskedGramMSELoss2(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    def forward(self, input, target, mask):

        a_ = torch.sub(GramMatrix()(input), target)

        # same issue, this would require mask to be a multidimensional
        # tensor to match the vgg layer shape:
        b_ = torch.mul(a_, GramMatrix()(mask))

        c_ = torch.pow(b_, 2)
        d_ = torch.mean(c_)
        return d_

        # out = nn.MSELoss()(input_gram, target)
        # return (out)


class MaskedGramMSELoss3(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss

    Apply a secondary mask to the style image?
    """
    def forward(self, input, target, mask, mask2):

        b_, c_, w_, h_ = target.size()
        masked_target = target.clone()

        for i in range(0, c_):
            masked_target[0][i] *= mask2

        b, c, w, h = input.size()
        masked_input = input.clone()

        # doing this every iteration is quite performance intensive:
        for i in range(0, c):
            masked_input[0][i] *= mask

        input_gram = GramMatrix()(masked_input)
        out = nn.MSELoss()(input_gram, masked_target)
        return (out)


class MaskedGramMSELoss(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss

    Apply a secondary mask to the style image?
    """

    def forward(self, input, target, mask):

        b, c, w, h = input.size()
        masked_input = input.clone()

        # test: init a tensor of same size with values
        # apply via hadamard without loss weight, and see what
        # values visualy match
        # then derive

        # cuda_device = 'cuda:%s' % torch.cuda.current_device()
        # t_ = torch.Tensor(w, h).detach().to(torch.device(cuda_device))
        # t_.fill_(0.5)
        # tn_ = (w*h) / t_.sum()
        # tw_ = torch.mul(t_, tn_)
        # for i in range(0, c):
        #     masked_input[0][i] *= tw_

        for i in range(0, c):
            masked_input[0][i] *= mask

        # for i in range(0, c):
        #     masked_input[0][i] /= mask

        # input_gram = GramMatrix()(input)
        input_gram = GramMatrix()(masked_input)
        # out = nn.MSELoss()(input_gram, target)
        # return (out)

        # Gatys constant layer weights effectively applies a multiplier to the
        # scalar loss here...which yields different results to applying same
        # loss element-wise to mask via hadamard product.
        #
        # so in Gatys approach it's:
        # 0. Gram of both input and target
        # 1. elementwise subtraction
        # 2. elementwise squared
        # 3. mean of tensor
        # 4. scalar multiplier of mean
        #
        # my masking approach, without the loss weighting is:
        # 0. Gram of target
        # 1. elementwise multiplier of constant in input
        # 2. gram of 1.
        # 3. elementwise subtraction
        # 4. elementwise squared
        # 5. mean of tensor

        # could we use mask sum as weight?

        # MSE loss:
        a_ = torch.sub(input_gram, target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        # return (c_)

        # apply scalar weight ala Gatys
        # d_ = torch.mul(c_, mask.mean()) # apply scalar weight
        # d_ = torch.mul(c_, 0.01526) # apply scalar weight
        d_ = torch.mul(c_, 0.01) # apply scalar weight
        return (d_)






class MSELoss(nn.Module):

    # def forward(self, input, target, mask):
    def forward(self, input, target):

        # print(1, input.shape)

        a_ = torch.sub(input, target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        return c_

        # out = nn.MSELoss()(input, target)
        # return out


class CustomMSELoss(nn.Module):

    def forward(self, input, target):

        # https://pytorch.org/docs/stable/torch.html# math-operations
        # torch.pow(2, thing)

        # I believe if we always use torch maths functions, autograd
        # should be able to work here:
        a_ = torch.sub(input, target)
        b_ = torch.pow(a_, 2)
        c_ = torch.mean(b_)
        return c_

        # torch.Size([1, 512, 64, 64])

        # mse_loss = ((input - target) ** 2).torch.mean()
        # return mse_loss


class CustomMSELoss2(nn.Module):

    def forward(self, input, target, mask):

        # https://pytorch.org/docs/stable/torch.html# math-operations
        # torch.pow(2, thing)

        a_ = torch.sub(input, target)

        # note: to apply the mask like this, it will have to be a multidimensional tensor of the same
        # dimensions of the input/target, to match the vgg layer at hand.  this will also use up
        # vram.  however unsure autograd can differentiate if I iterate through each filter and mult?
        b_ = torch.mul(a_, mask)

        c_ = torch.pow(2, b_)
        d_ = torch.mean(c_)
        return d_

        # mse_loss = ((input - target) ** 2).torch.mean()
        # return mse_loss





#
#
#
# class CustomMSELoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, y, y_pred):
#         ctx.save_for_backward(y, y_pred)
#         return (y_pred - y).pow(2).sum()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         yy, yy_pred = ctx.saved_tensors
#         grad_input = torch.neg(2.0 * (yy_pred - ctx.y))
#         return grad_input, grad_output