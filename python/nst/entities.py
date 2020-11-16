import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):

    layers = {}
    layers['r11'] = {'channels': 64, 'x': 512}
    layers['r12'] = {'channels': 64, 'x': 512}
    layers['r21'] = {'channels': 128, 'x': 256}
    layers['r22'] = {'channels': 128, 'x': 256}
    layers['r31'] = {'channels': 256, 'x': 128}
    layers['r32'] = {'channels': 256, 'x': 128}
    layers['r34'] = {'channels': 256, 'x': 128}
    layers['r41'] = {'channels': 512, 'x': 64}
    layers['r42'] = {'channels': 512, 'x': 64}
    layers['r43'] = {'channels': 512, 'x': 64}
    layers['r44'] = {'channels': 512, 'x': 64}
    layers['r51'] = {'channels': 512, 'x': 32}
    layers['r52'] = {'channels': 512, 'x': 32}
    layers['r53'] = {'channels': 512, 'x': 32}
    layers['r54'] = {'channels': 512, 'x': 32}

    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules

        # note: first two args of Conv2d are in channels, out channels
        # where is the x and y dimensions of each filter defined?
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, tensor, out_keys):
        """
        :param tensor: torch.Tensor
        :param out_keys: [str]
        :return: [torch.Tensor]
        """
        out_path = '/mnt/ala/research/danielf/nst/experiments/output/temp14/'

        out = {}

        out['r11'] = F.relu(self.conv1_1(tensor))
        # print(out['r11'].size(), out['r11'].dim())
        # layer_out = out_path + 'layer_r11.png'
        # utils.render_image(tensor, layer_out)

        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])

        out['r21'] = F.relu(self.conv2_1(out['p1']))
        # layer_out = out_path + 'layer_r21.png'
        # utils.render_image(tensor, layer_out)

        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])

        out['r31'] = F.relu(self.conv3_1(out['p2']))
        # layer_out = out_path + 'layer_r31.png'
        # utils.render_image(tensor, layer_out)

        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])

        out['r41'] = F.relu(self.conv4_1(out['p3']))
        # layer_out = out_path + 'layer_r41.png'
        # utils.render_image(tensor, layer_out)

        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])

        out['r51'] = F.relu(self.conv5_1(out['p4']))
        # layer_out = out_path + 'layer_r51.png'
        # utils.render_image(tensor, layer_out)

        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        result = [out[key] for key in out_keys]
        return result


# gram matrix and loss
class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
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
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


class GramMSELoss(nn.Module):
    """
    MSE = Mean Squared Error
    https://pytorch.org/docs/stable/nn.html#mseloss
    """
    # def forward(self, input, target, mask):
    def forward(self, input, target):
        # homebrew MSE loss, matches nn.MSELoss():
        a_ = torch.sub(GramMatrix()(input), target)
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

        # t_ = torch.Tensor(w, h).detach().to(torch.device("cuda:0"))
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