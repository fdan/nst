import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

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

    def forward(self, tensor_pyramid, out_keys, mask=torch.zeros(0)):
        """
        :param out_keys: [str]
        :return: [torch.Tensor]
        """
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

        def process(layer_activations, layer_name):
            b, c, w, h = layer_activations.size()

            if layer_name not in out_keys:
                return layer_activations

            if mask.numel() == 0:
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

        # a list of activation pyramids indexed by layer
        result = [out[key] for key in out_keys]

        return result
