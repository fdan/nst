import os
from typing import List

from models.baseModel import BaseModel

import torch
from torchvision import transforms
from torch.autograd import Variable

from nst.core.model import Nst, TorchStyle, TorchMaskedImage
import nst.core.utils as core_utils
from nst.settings import Image, StyleImage, NstSettings, WriterSettings

class Model(BaseModel):

    def __init__(self):
        super(Model, self).__init__()

        print('nst model constructor called')

        self.name = "Neural Style Transfer"

        self.options = ("engine",
                        "optimiser",
                        "style_pyramid_span",
                        "style_zoom",
                        "gram_weight",
                        "histogram_weight",
                        "histogram_bins",
                        "tv_weight",
                        "laplacian_weight",
                        "histogram_loss_type",
                        "gram_loss_type",
                        "content_loss_type",
                        "laplacian_loss_type",
                        "style_mips",
                        "style_mip_weights",
                        "style_layers",
                        "style_layer_weights",
                        "content_layer",
                        "content_layer_weight",
                        "temporal_weight",
                        "learning_rate",
                        "iterations",
                        "log_iterations",
                        # "enable_update",
                        "batch_size",
                        "dummy",
                        "mask_layers",
                        "random_rotate_mode",
                        "random_rotate",
                        "random_crop",
                        )

        self.inputs = {'opt_img': 3,
                       'style': 4,
                       'style_target': 1,
                       'content': 3,
                       'temporal_content': 4}

        self.outputs = {'output': 3}

        # option states
        self.engine = '\"gpu\"'
        self.optimiser = '\"lbfgs\"'
        self.histogram_loss_type = '\"mse\"'
        self.gram_loss_type = '\"mse\"'
        self.content_loss_type = '\"mse\"'
        self.laplacian_loss_type = '\"mse\"'
        self.style_mip_weights = '\"1.0,1.0,1.0,1.0\"'
        self.style_layers = '\"p1,p2,r31,r42\"'
        self.mask_layers = '\"p1,p2,r31,r42\"'
        self.style_layer_weights = '\"1.0,1.0,1.0,1.0\"'
        self.content_layer = '\"r41\"'

        self.style_mips = 4
        self.style_zoom = 1.0
        self.gram_weight = 1.0
        self.histogram_weight = 10000.0
        self.histogram_bins = 256
        self.tv_weight = 5.0
        self.laplacian_weight = 1.0
        self.style_pyramid_span = 0.5
        self.content_layer_weight = 1.0
        self.temporal_weight = 1.0
        self.learning_rate = 1.0
        # self.iterations = 200
        # self.log_iterations = 10
        self.iterations = 100
        self.log_iterations = 20

        self.random_rotate_mode = "none"
        self.random_rotate = "0, 360"
        self.random_crop = "0.3, 1.0"

        # internal
        # self.batch_size = 200
        self.batch_size = 20
        self.dummy = 1
        self.prepared = False

        self.nst_settings = NstSettings()
        self.last_result = torch.zeros(0)

    def set_iterations(self, iterations):
        self.iterations = iterations
        self.nst.settings.iterations = iterations

    def prepare(self, image_list):

        if self.prepared:
            print('mid-job, skipping preparation')
            return

        print('preparing nst model')
        # print('optimiser:', self.optimiser)

        self.nst = Nst()

        self.nst_settings.cuda = True if self.engine == "gpu" else False
        self.nst_settings.engine = self.engine
        self.nst_settings.model_path = os.getenv('NST_VGG_MODEL')
        self.nst_settings.optimiser = self.optimiser
        self.nst_settings.style_pyramid_span = float(self.style_pyramid_span)
        self.nst_settings.style_zoom = float(self.style_zoom)
        self.nst_settings.gram_weight = float(self.gram_weight)
        self.nst_settings.histogram_weight = float(self.histogram_weight)
        self.nst_settings.histogram_bins = int(self.histogram_bins)
        self.nst_settings.tv_weight = float(self.tv_weight)
        self.nst_settings.laplacian_weight = float(self.laplacian_weight)
        self.nst_settings.histogram_loss_type = self.histogram_loss_type
        self.nst_settings.gram_loss_type = self.gram_loss_type
        self.nst_settings.content_loss_type = self.content_loss_type
        self.nst_settings.laplacian_loss_type = self.laplacian_loss_type
        self.nst_settings.style_mips = int(self.style_mips)

        self.nst_settings.mip_weights = [float(x) for x in self.style_mip_weights.split(',')]
        self.nst_settings.style_layers = self.style_layers.split(',')
        self.nst_settings.mask_layers = self.mask_layers.split(',')
        self.nst_settings.style_layer_weights = [float(x) for x in self.style_layer_weights.split(',')]

        self.nst_settings.random_rotate_mode = self.random_rotate_mode
        self.nst_settings.random_rotate = [float(x) for x in self.random_rotate.split(',')]
        self.nst_settings.random_crop = [float(x) for x in self.random_crop.split(',')]

        # self.nst_settings.mip_weights = self.style_mip_weights
        # self.nst_settings.style_layers = self.style_layers
        # self.nst_settings.mask_layers = self.mask_layers
        # self.nst_settings.style_layer_weights = self.style_layer_weights

        self.nst_settings.content_layer = self.content_layer
        self.nst_settings.content_layer_weight = float(self.content_layer_weight)
        self.nst_settings.temporal_weight = float(self.temporal_weight)
        self.nst_settings.learning_rate = float(self.learning_rate)
        self.nst_settings.iterations = int(self.iterations)
        self.nst_settings.log_iterations = int(self.log_iterations)

        self.nst.settings = self.nst_settings

        if self.engine == "gpu":
            cuda = True
        else:
            cuda = False

        # handle no content scenario
        try:
            content_np = image_list[3]
            content_tensor = torch.Tensor(content_np.copy())
            content_tensor = color_in(content_tensor, do_cuda=cuda)
            self.nst.content = content_tensor
        except:
            self.nst.content = torch.zeros(0)

        if self.temporal_weight > 0:
            try:
                # print(1.1, len(image_list))
                temporal_content_np = image_list[4]
                temporal_content_tensor = torch.Tensor(temporal_content_np.copy())
                temporal_content_tensor = temporal_content_tensor.transpose(0, 2)
                temporal_content_tensor = temporal_content_tensor[:3:]
                temporal_content_tensor = temporal_content_tensor.transpose(0, 2)
                temporal_content_tensor = color_in(temporal_content_tensor, do_cuda=cuda)

                temporal_content_alpha_tensor = torch.Tensor(temporal_content_np.copy())
                temporal_content_alpha_tensor = temporal_content_alpha_tensor.transpose(0, 2)
                temporal_content_alpha_tensor[0] = temporal_content_alpha_tensor[3]
                temporal_content_alpha_tensor[1] = temporal_content_alpha_tensor[3]
                temporal_content_alpha_tensor[2] = temporal_content_alpha_tensor[3]
                temporal_content_alpha_tensor = temporal_content_alpha_tensor[:3:]
                temporal_content_alpha_tensor = temporal_content_alpha_tensor.transpose(0, 2)
                temporal_content_alpha_tensor = color_in(temporal_content_alpha_tensor, do_cuda=cuda, raw=True)

                # temporal_content = TorchMaskedImage(temporal_content_tensor, temporal_content_alpha_tensor)
                self.nst.temporal_content = temporal_content_tensor
                # self.nst.temporal_weight_mask = temporal_content_alpha_tensor

            except:
                print("something went wrong with the temporal content inputs...")
                self.nst.temporal_content = torch.zeros(0)
                # self.nst.temporal_weight_mask = torch.zeros(0)

        try:
            opt_np = image_list[0]
            opt_tensor = torch.Tensor(opt_np.copy())
            opt_tensor = color_in(opt_tensor, do_cuda=cuda)
            opt_tensor = Variable(opt_tensor.data.clone(), requires_grad=True)
            self.nst.opt_tensor = opt_tensor
        except:
            raise RuntimeError("An optimisation image must be given")


        try:
            style_np = image_list[1]
            style_tensor = torch.Tensor(style_np.copy())
            style_tensor = style_tensor.transpose(0, 2)
            style_tensor = style_tensor[:3:]
            style_tensor = style_tensor.transpose(0, 2)
            style_tensor = color_in(style_tensor, do_cuda=cuda)

            style_alpha_tensor = torch.Tensor(style_np.copy())
            style_alpha_tensor = style_alpha_tensor.transpose(0, 2)
            style_alpha_tensor[0] = style_alpha_tensor[3]
            style_alpha_tensor[1] = style_alpha_tensor[3]
            style_alpha_tensor[2] = style_alpha_tensor[3]
            style_alpha_tensor = style_alpha_tensor[:3:]
            style_alpha_tensor = style_alpha_tensor.transpose(0, 2)
            style_alpha_tensor = color_in(style_alpha_tensor, do_cuda=cuda, raw=True)

            style = TorchStyle(style_tensor, style_alpha_tensor)
            self.style = style

            try:
                # todo: handle scenario where no target is given
                style_target_np = image_list[2]
                style_target_tensor = torch.Tensor(style_target_np.copy())
                style_target_tensor = color_in(style_target_tensor, do_cuda=cuda, raw=True)
                style.target_map = style_target_tensor
            except:
                pass

            self.nst.styles.append(style)
        except:
            raise RuntimeError("There must be at least one style image")

        self.nst.prepare()

        print("finished preparing")
        print('-----------------------------------------------')
        print('starting job')

        self.prepared = True

    def inference(self) -> List[torch.Tensor]:
        # if not self.enable_update:
        #     print('update disabled')
        #     return # to do: return the opt iamge

        self.nst.start_iter = 1

        result = self.nst()
        result_np = color_out(result)
        return result_np


def color_in(tensor, do_cuda: bool, raw: bool=False) -> torch.Tensor:

    tforms_ = []

    #  turn to BGR
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    if not raw:
        # subtract imagenet mean
        tforms_ += [transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1])]

        # scale to imagenet values
        tforms_ += [transforms.Lambda(lambda x: x.mul_(255.))]

    tforms = transforms.Compose(tforms_)

    # note: oiio implicitely converts to 0-1 floating point data here regardless of format:
    tensor = torch.transpose(tensor, 2, 0)
    tensor = torch.transpose(tensor, 2, 1)

    tensor = tforms(tensor)

    if do_cuda:
        device = core_utils.get_cuda_device()
        tensor = tensor.detach().to(torch.device(device))
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)


def color_out(tensor):
    tforms_ = []

    tforms_ += [transforms.Lambda(lambda x: x.mul_(1. / 255.))]

    # add imagenet mean
    tforms_ += [transforms.Normalize(mean=[(-0.40760392), -0.45795686, -0.48501961], std=[1, 1, 1])]

    # turn to RGB
    tforms_ += [transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])]

    tforms = transforms.Compose(tforms_)

    tensor_ = torch.clone(tensor)
    t = tforms(tensor_.data[0].cpu().squeeze())

    t = torch.transpose(t, 2, 0)
    t = torch.transpose(t, 0, 1)
    t = t.contiguous()
    return t.numpy()
