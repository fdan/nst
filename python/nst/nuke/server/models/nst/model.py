from models.baseModel import BaseModel
# from models.common.model_builder import baseline_model
from models.common.util import print_, get_ckpt_list, linear_to_srgb, srgb_to_linear
import message_pb2


import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from nst.core.model import Nst, TorchStyle
import nst.core.utils as core_utils

class Model(BaseModel):

    def __init__(self):
        super(Model, self).__init__()
        self.name = "Neural Style Transfer"

        self.options = ("engine",
                        "optimiser",
                        "pyramid_scale_factor",
                        "style_mips",
                        "style_layers",
                        "style_layer_weights",
                        "style_mip_weights",
                        "content_layer",
                        "content_layer_weight",
                        "content_mips",
                        "learning_rate",
                        "scale",
                        "iterations",
                        "log_iterations",

                        # farm related
                        "farm_engine",
                        "farm_optimiser",
                        "content_fp",
                        "style1_fp",
                        "style2_fp",
                        "style3_fp",
                        "style_targets_fp",
                        )

        self.buttons = ("run", "farm")

        self.inputs = {'content': 3,
                       'style1': 4,
                       'style1_target': 1,
                       'style2': 4,
                       'style2_target': 1,
                       'style3': 4,
                       'style3_target': 1}

        self.outputs = {'output': 3}

        # option states
        self.engine = "gpu"
        self.optimiser = "adam"
        self.pyramid_scale_factor = 0.63
        self.style_mips = 5
        self.style_layers = 'p1,p2,r31,r42'
        self.style_layer_weights = '1.0,1.0,1.0,1.0'
        self.style_mip_weights = "1.0,1.0,1.0,1.0,1.0"
        self.content_layer = 'r41'
        self.content_layer_weight = "1.0"
        self.content_mips = 1
        self.learning_rate = 1.0
        self.scale = 1.0
        self.iterations = 500
        self.log_iterations = 1

        # farm related
        self.farm_engine = "cpu"
        self.farm_optimiser = "lbfgs"
        self.content_fp = ""
        self.style1_fp = ""
        self.style1_target_fp = ""
        self.style2_fp = ""
        self.style2_target_fp = ""
        self.style3_fp = ""
        self.style3_target_fp = ""
        self.style_targets_fp = ""

        # button states
        self.run = False
        self.farm = False

        # nst init
        self.nst = Nst()
        self.nst.cuda = True if self.engine == "gpu" else False
        self.nst.engine = self.engine
        self.nst.optimiser = self.optimiser
        self.nst.pyramid_scale_factor = self.pyramid_scale_factor
        self.nst.style_mips = self.style_mips
        self.nst.style_layers = self.style_layers
        self.nst.style_layer_weights = self.style_layer_weights
        self.nst.style_mip_weights = self.style_mip_weights
        self.nst.content_layer = self.content_layer
        self.nst.content_layer_weight = self.content_layer_weight
        self.nst.content_mips = self.content_mips
        self.nst.learning_rate = self.learning_rate
        self.nst.scale = self.scale
        self.nst.iterations = self.iterations
        self.nst.log_iterations = self.log_iterations

    def inference(self, image_list):

        if self.engine == "gpu":
            cuda = True
        else:
            cuda = False

        try:
            content_np = image_list[0]
            content_tensor = torch.Tensor(content_np)
            content_tensor = color_in(content_tensor, do_cuda=cuda)
        except IndexError:
            content_tensor = None
        finally:
            self.nst.content = content_tensor

        try:
            style1_np = image_list[1]
            style1_tensor = torch.Tensor(style1_np).transpose(0, 2)[1::].transpose(0, 2)
            style1_tensor = color_in(style1_tensor, do_cuda=cuda)
            style1_alpha_tensor = torch.Tensor(style1_np).transpose(0, 2)[3].transpose(0, 2)
            style1_alpha_tensor = color_in(style1_alpha_tensor, do_cuda=cuda, raw=True)
            style1 = TorchStyle(style1_tensor, style1_alpha_tensor)
        except IndexError:
            raise RuntimeError("There must be at least one style image")

        try:
            style1_target_np = image_list[2]
            style1_target_tensor = torch.Tensor(style1_target_np)
            style1_target_tensor = color_in(style1_target_tensor, do_cuda=cuda, raw=True)
        except:
            style1_target_tensor = None
        finally:
            style1.target_map = style1_target_tensor
            self.nst.styles.append(style1)


        try:
            style2_np = image_list[3]
            style2_tensor = torch.Tensor(style2_np).transpose(0, 2)[1::].transpose(0, 2)
            style2_tensor = color_in(style2_tensor, do_cuda=cuda)
            style2_alpha_tensor = torch.Tensor(style2_np).transpose(0, 2)[3].transpose(0, 2)
            style2_alpha_tensor = color_in(style2_alpha_tensor, do_cuda=cuda, raw=True)
            style2 = TorchStyle(style2_tensor, style2_alpha_tensor)
        except IndexError:
            style2 = None

        try:
            style2_target_np = image_list[4]
            style2_target_tensor = torch.Tensor(style2_target_np)
            style2_target_tensor = color_in(style2_target_tensor, do_cuda=cuda, raw=True)
        except IndexError:
            style2_target_tensor = None
        finally:
            if style2:
                style2.target_map = style2_target_tensor
                self.nst.styles.append(style2)


        try:
            style3_np = image_list[5]
            style3_tensor = torch.Tensor(style3_np).transpose(0, 2)[1::].transpose(0, 2)
            style3_tensor = color_in(style3_tensor, do_cuda=cuda)
            style3_alpha_tensor = torch.Tensor(style3_np).transpose(0, 2)[3].transpose(0, 2)
            style3_alpha_tensor = color_in(style3_alpha_tensor, do_cuda=cuda, raw=True)
            style3 = TorchStyle(style3_tensor, style3_alpha_tensor)
        except IndexError:
            style3 = None

        try:
            style3_target_np = image_list[6]
            style3_target_tensor = torch.Tensor(style3_target_np)
            style3_target_tensor = color_in(style3_target_tensor, do_cuda=cuda, raw=True)
        except IndexError:
            style3_target_tensor = None
        finally:
            if style3:
                style3.target_map = style3_target_tensor
                self.nst.styles.append(style3)

        self.nst.prepare()
        result_tensor = self.nst()
        result_np = color_out(result_tensor)
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
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])

    # what's this do?
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor.data[0].cpu().squeeze())
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img