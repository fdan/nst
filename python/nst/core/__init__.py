import json
import os
import torch


class TorchStyle(object):
    def __init__(self, tensor, alpha=torch.zeros(0), target_map=torch.zeros(0)):
        self.tensor = tensor
        self.alpha = alpha
        self.target_map = target_map
        self.scale = 1.0


class Image(object):
    def __init__(self) -> None:
        self.rgb_filepath: str = ''
        self.colorspace: str = 'acescg'


class StyleImage(Image):
    def __init__(self) -> None:
        super(StyleImage, self).__init__()
        self.alpha_filepath: str = ''
        self.target_map_filepath: str = ''
        self.colorspace: str = 'srgb_texture'


class BaseSettings(object):
    def __init__(self):
        return

    def __repr__(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=4)

    def save(self, fp):
        pardir = os.path.abspath(os.path.join(fp, os.path.pardir))
        os.makedirs(pardir, exist_ok=True)
        with open(fp, 'w') as outfile:
            json.dump(json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=4), outfile)

    def load(self, fp):
        with open(fp) as json_file:
            data = json.load(json_file)
        self.__dict__ = json.loads(data)


class NstSettings(BaseSettings):
    def __init__(self):
        super(NstSettings, self).__init__()
        self.engine = 'gpu'
        self.cuda = True
        self.model_path = os.getenv('NST_VGG_MODEL')
        self.optimiser_name = 'adam'
        self.cuda_device = 0
        self.pyramid_scale_factor = 0.63
        self.style_mips = 4
        self.style_layers = ['p1', 'p2', 'r31', 'r42']
        self.style_layer_weights = [1.0, 1.0, 1.0, 1.0]
        self.content_layer = 'r41'
        self.content_layer_weight = 1.0
        self.style_mip_weights = [1.0, 1.0, 1.0, 1.0]
        self.content_mips = 1
        self.optimiser = 'adam'
        self.learning_rate = 1.0
        self.scale = 1.0
        self.iterations = 500
        self.log_iterations = 20


class WriterSettings(BaseSettings):
    def __init__(self) -> None:
        super(WriterSettings, self).__init__()
        self.styles = [StyleImage()]
        self.content = Image()
        self.opt_image = Image()
        self.out: str = ''
        self.progressive_output = False
        self.progressive_ext = 'jpg'
        self.write_gradients = False
        self.gradient_ext = 'jpg'
        self.write_style_pyramid = False
        self.write_style_activations = False
        self.write_content_activations = False
        self.write_gram = False
        self.frame = ''
        self.out_colorspace = 'srgb_texture'
        self.core = NstSettings()