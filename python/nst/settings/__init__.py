import json
import os
from platform import python_version


class Image(object):
    def __init__(self):
        self.rgb_filepath = ''
        self.colorspace = 'acescg'


class StyleImage(Image):
    def __init__(self):
        super(StyleImage, self).__init__()
        self.rgba_filepath = ''
        self.target_map_filepath = ''
        self.colorspace = 'srgb_texture'
        ## to do
        # self.mips = 4
        # self.mip_weights = [1.0]*4
        # self.layers = ['r31', 'r42']
        # self.layer_weights = [0.01, 0.005]
        # self.pyramid_span = 0.5
        # self.zoom = 1.0
        ##


class BaseSettings(object):
    def __init__(self):
        return

    def __repr__(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=4)

    def save(self, fp):
        pardir = os.path.abspath(os.path.join(fp, os.path.pardir))

        v = python_version().split('.')[0]
        if v == 3:
            os.makedirs(pardir, exist_ok=True)
        else:
            try:
                os.makedirs(pardir)
            except:
                if os.path.isdir(pardir):
                    pass

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
        self.cpu_threads = 48
        self.cuda = True
        self.model_path = os.getenv('NST_VGG_MODEL')
        self.cuda_device = 0
        self.content_layer = 'r41'
        self.content_layer_weight = 1.0
        self.content_mips = 1
        self.optimiser = 'adam'
        self.learning_rate = 10.0
        self.iterations = 500
        self.log_iterations = 20

        ## to do: migrate these to TorchStyle and StyleImage classes
        self.style_zoom = 1.0
        self.style_pyramid_span = 0.5
        self.style_mips = 4
        self.mip_weights = [1.0, 1.0, 1.0, 1.0]
        self.style_layers = ['p1', 'p2', 'r31', 'r42']
        self.style_layer_weights = [1.0, 1.0, 1.0, 1.0]
        ##


class WriterSettings(BaseSettings):
    def __init__(self):
        super(WriterSettings, self).__init__()
        self.styles = [StyleImage()]
        self.content = Image()
        self.opt_image = Image()
        self.out = ''
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

    def load(self, fp):
        with open(fp) as json_file:
            data = json.load(json_file)
            data = json.loads(data)
        self.__dict__ = data

        from pprint import pprint
        pprint(data)

        core = data.get('core')
        self.core = NstSettings()
        self.core.__dict__ = core

        content = data.get('content')
        self.content = Image()
        self.content.__dict__ = content

        opt_image = data.get('opt_image')
        self.opt_image = Image()
        self.opt_image.__dict__ = opt_image

        styles = []
        for style in data.get('styles'):
            style_ = StyleImage()
            style_.__dict__= style
            styles.append(style_)
        self.styles = styles