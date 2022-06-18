from models.baseModel import BaseModel
# from models.common.model_builder import baseline_model
from models.common.util import print_, get_ckpt_list, linear_to_srgb, srgb_to_linear
import message_pb2

from nst2.core.model import Nst

class Model(BaseModel):

    def __init__(self):
        super(Model, self).__init__()
        self.name = "Neural Style Transfer"
        self.nst = Nst()

    def inference(self, *inputs):
        pass

