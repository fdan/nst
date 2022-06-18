# Copyright (c) 2019 Foundry.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from ..baseModel import BaseModel

import cv2
import numpy as np

from ..common.util import linear_to_srgb, srgb_to_linear

import message_pb2

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Gaussian Blur'

        self.kernel_size = 5
        self.make_blur = False

        # Define options
        self.options = ('kernel_size',)
        self.buttons = ('make_blur',)

        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = linear_to_srgb(image)
        image = (image * 255).astype(np.uint8)
        kernel = self.kernel_size * 2 + 1
        blur = cv2.GaussianBlur(image, (kernel, kernel), 0)
        blur = blur.astype(np.float32) / 255.
        blur = srgb_to_linear(blur)
        
        # If make_blur button is pressed in Nuke
        if self.make_blur:
            script_msg = message_pb2.FieldValuePairAttrib()
            script_msg.name = "PythonScript"
            # Create a Python script message to run in Nuke
            python_script = self.blur_script(blur)
            script_msg_val = script_msg.values.add()
            script_msg_str = script_msg_val.string_attributes.add()
            script_msg_str.values.extend([python_script])
            return [blur, script_msg]

        return [blur]

    def blur_script(self, image):
        """Return the Python script function to create a pop up window in Nuke.

        The pop up window displays the brightest pixel position of the given image.
        """
        # Compute brightest pixel of the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        [min_val, max_val, min_loc, max_loc] = cv2.minMaxLoc(gray)
        # Y axis are inversed in Nuke
        max_loc = (max_loc[0], image.shape[0] - max_loc[1])
        popup_msg = (
            "Brightest pixel of the blurred image\\n"
            "Location: {}, Value: {:.3f}."
            ).format(max_loc, max_val)
        script = "nuke.message('{}')\n".format(popup_msg)
        return script