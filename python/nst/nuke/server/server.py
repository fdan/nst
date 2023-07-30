# Copyright (c) 2018 Foundry.
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

import torch.multiprocessing as mp
from multiprocessing import Process, Queue
import math
import argparse
import os
import importlib
import socket  # to get machine hostname
import traceback

try:  # python3
    import socketserver
except ImportError:  # python2
    import SocketServer as socketserver

import numpy as np

#from message_pb2 import *
from messageLive_pb2 import *

ML_SERVER_DIR = os.getenv('ML_SERVER_DIR')


class MLTCPServer(socketserver.TCPServer):
    def __init__(self, server_address, handler_class, auto_bind=True):
        self.verbose = False
        model_dir = os.path.join(ML_SERVER_DIR, 'models')
        os.chdir(model_dir)

        self.cancel = False
        # Each directory in models/ containing a model.py file is an available ML model
        # self.available_models = [name for name in next(os.walk('models'))[1]
        #     if os.path.isfile(os.path.join('models', name, 'model.py'))]
        self.available_models = [name for name in next(os.walk(model_dir))[1]
                                 if os.path.isfile(os.path.join(model_dir, name, 'model.py'))]
        self.available_models.sort()
        self.models = {}
        for model in self.available_models:
            print('Importing models.{}.model'.format(model))
            self.models[model] = importlib.import_module('models.{}.model'.format(model)).Model()

        print(self.models['nst'])

        socketserver.TCPServer.__init__(self, server_address, handler_class, auto_bind)
        return


def send_msg(handler, msg):
    handler.vprint('Serializing message')
    s = msg.SerializeToString()
    msg_len = msg.ByteSize()
    totallen = 12 + msg_len
    msg_ = bytes(str(totallen).zfill(12).encode('utf-8')) + s
    handler.vprint('Sending response message of size: {}'.format(totallen))

    totalsent = 0
    while totalsent < msg_len:
        sent = handler.request.send(msg_[totalsent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        totalsent = totalsent + sent

    handler.vprint('-----------------------------------------------')


class ImageProcessTCPHandler(socketserver.BaseRequestHandler):
    """This request handler is instantiated once per connection."""

    def handle(self):
        # Read the data headers
        data_hdr = self.request.recv(12)
        self.vprint('Received data header: {}'.format(data_hdr))
        sz = int(data_hdr)
        self.vprint('Receiving message of size: {}'.format(sz))

        # Read data
        data = self.recvall(sz)
        self.vprint('{} bytes read'.format(len(data)))

        # Parse the message
        req_msg = RequestWrapper()
        req_msg.ParseFromString(data)
        self.vprint('Message parsed')

        # Process message
        self.process_message(req_msg)

    def process_cancel(self, message):
        self.vprint('Cancelling inference')
        self.server.cancel = True

    def process_message(self, message):
        if message.HasField('r1'):
            self.vprint('Received info request')
            self.process_info(message)
        elif message.HasField('r2'):
            self.vprint('Received inference request')
            self.process_inference(message)
        elif message.HasField('r3'):
            self.vprint('Received cancel request')
            self.process_cancel(message)
        else:
            msg = "Server received unindentified request from client."
            send_msg(self, msg)

    def process_info(self, message):
        resp_msg = RespondWrapper()
        resp_msg.info = True
        resp_info = RespondInfo()
        resp_info.num_models = len(self.server.available_models)
        # Add all model info into the message
        for model in self.server.available_models:
            m = resp_info.models.add()
            m.name = model
            m.label = self.server.models[model].get_name()
            # Add inputs
            for inp_name, inp_channels in self.server.models[model].get_inputs().items():
                inp = m.inputs.add()
                inp.name = inp_name
                inp.channels = inp_channels
            # Add outputs
            for out_name, out_channels in self.server.models[model].get_outputs().items():
                out = m.outputs.add()
                out.name = out_name
                out.channels = out_channels
            # Add options
            for opt_name, opt_value in self.server.models[model].get_options().items():
                if type(opt_value) == int:
                    opt = m.int_options.add()
                elif type(opt_value) == float:
                    opt = m.float_options.add()
                elif type(opt_value) == bool:
                    opt = m.bool_options.add()
                elif type(opt_value) == str:
                    opt = m.string_options.add()
                    # TODO: Implement multiple choice
                else:
                    # Send an error response message to the Nuke Client
                    option_error = ("Model option of type {} is not implemented. "
                                    "Broadcasted options need to be one of bool, int, float, str."
                                    ).format(type(opt_value))
                    return self.errormsg(option_error)
                opt.name = opt_name
                opt.values.extend([opt_value])
            # Add buttons
            for button_name, button_value in self.server.models[model].get_buttons().items():
                if type(button_value) == bool:
                    button = m.button_options.add()
                else:
                    return self.errormsg("Model button needs to be of type bool.")
                button.name = button_name
                button.values.extend([button_value])

        # Add RespondInfo message to RespondWrapper
        resp_msg.r1.CopyFrom(resp_info)
        send_msg(self, resp_msg)

    def process_inference(self, message):
        req = message.r2
        m = req.model

        model = self.server.models[m.name]

        self.vprint('Requesting inference on model: {}'.format(m.name))

        # Parse model options
        opt = {}
        for options in [m.bool_options, m.int_options, m.float_options, m.string_options]:
            for option in options:
                opt[option.name] = option.values[0]
        # Set model options
        self.server.models[m.name].set_options(opt)
        # Parse model buttons
        btn = {}
        for button in m.button_options:
            btn[button.name] = button.values[0]
        self.server.models[m.name].set_buttons(btn)

        # determine where we are in a job/batch
        job_progress = float(req.batch_current + 1) / float(req.batch_total) * 100

        # Parse images if in first batch
        if req.clearcache:
            print('clearcache called')
            model.prepared = False
            img_list = []
            for byte_img in req.images:
                img = np.frombuffer(byte_img.image, dtype='<f4')
                height = byte_img.height
                width = byte_img.width
                channels = byte_img.channels
                img = np.reshape(img, (channels, height, width))
                img = np.transpose(img, (1, 2, 0))
                img = np.flipud(img)
                img_list.append(img)
            model.prepare(img_list)
            model.set_iterations(model.batch_size)
        else:
            print('not clearing cache')

        try:
            # Running inference
            self.vprint('Starting inference')

            # main call:
            model = self.server.models[m.name]

            # an issue here is that once inference starts we can't pause and poll
            # for a kill signal.

            # todo: do the inference in a multiproc Process
            result = model.inference()

            #

            # model.share_memory()
            # processes = [1]
            # for rank in range(1):
            #     p = mp.Process(target=model.inference)
            #     p.start()
            #     processes.append(p)
            # for p in processes:
            #     p.join()

            # queue = Queue()
            # process = torch.multiprocessing.Process(target=model.inference)
            # process.start()
            # process.join() # this blocks
            # result = queue.get()


            print('job progress: %d' % job_progress + '%')
            if job_progress == 100:
                print('job complete')
                print('-----------------------------------------------')

            resp_msg = RespondWrapper()
            resp_msg.info = True
            resp_inf = RespondInference()
            img = np.flipud(result)
            image = resp_inf.images.add()
            image.width = np.shape(img)[1]
            image.height = np.shape(img)[0]
            image.channels = np.shape(img)[2]
            img = np.transpose(img, (2, 0, 1))
            image.image = img.tobytes()
            resp_inf.num_images = 1
            resp_msg.r2.CopyFrom(resp_inf)
            self.vprint('sending inteference response')

            # todo: want to call this when next batch is running in subproc
            send_msg(self, resp_msg)
            # p = Process(target=send_msg, args=(self, resp_msg))
            # p.start()
            # if it's the last batch, join to avoid main proc hangup
            # if b == batches-1:
            #     p.join()


        except Exception as e:
            # Pass error message to the client
            self.vprint('Exception caught on inference on model:')
            self.vprint(str(traceback.print_exc()))
            resp_msg = self.errormsg(str(e))
            send_msg(self, resp_msg)

    def recvall(self, n):
        """Helper function to receive node bytes or return None if EOF is hit"""
        data = b''
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def sendmsg(self, msg, msglen):
        totalsent = 0
        while totalsent < msglen:
            sent = self.request.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent = totalsent + sent

    def errormsg(self, error):
        """Create an error message to send a Server error to the Nuke Client"""
        resp_msg = RespondWrapper()
        resp_msg.info = True
        error_msg = Error()  # from message_pb2.py
        error_msg.msg = error
        resp_msg.error.CopyFrom(error_msg)
        return resp_msg

    def vprint(self, string):
        if self.server.verbose:
            print('Server -> ' + string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning inference server.')
    parser.add_argument('port', type=int, help='Port number for the server to listen to.')
    args = parser.parse_args()

    # Get the current hostname of the server
    server_hostname = socket.gethostbyname(socket.gethostname())
    # Create the server
    server = MLTCPServer((server_hostname, args.port),
                         ImageProcessTCPHandler, False)

    # Bind and activate the server
    server.allow_reuse_address = True
    server.server_bind()
    server.server_activate()
    print('Server -> Listening on port: {}'.format(args.port))
    server.serve_forever()
