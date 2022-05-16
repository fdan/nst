#! /usr/bin/env python

import subprocess
import sys
import random
import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False


if torch.cuda.is_available():
    print('gpu is available')

    cuda_device = 'cuda:%s' % torch.cuda.current_device()
    print('cuda device:', cuda_device)
    torch.zeros((5, 5, 5, 5)).detach().to(torch.device(cuda_device))

    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader' ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)

    sys.exit(0)
else:
    print('no gpu was found')
    sys.exit(1)




