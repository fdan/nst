#! /usr/bin/bash

singularity exec --nv --bind /mnt/ala nst5.sif python -c 'import torch;torch.cuda.is_available()'
