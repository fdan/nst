#! /usr/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/dan/git/nst/python
export NST_VGG_MODEL='/home/dan/git/nst/bin/Models/vgg_conv.pth'

singularity shell --nv nst.sif
