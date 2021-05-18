#! /usr/bin/bash

export TRACTOR_ENGINE=frank:5600
export NST_HOME=/mnt/ala/research/danielf/2021/git/nst
export PYTHONPATH=$NST_HOME/python:/mnt/ala/research/danielf/tractor/python
export NST_SIF=$NST_HOME/environments/singularity/pytorch-1.7.1_cuda-11.0/nst.sif
export NST_VGG_MODEL=$NST_HOME/bin/Models/vgg_conv.pth

singularity shell --nv --bind /mnt/ala $NST_SIF
