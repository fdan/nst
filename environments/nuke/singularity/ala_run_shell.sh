#! /usr/bin/bash

export TRACTOR_ENGINE=frank:5600
export NST_HOME=/mnt/ala/research/danielf/git/nst
export PYTHONPATH=$NST_HOME/python
export NST_SIF=$NST_HOME/environments/nuke/singularity/nst_nuke.sif
export NST_VGG_MODEL=$NST_HOME/models/vgg_conv.pth

singularity shell --nv --bind /mnt/ala $NST_SIF
