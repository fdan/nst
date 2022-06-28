#! /usr/bin/bash

export TRACTOR_ENGINE=frank:5600
export NST_HOME=/mnt/ala/research/danielf/git/nst
export PYTHONPATH=$NST_HOME/python:/mnt/ala/research/danielf/tractor/python
export NST_SIF=/home/13448206/git/nst/python/nst/nuke/environment/singularity/nst_nuke.sif
export NST_VGG_MODEL=$NST_HOME/bin/Models/vgg_conv.pth

singularity shell --nv --bind /mnt/ala $NST_SIF
