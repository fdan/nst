#! /usr/bin/bash

export NST_HOME=/mnt/ala/research/danielf/git/nst
export PYTHONPATH=$NST_HOME/python
export NST_SIF=$NST_HOME/environments/nuke/singularity/nst_nuke.sif
export NST_VGG_MODEL=$NST_HOME/models/vgg_conv.pth
export ML_SERVER_DIR=/mnt/ala/research/danielf/git/nst/python/nst/nuke/server

singularity exec --nv --bind /mnt/ala $NST_SIF python3 -m pdb /mnt/ala/research/danielf/git/nst/python/nst/nuke/server/server.py 55555
