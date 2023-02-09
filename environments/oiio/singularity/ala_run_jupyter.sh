#! /usr/bin/bash

export TRACTOR_ENGINE=frank:5600
export NST_TEMPORAL_HOME=/mnt/ala/research/danielf/2021/git/nst-temporal
export NST_HOME=/mnt/ala/research/danielf/git/nst
export PYTHONPATH=$NST_HOME/python:/mnt/ala/research/danielf/tractor/python
export NST_SIF=$NST_HOME/environments/oiio/singularity/nst_oiio.sif
export NST_VGG_MODEL=$NST_HOME/bin/Models/vgg_conv.pth

singularity exec --nv --bind /mnt/ala --env MPLCONFIGDIR=/tmp/mpl $NST_SIF jupyter lab --notebook-dir $NST_TEMPORAL_HOME --allow-root


