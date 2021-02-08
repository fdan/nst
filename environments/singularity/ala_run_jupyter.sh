#! /usr/bin/bash

# PYTHONPATH and VGG model path env vars to come from nst rez package

singularity exec --nv --home /mnt/ala:/mnt/ala --env MPLCONFIGDIR=/tmp/mpl nst.sif jupyter lab --notebook-dir /mnt/ala/home/133235/git/nst-temporal --allow-root
