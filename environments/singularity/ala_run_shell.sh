#! /usr/bin/bash

# PYTHONPATH and VGG model path env vars to come from nst rez package

singularity shell --nv --home /mnt/ala:/mnt/ala --env MPLCONFIGDIR=/tmp/mpl nst.sif
