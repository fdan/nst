#! /usr/bin/bash

#export NST_TEMPORAL_HOME=/mnt/ala/research/danielf/2021/nst-temporal
#export NST_HOME=/mnt/ala/research/danielf/2021/nst
#export PYTHONPATH=$PYTHONPATH:/scratch/dan/nst/python
#export NST_VGG_MODEL=


singularity exec --nv --home /scratch/dan/:/scratch/dan/ nst.sif $NST_HOME/bin/nst --style $NST_TEMPORAL_HOME/style/034_c.png --content $NST_TEMPORAL_HOME/renders/comp/cv_075_comp_v003_512k.####.png --mask $NST_TEMPORAL_HOME/renders/masks/17f.####.exr --opt $NST_TEMPORAL_HOME/renders/optImage/opt_v001.####.tif --out $NST_TEMPORAL_HOME/renders/output/034c/v006/034c_v006.####.png --frames 1038:1039 --iterations 500 --cweights 1.0 --clayers r42 --slayers r12:r21:r31:r34:r41:r42:r43 --sweights 0.2:0.02:0.02:0.02:0.2:0.02:0.02
