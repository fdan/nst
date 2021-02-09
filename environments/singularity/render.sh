#! /usr/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/dan/git/nst/python
export NST_VGG_MODEL='/home/dan/git/nst/bin/Models/vgg_conv.pth'

singularity exec --nv --home /home/dan/git:/home/dan/git nst.sif /home/dan/git/nst/bin/nst --style /home/dan/git/nst-temporal/style/034_c.png --content /home/dan/git/nst-temporal/renders/comp/cv_075_comp_v003_512k.####.png --mask /home/dan/git/nst-temporal/renders/masks/17f.####.exr --opt /home/dan/git/nst-temporal/renders/optImage/opt_v001.####.tif --out /home/dan/git/nst-temporal/renders/output/034c/v005/034c_v005.####.png --frames 1038:1039 --iterations 500 --cweights 1.0 --clayers r42 --slayers r12:r21:r31:r34:r41:r42:r43 --sweights 0.2:0.02:0.02:0.02:0.2:0.02:0.02
