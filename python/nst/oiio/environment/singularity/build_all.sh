#! /usr/bin/bash

sudo singularity build /scratch/nst_oiio.sif nst_oiio.def
mv /scratch/nst_oiio.sif .
