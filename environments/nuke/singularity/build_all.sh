#! /usr/bin/bash

sudo singularity build /scratch/nst_nuke.sif nst_nuke.def
mv /scratch/nst_nuke.sif .
