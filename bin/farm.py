#! /usr/bin/env python

from optparse import OptionParser
import os
from pprint import pprint
import glob

import nst_farm


def main():
    """
    usage:

    nst --style ../style/034_c.png
    --content ../renders/comp/cv_075_comp_v003_512k.*.png
    --mask ../renders/masks/17f.*.tif
    --opt ../renders/optImage/opt_v001.*.tif
    --out ../renders/output/034c/v001/034c_v001.*.png
    --frames 1038:1048 --iterations 500 --cweights 1.0 --clayers r42
    --slayers r12:r21:r31:r34:r41:r42:r43 --sweights 0.2:0.02:0.02:0.02:0.2:0.02:0.02
    """

    p = OptionParser()
    p.add_option("", "--from-content", dest='from_content', action='store_true')
    p.add_option("", "--style", dest='style', action='store')
    p.add_option("", "--content", dest='content', action='store')
    p.add_option("", "--mask", dest='mask', action='store')
    p.add_option("", "--opt", dest='opt', action='store')
    p.add_option("", "--out", dest='out', action='store')
    p.add_option("", "--frames", dest='frames', action='store')
    p.add_option("", "--iterations", dest='iterations', action='store')
    p.add_option("", "--clayers", dest='clayers', action='store')
    p.add_option("", "--cweights", dest='cweights', action='store')
    p.add_option("", "--slayers", dest='slayers', action='store')
    p.add_option("", "--sweights", dest='sweights', action='store')
    p.add_option("", "--smasks", dest='smasks', action='store')

    opts, args = p.parse_args()

    # check_outputs(opts.out)

    inputs_to_check = []

    if opts.opt:
        inputs_to_check.append(opts.opt)
    if opts.mask:
        inputs_to_check.append(opts.mask)
    if opts.content:
        inputs_to_check.append(opts.content)

    check_inputs(opts.style, inputs_to_check)

    nst_farm.conda.doit(opts)


def check_inputs(style, inputs):
    assert os.path.isfile(style)

    for i in inputs:
        i = i.replace('####', '*')
        try:
            assert len(glob.glob(i)) > 0
        except:
            print('invalid input:', i)


if __name__ == "__main__":
    main()

"""
nst --style ../style/034_c.png --content ../renders/comp/cv_075_comp_v003_512k.####.png --mask ../renders/masks/17f.####.tif --opt ../renders/optImage/opt_v001.####.tif --out ../renders/output/034c/v001/034c_v001.####.png --frames 1038:1048 --iterations 500 --cweights 1.0 --clayers r42 --slayers r12:r21:r31:r34:r41:r42:r43 --sweights 0.2:0.02:0.02:0.02:0.2:0.02:0.02 
"""