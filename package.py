# -*- coding: utf-8 -*-

name = 'nst'

version = '0.0.8'

requires = [
            'conda_pytorch',
            'tractor',
            'ffmpeg'
            ]

build_requires = [
                  'python',
                  'nose'
]

def commands():
    env.NST_VGG_MODEL.set('/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth')
    env.PATH.append('{root}/bin')
    env.PYTHONPATH.append('{root}/python')
