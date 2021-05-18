# -*- coding: utf-8 -*-

name = 'nst'

version = '2.0.1'

requires = [
            'tractor',
            'ffmpeg'
            ]

build_requires = [
                  'python'
]

def commands():
    env.NST_VGG_MODEL.set('/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth')
    env.PATH.append('{root}/bin')
    env.PYTHONPATH.append('{root}/python')
    env.NST_HOME.set('{root}')
