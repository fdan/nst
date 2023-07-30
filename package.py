# -*- coding: utf-8 -*-

name = 'nst'

version = '3.0.2'

authors = ['daniel.flood']

requires = [
            'tractor',
            # 'oiio',
            # 'ocio_configs',
            # 'shotgun_utils'
           ]

build_requires = ['cmake-3.10+']

variants = [
#           ['nuke-12'],
            ['nuke-13', 'devtoolset-6']
           ]


def commands():
    env.NUKE_PATH.append('{root}/lib')
    env.NUKE_PATH.append('{root}/gizmos')
    env.PYTHONPATH.append('{root}/python')
    env.PATH.append('{root}/bin')
    env.NST_VGG_MODEL.set('/mnt/ala/research/danielf/git/nst/models/vgg_conv.pth')



