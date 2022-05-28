from collections import ChainMap
import json

from nst import StyleImager
import nst

try:
    import tractor.api.author
except ImportError:
    pass


VGG_LAYERS = ['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r33', 'r34', 'r41', 'r42', 'r43', 'r44', 'r51', 'r52', 'r53', 'r54']
GATYS_LAYERS = ['r11', 'r31', 'r41', 'r51']


def lerp(start, end, step):
    diff = abs(start - end)
    incr = diff / float(step-1)
    steps = []
    value = start
    for x in range(0, step):
        steps.append(value)
        value -= incr
    return steps


def make_tractor_job(title='style_transfer', service='Studio', atmost=28):
    job = tractor.api.author.Job()
    job.title = title
    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
    job.envkey = envkey
    job.service = service
    job.tier = 'batch'
    job.atmost = atmost
    return job


def make_singularity_cmd():
    cmd = []
    cmd += ['singularity']
    cmd += ['exec']
    cmd += ['--nv']
    cmd += ['--bind']
    cmd += ['/mnt/ala']
    cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
    cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
    return cmd


def wedge(style_image, content, mips, varying_mips, start, end, step, out_dir):
    style_image_name = style_image.split('/')[-1]

    job = make_tractor_job(title='style_transfer_wedge_%s' % style_image_name)

    for weight in lerp(start, end, step):

        for vm in varying_mips.keys():
            vm.layers['r31'] = weight

        style = nst.Style(style_image, mips)

        cmd = make_singularity_cmd()
        cmd += ['--from-content', True]
        cmd += ['--style', style.image]
        cmd += ['--engine', 'cpu']
        cmd += ['--content', content]
        cmd += ['--out', '%s/out_%s.exr' % (out_dir, weight)]

        mds = [mip.as_dict() for mip in style.mips]
        md = dict(ChainMap(*mds))
        cmd += ['--mips', json.dumps(md)]

        job.newTask(title="style_transfer_wedge_%s_%s" % (style_image_name, weight), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass


def nst_job_v2(style_image_1,
               content_image=None,
               out='',
               from_content=True,
               opt_x=500,
               opt_y=500,
               service='Studio',
               job_title=None,
               style_zoom=1.0,
               style_rescale=1.0,
               progressive=False,
               log_iterations=100,
               iterations=500,
               zoom_factor=0.17,
               gauss_scale_factor=0.63,
               opt=None,
               style_scale=1.0,
               content_layers='r41',
               content_weights='1.0',
               content_mips='1',
               content_mip_weights='1.0',
               style_mips_1='5',
               style_layers_1='p1:r32',
               style_weights_1='0.5:0.5',
               style_mip_weights_1='1.0,1.0,1.0,1.0,1.0:1.0,1.0,1.0,1.0,1.0',
               style_in_mask_1=None,
               style_target_map_1=None,
               style_image_2=None,
               style_mips_2='5',
               style_layers_2='p1:r32',
               style_weights_2='0.5:0.5',
               style_mip_weights_2='1.0,1.0,1.0,1.0,1.0:1.0,1.0,1.0,1.0,1.0',
               style_in_mask_2=None,
               style_target_map_2=None):


    style_image_name_1 = style_image_1.split('/')[-1]

    if not job_title:
        job_title = 'style_transfer_wedge_%s' % style_image_name_1

    job = make_tractor_job(title=job_title, atmost=56, service=service)

    cmd = make_singularity_cmd()
    cmd += ['--from-content', True]
    cmd += ['--engine', 'cpu']
    cmd += ['--out', '%s' % out]
    cmd += ['--zoom_factor', zoom_factor]
    cmd += ['--gauss_scale_factor', gauss_scale_factor]
    cmd += ['--style-rescale', style_rescale]
    cmd += ['--style-zoom', style_zoom]
    cmd += ['--iterations', iterations]
    cmd += ['--progressive', progressive]
    cmd += ['--from-content', from_content]
    cmd += ['--log-iterations', log_iterations]
    if opt:
        cmd += ['--opt', opt]
    if opt_x:
        cmd += ['--opt-x', opt_x]
    if opt_y:
        cmd += ['--opt-y', opt_y]

    if content_image:
        cmd += ['--content', content_image]
        cmd += ['--clayers', content_layers]
        cmd += ['--cweights', content_weights]
        cmd += ['--cmips', content_mips]
        cmd += ['--cmipweights', content_mip_weights]

    cmd += ['--style-1', style_image_1]
    cmd += ['--slayers-1', style_layers_1]
    cmd += ['--sweights-1', style_weights_1]
    cmd += ['--smips-1', style_mips_1]
    cmd += ['--smipweights-1', style_mip_weights_1]
    if style_target_map_1:
        cmd += ['--style-target-map-1', style_target_map_1]
    if style_in_mask_1:
        cmd += ['--style-in-mask-1', style_in_mask_1]

    if style_image_2:
        cmd += ['--style-2', style_image_2]
        cmd += ['--slayers-2', style_layers_2]
        cmd += ['--sweights-2', style_weights_2]
        cmd += ['--smips-2', style_mips_2]
        cmd += ['--smipweights-2', style_mip_weights_2]
        if style_target_map_2:
            cmd += ['--style-target-map-2', style_target_map_2]
        if style_in_mask_2:
            cmd += ['--style-in-mask-2', style_in_mask_2]

    job.newTask(title="style_transfer_wedge_%s" % (style_image_name_1), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass


def nst_job(style_image, content, mips, out_dir, content_scale=1.0, opt=None, content_layers='r41', content_weights='1.0'):
    style_image_name = style_image.split('/')[-1]

    job = make_tractor_job(title='style_transfer_wedge_%s' % style_image_name)

    style = nst.Style(style_image, mips)

    cmd = make_singularity_cmd()
    cmd += ['--from-content', True]
    cmd += ['--style', style.image]
    cmd += ['--engine', 'cpu']
    cmd += ['--content', content]
    cmd += ['--clayers', content_layers]
    cmd += ['--cweights', content_weights]
    cmd += ['--content-scale', content_scale]
    cmd += ['--out', '%s/out.exr' % (out_dir)]

    if opt:
        cmd += ['--opt', opt]

    mds = [mip.as_dict() for mip in style.mips]
    md = dict(ChainMap(*mds))
    cmd += ['--mips', json.dumps(md)]

    job.newTask(title="style_transfer_wedge_%s" % (style_image_name), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass


def style_contact_sheet(style_image, x, y, mips, iterations, engine, outdir, version='v001', service="Studio", vgg_layers=VGG_LAYERS):
    """
    Given a style image, create a bunch of farm tasks for each vgg layer,
    halving the style resolution via mip mapping.
    """
    style_image_filename = style_image.split('/')[-1]

    job = tractor.api.author.Job()
    job.title = "style_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']

    job.envkey = envkey
    job.service = service
    job.tier = 'batch'
    job.atmost = 28

    for mip in range(1, mips):
        scale = 1. / float(mip)

        for layer in vgg_layers:
            cmd = []
            cmd += ['singularity']
            cmd += ['exec']
            cmd += ['--nv']
            cmd += ['--bind']
            cmd += ['/mnt/ala']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
            cmd += ['--from-content', False]
            cmd += ['--style', style_image]
            cmd += ['--engine', engine]
            cmd += ['--out', "%s/%s/%s/style/mip%s/%s.exr" % (outdir, style_image_filename, version, mip, layer)]
            cmd += ['--iterations', iterations]
            cmd += ['--mips', "{\"%s\": {\"%s\": 1.0}}" % (scale, layer)]
            cmd += ['--style-scale', scale]
            cmd += ['--opt-x', x]
            cmd += ['--opt-y', y]

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer), argv=cmd)

        cmd = []
        cmd += ['singularity']
        cmd += ['exec']
        cmd += ['--nv']
        cmd += ['--bind']
        cmd += ['/mnt/ala']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
        cmd += ['--from-content', False]
        cmd += ['--style', style_image]
        cmd += ['--engine', engine]
        cmd += ['--out', "%s/%s/%s/style/mip%s/%s.exr" % (outdir, style_image_filename, version, mip, 'gatys_layers')]
        cmd += ['--iterations', iterations]
        cmd += ['--mips', "{\"%s\": {\"r11\": 0.244140625, \"r21\": 0.06103515625, \"r31\": 0.0152587890625, \"r41\": 0.003814697265625, \"r51\": 0.003814697265625}}" % (scale)]
        cmd += ['--style-scale', scale]
        cmd += ['--opt-x', x]
        cmd += ['--opt-y', y]        
        job.newTask(title="%s_mip_%s_gatysLayers" % (style_image_filename, mip), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass

    #print(job.asTcl())


def nst_contact_sheet(style_image, content_image, mips, iterations, engine, outdir, version='v001', service="Studio", opt=None, vgg_layers=VGG_LAYERS, do_gatys=True):
    """
    Given a style image, create a bunch of farm tasks for each vgg layer,
    halving the style resolution via mip mapping.
    """
    style_image_filename = style_image.split('/')[-1]

    job = tractor.api.author.Job()
    job.title = "nst_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']

    job.envkey = envkey
    job.service = service
    job.tier = 'batch'
    job.atmost = 28

    for mip in range(1, mips):
        scale = 1. / float(mip)

        for layer in vgg_layers:
            cmd = []
            cmd += ['singularity']
            cmd += ['exec']
            cmd += ['--nv']
            cmd += ['--bind']
            cmd += ['/mnt/ala']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
            cmd += ['--from-content', True]
            cmd += ['--style', style_image]
            cmd += ['--content', content_image]
            cmd += ['--engine', engine]

            if opt:
                cmd += ['--opt', opt]

            cmd += ['--out', "%s/%s/%s/nst/mip%s/%s.exr" % (outdir, style_image_filename, version, mip, layer)]
            cmd += ['--iterations', iterations]
            cmd += ['--mips', "{\"%s\": {\"%s\": 1.0}}" % (scale, layer)]
            cmd += ['--style-scale', scale]
            cmd += ['--clayers', layer]
            cmd += ['--cweights', 1.0]

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer), argv=cmd)

        if do_gatys:
            cmd = []
            cmd += ['singularity']
            cmd += ['exec']
            cmd += ['--nv']
            cmd += ['--bind']
            cmd += ['/mnt/ala']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
            cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
            cmd += ['--from-content', True]
            cmd += ['--style', style_image]
            cmd += ['--content', content_image]
            cmd += ['--engine', engine]
            cmd += ['--out', "%s/%s/%s/nst/mip%s/%s.exr" % (outdir, style_image_filename, version, mip, 'gatys_layers')]
            cmd += ['--iterations', iterations]
            cmd += ['--style-scale', scale]
            # cmd += ['--mips', "{\"%s\": {\"r11\": 0.244140625, \"r21\": 0.06103515625, \"r31\": 0.0152587890625, \"r41\": 0.003814697265625, \"r51\": 0.003814697265625}}" % (scale)]
            cmd += ['--mips', "{\"%s\": {\"r21\": 0.06103515625, \"r31\": 0.0152587890625, \"r41\": 0.003814697265625, \"r51\": 0.003814697265625}}" % (scale)]
            cmd += ['--clayers', 'r41']
            cmd += ['--cweights', 1.0]
            job.newTask(title="%s_mip_%s_gatysLayers" % (style_image_filename, mip), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass




