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


def nst_job_v2(style_image, content_image, mips, out_dir, opt=None, content_layers='r41', content_weights='1.0',
               content_mips='1', content_mip_weights='1.0', style_mips='5', style_layers='p1:r32', style_weights='0.5:0.5',
               style_mip_weights='1.0,1.0,1.0,1.0,1.0:1.0,1.0,1.0,1.0,1.0'):

    style_image_name = style_image.split('/')[-1]
    job = make_tractor_job(title='style_transfer_wedge_%s' % style_image_name)

    cmd = make_singularity_cmd()
    cmd += ['--from-content', True]
    cmd += ['--engine', 'cpu']

    cmd += ['--content', content_image]
    cmd += ['--clayers', content_layers]
    cmd += ['--cweights', content_weights]
    cmd += ['--cmips', content_mips]
    cmd += ['--cmipweights', content_mip_weights]

    cmd += ['--style', style_image]
    cmd += ['--slayers', style_layers]
    cmd += ['--sweights', style_weights]
    cmd += ['--smips', style_mips]
    cmd += ['--smipweights', style_mip_weights]

    cmd += ['--out', '%s/out.exr' % (out_dir)]

    if opt:
        cmd += ['--opt', opt]

    job.newTask(title="style_transfer_wedge_%s" % (style_image_name), argv=cmd)

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




