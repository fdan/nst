from nst import StyleImager

try:
    import tractor.api.author
except ImportError:
    pass


def style_contact_sheeet(style_image, x, y, mips, iterations, engine, outdir):
    """
    Given a style image, create a bunch of farm tasks for each vgg layer,
    halving the style resolution via mip mapping.
    """
    vgg_layers = ['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r33', 'r34', 'r41', 'r42', 'r43', 'r44', 'r51', 'r52', 'r53', 'r54']

    style_image_filename = style_image.split('/')[-1]

    job = tractor.api.author.Job()
    job.title = "style_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']

    job.envkey = envkey
    job.service = 'lighting'
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
            cmd += ['--out', "%s/%s/mip%s/%s.exr" % (outdir, style_image_filename, mip, layer)]
            cmd += ['--iterations', iterations]
            cmd += ['--slayers', layer]
            cmd += ['--sweights', 1.0]
            cmd += ['--style-scale', scale]
            cmd += ['--opt-x', x]
            cmd += ['--opt-y', y]

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer), argv=cmd)

        gatys_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        gatys_weights = [0.244140625, 0.06103515625, 0.0152587890625, 0.003814697265625, 0.003814697265625]

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
        cmd += ['--out', "%s/%s/mip%s/%s.exr" % (outdir, style_image_filename, mip, 'gatys_layers')]
        cmd += ['--iterations', iterations]
        cmd += ['--slayers', ':'.join([str(x) for x in gatys_layers])]
        cmd += ['--sweights', ':'.join([str(x) for x in gatys_weights])]
        cmd += ['--style-scale', scale]
        cmd += ['--opt-x', x]
        cmd += ['--opt-y', y]        
        job.newTask(title="%s_mip_%s_gatysLayers" % (style_image_filename, mip), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass

    #print(job.asTcl())


def nst_contact_sheeet(style_image, content_image, mips, iterations, engine, outdir):
    """
    Given a style image, create a bunch of farm tasks for each vgg layer,
    halving the style resolution via mip mapping.
    """
    vgg_layers = ['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r33', 'r34', 'r41', 'r42', 'r43', 'r44', 'r51', 'r52', 'r53', 'r54']

    style_image_filename = style_image.split('/')[-1]

    job = tractor.api.author.Job()
    job.title = "nst_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']

    job.envkey = envkey
    job.service = 'lighting'
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
            cmd += ['--out', "%s/%s/mip%s/%s.exr" % (outdir, style_image_filename, mip, layer)]
            cmd += ['--iterations', iterations]
            cmd += ['--slayers', layer]
            cmd += ['--sweights', 1.0]
            cmd += ['--style-scale', scale]
            cmd += ['--clayers', layer]
            cmd += ['--cweights', 1.0]

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer), argv=cmd)

        gatys_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        gatys_weights = [0.244140625, 0.06103515625, 0.0152587890625, 0.003814697265625, 0.003814697265625]

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
        cmd += ['--out', "%s/%s/mip%s/%s.exr" % (outdir, style_image_filename, mip, 'gatys_layers')]
        cmd += ['--iterations', iterations]
        cmd += ['--style-scale', scale]
        cmd += ['--slayers', ':'.join([str(x) for x in gatys_layers])]
        cmd += ['--sweights', ':'.join([str(x) for x in gatys_weights])]
        cmd += ['--clayers', 'r41']
        cmd += ['--cweights', 1.0]
        job.newTask(title="%s_mip_%s_gatysLayers" % (style_image_filename, mip), argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass




