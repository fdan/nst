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
    job.title = "style_transfer_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/mnt/ala/research/danielf/2021/git/nst/python']
    envkey += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth']

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

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer))
            job.spool()


def nst_contact_sheeet(style_image, content_image, mips, iterations, engine, outdir):
    """
    Given a style image, create a bunch of farm tasks for each vgg layer,
    halving the style resolution via mip mapping.
    """
    vgg_layers = ['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r33', 'r34', 'r41', 'r42', 'r43', 'r44', 'r51', 'r52', 'r53', 'r54']

    style_image_filename = style_image.split('/')[-1]

    job = tractor.api.author.Job()
    job.title = "style_transfer_contact_sheet_%s" % style_image_filename

    envkey = []
    envkey += ['setenv PYTHONPATH=/mnt/ala/research/danielf/2021/git/nst/python']
    envkey += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth']

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

            job.newTask(title="%s_mip_%s_layer_%s" % (style_image_filename, mip, layer))
            job.spool()




