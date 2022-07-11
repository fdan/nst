import nst.settings as settings
import tractor.api.author

import nuke

import uuid
import os
import stat


def write_cmd_file(output_dir):
    cmd = []
    cmd += ['singularity']
    cmd += ['exec']
    cmd += ['--nv']
    cmd += ['--bind']
    cmd += ['/mnt/ala']
    cmd += ['/mnt/ala/research/danielf/git/nst/python/nst/oiio/environment/singularity/nst_oiio.sif']
    cmd += ['/mnt/ala/research/danielf/git/nst/bin/nst']

    cmd += ['--load']
    cmd += [os.path.abspath(os.path.join(output_dir, 'nst.json'))]

    cmd_sh = '#! /usr/bin/bash\n\n' + ' '.join(cmd) + '\n'

    try:
        # os.makedirs(output_dir, exist_ok=True) - python3
        os.makedirs(output_dir)
    except:
        if os.path.isdir(output_dir):
            pass

    cmd_fp = os.path.abspath(os.path.join(output_dir, 'cmd.sh'))

    # python3
    # with open(cmd_fp, mode="wt", encoding="utf-8") as file:
    #     file.write(cmd_sh)

    with open(cmd_fp, mode="wt") as file:
        file.write(cmd_sh)

    st = os.stat(cmd_fp)
    os.chmod(cmd_fp, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return cmd_fp


def submit(settings, service_key, job_name='nst_job'):
    output_dir = os.path.abspath(os.path.join(settings.out, os.pardir))
    cmd_fp = write_cmd_file(output_dir)

    nst_json = os.path.abspath(os.path.join(output_dir, 'nst.json'))
    settings.save(nst_json)

    job = tractor.api.author.Job()
    job.title = job_name
    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
    job.envkey = envkey
    job.service = service_key
    job.tier = 'batch'
    job.atmost = 56

    cmd = [cmd_fp]

    job.newTask(title="style_transfer_test", argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass


def nuke_submit(node):
    c = [x for x in node.nodes() if x.Class() == "MLClient"]
    assert len(c) == 1
    mlc = c[0]

    ws = settings.WriterSettings()

    style1 = settings.StyleImage()
    style1.rgba_filepath = mlc.knob('style1_fp').value()
    style1.target_map_filepath = mlc.knob('style1_target_fp').value()

    # todo: check if inputs are connected
    style2 = settings.StyleImage()
    style2.rgba_filepath = mlc.knob('style2_fp').value()
    style2.target_map_filepath = mlc.knob('style2_target_fp').value()

    style3 = settings.StyleImage()
    style3.rgba_filepath = mlc.knob('style3_fp').value()
    style3.target_map_filepath = mlc.knob('style3_target_fp').value()

    ws.styles = [style1, style2, style3]

    content = settings.Image()
    content.rgb_filepath = mlc.knob('content_fp').value()
    ws.content = content

    opt = settings.Image()
    opt.rgb_filepath = mlc.knob('opt_fp').value()
    ws.opt_image = opt

    ws.out = mlc.knob('out_fp').value()

    ws.core.engine = mlc.knob('farm_engine').value()
    ws.core.optimiser = mlc.knob('farm_optimiser').value()
    ws.core.pyramid_scale_factor = mlc.knob('pyramid_scale_factor').value()
    ws.core.style_mips = mlc.knob('style_mips').value()
    ws.core.style_mip_weights = mlc.knob('style_mip_weights').value()
    ws.core.style_layers = mlc.knob('style_layers').value()
    ws.core.style_layer_weights = mlc.knob('style_layer_weights').value()
    ws.core.content_layer = mlc.knob('content_layer').value()
    ws.core.content_layer_weight = mlc.knob('content_layer_weight').value()
    ws.core.content_mips = mlc.knob('content_mips').value()
    ws.core.learning_rate = mlc.knob('farm_learning_rate').value()
    ws.core.scale = mlc.knob('farm_scale').value()
    ws.core.iterations = mlc.knob('farm_iterations').value()
    ws.core.log_iterations = 1

    output_dir = os.path.abspath(os.path.join(ws.out, os.pardir))
    nst_json = os.path.abspath(os.path.join(output_dir, 'nst.json'))
    ws.save(nst_json)

    # 1. determine which inputs are single frame and which are image sequences
    # 2. tr task/s to write out inputs for frame range
    # 3. nst task (todo: handle style image with mask as alpha channel)
    # 4. cleanup, remove temp files

    # write out media cache for oiio

    nst_inputs = {'content': 0, 'opt': 1, 'style1': 2, 'style1_target': 3, 'style2': 4, 'style2_target': 5,
                  'style3': 6, 'style3_target': 7}

    varying_inputs = []
    if node.input(nst_inputs['content']):
        varying_inputs += [node.node('content').dependent()[0]]
    if node.input(nst_inputs['opt']):
        varying_inputs += [node.node('opt').dependent()[0]]

    static_inputs = []
    if node.input(nst_inputs['style1']):
        static_inputs += [node.node('style1').dependent()[0]]
    if node.input(nst_inputs['style1_target']):
        static_inputs += [node.node('style1_target').dependent()[0]]
    if node.input(nst_inputs['style2']):
        static_inputs += [node.node('style2').dependent()[0]]
    if node.input(nst_inputs['style2_target']):
        static_inputs += [node.node('style2_target').dependent()[0]]
    if node.input(nst_inputs['style3']):
        static_inputs += [node.node('style3').dependent()[0]]
    if node.input(nst_inputs['style3_target']):
        static_inputs += [node.node('style3_target').dependent()[0]]

    job = tractor.api.author.Job()
    job.title = node.knob('job_name').value() or "nuke_nst_job_%s" % str(uuid.uuid4())[:8:]

    job.service = node.knob('service_key').value()
    job.tier = node.knob('tier').value()
    job.atmost = int(node.knob('atmost').value())

    frames = node.knob('frames').value()

    varying_input_names = ','.join(['%s.%s' % (node.name(), x.name()) for x in varying_inputs])
    static_input_names = ','.join(['%s.%s' % (node.name(), x.name()) for x in static_inputs])

    nuke_script = nuke.scriptName()

    envkey_nuke = ['rez-pkgs=nuke-12']
    envkey_nuke += ['setenv PYTHONPATH=/mnt/ala/research/danielf/git/nst/python:$PYTHONPATH']
    envkey_nuke += ['setenv NUKE_PATH=/mnt/ala/research/danielf/git/nst/nuke/build/ml-client:$NUKE_PATH']
    envkey_nuke += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/git/nst/bin/Models/vgg_conv.pth']

    if varying_inputs:
        cmd_1 = ['$NUKE_BIN', '-X', varying_input_names, '-F', frames, nuke_script]
        task_1 = job.newTask(title='nst_varying_input_cache')
        command_1 = tractor.api.author.Command(argv=cmd_1, envkey=envkey_nuke)
        task_1.addCommand(command_1)

    if static_inputs:
        cmd_2 = ['$NUKE_BIN', '-X', static_input_names, '-F', "1", nuke_script]
        task_2 = job.newTask(title='nst_static_input_cache')
        command_2 = tractor.api.author.Command(argv=cmd_2, envkey=envkey_nuke)
        task_2.addCommand(command_2)

    # oiio job

    output_dir = os.path.abspath(os.path.join(ws.out, os.pardir))
    cmd_3 = write_cmd_file(output_dir)
    task_3 = job.newTask(title="oiio_nst_job")
    envkey_oiio = []
    envkey_oiio += ['setenv PYTHONPATH=/mnt/ala/research/danielf/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey_oiio += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey_oiio += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/git/nst/bin/Models/vgg_conv.pth']
    envkey_oiio += ['setenv OCIO=/mnt/ala/research/danielf/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
    command_3 = tractor.api.author.Command(argv=cmd_3, envkey=envkey_oiio)
    task_3.addCommand(command_3)

    if varying_inputs:
        task_2.addChild(task_1)
    if static_inputs:
        task_3.addChild(task_2)

    print(job.asTcl())

    job.spool()










