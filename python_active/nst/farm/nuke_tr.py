import nst.settings as settings
import tractor.api.author

import nuke
from . import tr

import datetime
import uuid
import shutil
import os


def init_write_nodes(node):
    out_fp = node.knob('out_fp').value()
    out_fp = out_fp.replace('%04d', '####')
    out_dir = os.path.abspath(os.path.join(out_fp, os.path.pardir))
    now = datetime.datetime.today()
    temp_dir = '%s-%s-%s--%s-%s-%s' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    temp_dir = os.path.join(out_dir, temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    content_fp = os.path.join(temp_dir, 'content/content.####.exr')
    opt_fp = os.path.join(temp_dir, 'opt/opt.####.exr')
    style_fp = os.path.join(temp_dir, 'style/style.####.exr')
    style_target_fp = os.path.join(temp_dir, 'style_target/style_target.####.exr')
    node.node('content_write').knob('file').setValue(content_fp)
    node.node('opt_write').knob('file').setValue(opt_fp)
    node.node('style_write').knob('file').setValue(style_fp)
    node.node('style_target_write').knob('file').setValue(style_target_fp)

    # change gizmo switches to use farm resolution
    lookdev_resolution = node.knob('resolution').value()
    farm_resolution = node.knob('farm_resolution').value()
    node.knob('resolution').setValue(farm_resolution)

    # # if run_parent is true, chance the opt image to be the output from the parent job
    # parent = node.knob('parent').value
    # if parent and node.knob('run_parent').value():
    #     # get the parent gizmo's out_fp file knob
    #     p_node = nuke.toNode(parent)
    #     p_out_fp = p_node.knob('out_fp').value()
    #     node.node('opt_override_read').knob('file').setValue(p_out_fp)
    #     node.node('opt_override_switch').knob('which').setValue(1)

    nuke.scriptSave()
    orig_script_path = nuke.scriptName()
    archive_script_path = os.path.join(temp_dir, os.path.basename(orig_script_path))
    shutil.copy(orig_script_path, archive_script_path)

    # swap back to the original lookdev resolution
    node.knob('resolution').setValue(lookdev_resolution)

    # # reset the opt override switch
    # if parent and node.knob('run_parent').value():
    #     node.node('opt_override_switch').knob('which').setValue(0)

    pd = {
        'temp_dir': temp_dir,
        'script_path': archive_script_path,
        'out_fp': out_fp,
        'content_fp': content_fp,
        'opt_fp': opt_fp,
        'style_fp': style_fp,
        'style_target_fp': style_target_fp
    }
    return pd


def nuke_submit(node, dry_run=False):
    """
    node: the gizmo node
    """
    # to do
    # do png and mov generation after nst job
    # if temporal coherence is checked, frames must be done in serial or as a single giant job (?)

    cache_paths = init_write_nodes(node)

    ws = settings.WriterSettings()

    script_path = cache_paths['script_path']
    temp_dir = cache_paths['temp_dir']

    temporal_coherence = node.knob('temporal_coherence').value()

    ws.core.engine = node.knob('farm_engine').value()
    ws.core.optimiser = node.knob('farm_optimiser').value()
    ws.core.style_mips = int(node.knob('farm_style_mips').value())
    ws.core.learning_rate = float(node.knob('farm_learning_rate').value())
    ws.core.iterations = int(node.knob('farm_iterations').value())
    ws.core.log_iterations = 10

    # only apply proxy scaling if engine resolution is proxy!
    lookdev_resolution = node.knob('resolution').value()

    if lookdev_resolution == 'proxy':
        proxy_scale = float(node.knob('proxy_scale').value())
    else:
        proxy_scale = 1.0

    nuke_pyramid_span = float(node.knob('style_pyramid_span').value())
    nuke_style_zoom = float(node.knob('style_zoom').value())
    ws.core.style_pyramid_span = nuke_pyramid_span * proxy_scale
    ws.core.style_zoom = nuke_style_zoom / proxy_scale

    # if lookdev_resolution == 'proxy':
    #     ws.core.laplacian_filter_kernel /= proxy_scale
    #     ws.core.laplacian_filter_kernel = int(ws.core.laplacian_filter_kernel + 1)
    #     ws.core.laplacian_blur_kernel /= proxy_scale
    #     ws.core.laplacian_blur_kernel = int(ws.core.laplacian_blur_kernel + 1)
    #     # ws.core.laplacian_blur_sigma /= proxy_scale

    ws.core.mip_weights = [float(x) for x in node.knob('style_mip_weights').value().split(',')]
    ws.core.style_layers = node.knob('style_layers').value().split(',')
    ws.core.mask_layers = node.knob('mask_layers').value().split(',')
    ws.core.style_layer_weights = [float(x) for x in node.knob('style_layer_weights').value().split(',')]
    ws.core.content_layer = node.knob('content_layer').value()
    ws.core.content_layer_weight = float(node.knob('content_layer_weight').value())
    ws.core.gram_weight = float(node.knob('gram_weight').value())
    ws.core.histogram_weight = float(node.knob('histogram_weight').value())
    ws.core.tv_weight = float(node.knob('tv_weight').value())
    ws.core.laplacian_weight = float(node.knob('laplacian_weight').value())

    ws.core.histogram_loss_type = node.knob('histogram_loss_type').value()
    ws.core.gram_loss_type = node.knob('gram_loss_type').value()
    ws.core.content_loss_type = node.knob('content_loss_type').value()
    ws.core.laplacian_loss_type = node.knob('laplacian_loss_type').value()

    after_jobs = node.knob('run_after').value()

    job = tractor.api.author.Job()
    job.title = node.knob('job_name').value()
    job.service = node.knob('service_key').value()
    job.tier = node.knob('tier').value()
    job.atmost = int(ws.core.cpu_threads)
    frames = str(node.knob('frames').value())

    if after_jobs:
        job.afterjids = [int(j) for j in after_jobs.split(' ')]

    rez_resolve = ' '.join([x for x in os.getenv('REZ_RESOLVE').split(' ')
                            if 'tk_nuke' not in x
                            and 'tractor' not in x
                            and 'nst' not in x
                            and 'devtoolset' not in x])

    rez_packages = 'rez-pkgs=' + rez_resolve

    # nuke job to cache inputs
    write_nodes = []
    for x in node.nodes():
        if x.Class() != 'Write':
            continue
        if 'snapshot' in x.name():
            continue
        if not x.knob('file').value():
            continue
        write_nodes.append('%s.%s' % (node.name(), x.name()))
    write_nodes = ','.join(write_nodes)

    # write_nodes = ','.join(['%s.%s' % (node.name(), x.name()) for x in node.nodes() if x.Class() == 'Write'])

    nuke_cmd = ['nukex_run']
    nuke_cmd += ['-t']
    nuke_cmd += ['-F']
    nuke_cmd += [frames]
    nuke_cmd += ['-X']
    nuke_cmd += ['%s' % write_nodes]
    nuke_cmd += [script_path]
    # how to limit threads?
    nuke_tractor_cmd = tractor.api.author.Command(argv=nuke_cmd)
    nuke_tractor_cmd.envkey = [rez_packages]
    nuke_task = job.newTask(title='nuke_input_cache')
    nuke_task.addCommand(nuke_tractor_cmd)

    # add cleanup job
    cleanup_cmd = ['rm', '-rf', temp_dir]
    cleanup_tractor_cmd = tractor.api.author.Command(argv=cleanup_cmd)
    cleanup_task = job.newTask(title='nuke_cache_cleanup')
    cleanup_task.addCommand(cleanup_tractor_cmd)

    # oiio job - no temporal coherence
    frame_list = tr.eval_frames(frames)
    for frame in frame_list:
        ws.frame = frame

        out_fp = cache_paths['out_fp'].replace('####', '%04d' % frame)
        ws.out = out_fp

        content_fp = cache_paths['content_fp'].replace('####', '%04d' % frame)
        content = settings.Image()
        content.rgb_filepath = content_fp
        ws.content = content

        opt_fp = cache_paths['opt_fp'].replace('####', '%04d' % frame)
        opt = settings.Image()
        opt.rgb_filepath = opt_fp
        ws.opt_image = opt

        style_fp = cache_paths['style_fp'].replace('####', '%04d' % frame)
        style_target_fp = cache_paths['style_target_fp'].replace('####', '%04d' % frame)
        style = settings.StyleImage()
        style.rgba_filepath = style_fp
        style.target_map_filepath = style_target_fp
        ws.styles = [style]

        output_dir = os.path.abspath(os.path.join(ws.out, os.pardir))
        nst_json = os.path.abspath(os.path.join(output_dir, 'nst.%04d.json' % frame))
        ws.save(nst_json)
        frame_cmd = tr.write_cmd_file(output_dir, frame=frame)
        frame_task = job.newTask(title="oiio_nst_job_%04d" % frame)

        if not temporal_coherence:
            frame_task.addChild(nuke_task)
        else:
            # chain to previous frame
            pass

        cleanup_task.addChild(frame_task)

        envkey_oiio = []
        envkey_oiio += ['setenv PYTHONPATH=/mnt/ala/research/danielf/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
        envkey_oiio += ['setenv TRACTOR_ENGINE=frank:5600']
        envkey_oiio += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/git/nst/models/vgg_conv.pth']
        envkey_oiio += ['setenv OCIO=/mnt/ala/research/danielf/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
        command_3 = tractor.api.author.Command(argv=frame_cmd, envkey=envkey_oiio)
        frame_task.addCommand(command_3)

    # print(job.asTcl())

    if not dry_run:
        job.spool()

        # try:
        #     print(job.spool())
        # except:
        #     pass

    print("job submitted", dir(job))
