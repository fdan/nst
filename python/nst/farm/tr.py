import os
import stat

import tractor.api.author


def write_cmd_file(output_dir, frame=None):
    cmd = []
    cmd += ['singularity']
    cmd += ['exec']
    cmd += ['--nv']
    cmd += ['--bind']
    cmd += ['/mnt/ala']
    cmd += ['/mnt/ala/research/danielf/git/nst/environments/oiio/singularity/nst_oiio.sif']
    cmd += ['/mnt/ala/research/danielf/git/nst/bin/nst']

    cmd += ['--load']

    if frame:
        cmd += [os.path.abspath(os.path.join(output_dir, 'nst.%04d.json' % frame))]
    else:
        cmd += [os.path.abspath(os.path.join(output_dir, 'nst.json'))]

    cmd_sh = '#! /usr/bin/bash\n\n' + ' '.join(cmd) + '\n'

    try:
        os.makedirs(output_dir)
    except:
        if os.path.isdir(output_dir):
            pass

    if frame:
        cmd_fp = os.path.abspath(os.path.join(output_dir, 'cmd.%04d.sh' % frame))
    else:
        cmd_fp = os.path.abspath(os.path.join(output_dir, 'cmd.sh'))

    with open(cmd_fp, mode="wt") as file:
        file.write(cmd_sh)

    st = os.stat(cmd_fp)
    os.chmod(cmd_fp, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return cmd_fp


def submit(settings, service_key, job_name='danf_phd_style_transfer_job', threads=48):
    settings.core.cpu_threads = threads
    settings.core.engine = 'cpu'
    output_dir = os.path.abspath(os.path.join(settings.out, os.pardir))
    cmd_fp = write_cmd_file(output_dir)

    nst_json = os.path.abspath(os.path.join(output_dir, 'nst.json'))
    settings.save(nst_json)

    job = tractor.api.author.Job()
    job.title = job_name
    envkey = []
    envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
    envkey += ['setenv TRACTOR_ENGINE=frank:5600']
    envkey += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/git/nst/models/vgg_conv.pth']
    envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
    job.envkey = envkey
    job.service = service_key
    job.tier = 'batch'
    job.atmost = threads

    cmd = [cmd_fp]

    job.newTask(title="style_transfer_test", argv=cmd)

    try:
        job.spool()
    except TypeError:
        pass


def eval_frames(frames):
    """
    Return a list of all frame numbers to be rendered.
    Step means every nth frame.
    """
    frames_list = []

    token2 = frames.split('-')

    if len(token2) == 1:
        return [int(float(token2[0]))]

    elif len(token2) == 2:
        start = token2[0]
        end = token2[1]
        step = 1

    token3 = end.split('x')

    if len(token3) == 1:
        return [int(token3[0])]

    elif len(token3) == 2:
        end = token3[0]
        step = token3[1]

    frame_range = []
    for i in range(int(start), int(end)+1):
        frame_range.append(i)
    frames_list += frame_range[::int(step)]

    # remove any repeated frames
    unique_frames = list(set(frames_list))
    unique_frames.sort()
    return unique_frames










