import os

try:
    import tractor.api.author
except ImportError:
    pass


def get_full_path(filename):
    if not filename.startswith('/'):
        return os.getcwd() + '/' + filename
    return filename


def doit(opts):
    cmd = get_batch_cmd(opts)
    frames = eval_frames(opts.frames)

    user = os.getenv('CREWNAME' or 'unknownUser')
    jobname = 'style_transfer_{0}'.format(user)

    job = tractor.api.author.Job()
    job.title = jobname

    rez_package_version = os.getenv('REZ_NST_VERSION')
    nst_package = 'nst-{0} conda_pytorch'.format(rez_package_version)
    rez_packages = 'rez-pkgs={0}'.format(nst_package)
    job.envkey = [rez_packages]

    job.service = '(@.gpucount > 0)'
    job.tier = 'batch'
    job.atmost = 56

    for frame in frames:
        frame_cmd = cmd + ['--frames', str(frame)]
        frame_task_name = jobname + '_%04d' % frame
        task = tractor.api.author.Task(title=frame_task_name)
        task.newCommand(argv=['setup-conda-env', '-i'])
        task.newCommand(argv=frame_cmd)
        task.newCommand(argv=['setup-conda-env', '-r'])

        job.addChild(task)

    print(job.asTcl())
    jobid = job.spool()
    print('job sent to farm, id:', jobid)


def get_batch_cmd(opts):
    cmd = []
    cmd += ['nst']
    cmd += ['--style', opts.style]
    cmd += ['--content', opts.content]
    cmd += ['--out', opts.out]
    cmd += ['--clayers', opts.clayers]
    cmd += ['--cweights', opts.cweights]
    cmd += ['--slayers', opts.slayers]
    cmd += ['--sweights', opts.sweights]

    if opts.iterations:
        cmd += ['--iterations', opts.iterations]

    if opts.from_content:
        cmd += ['--from-content']

    # cmd += ['--engine', engine, ';']

    return cmd


def eval_frames(frames):
    """
    Return a list of all frame numbers to be rendered.
    Step means every nth frame.
    """
    frames_ = []

    for token1 in frames.split(';'):

        # check for individual frames
        try:
            int(token1)
            frames_.append(int(token1))

        # check for frame ranges
        except ValueError:
            token2 = token1.split(':')
            if len(token2) != 2:
                continue
            start = token2[0]
            end = token2[1]
            step = 1

            token3 = end.split('%')
            if len(token3) == 2:
                end = token3[0]
                step = token3[1]

            frame_range = []
            for i in range(int(start), int(end)+1):
                frame_range.append(i)
            frames_ += frame_range[::int(step)]

    # remove any repeated frames
    unique_frames = list(set(frames_))
    unique_frames.sort()
    return unique_frames