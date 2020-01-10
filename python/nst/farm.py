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
    cmds = get_batch_cmd(opts)

    output_dir = get_full_path(opts.output_dir)
    prog = opts.progressive

    user = os.getenv('CREWNAME' or 'unknownUser')
    jobname = 'style_transfer_{0}'.format(user)

    job = tractor.api.author.Job()
    job.title = jobname

    rez_package_version = os.getenv('REZ_NST_VERSION')
    nst_package = 'nst-{0}'.format(rez_package_version)
    rez_packages = 'rez-pkgs={0}'.format(nst_package)
    job.envkey = [rez_packages]

    job.service = 'Studio'
    job.tier = 'batch'

    for cmd in cmds:
        task_name = jobname + '_task'
        task = tractor.api.author.Task(title=task_name)
        task.newCommand(argv=['setup-conda-env', '-i'])
        task.newCommand(argv=cmd)
        # task.newCommand(argv=['setup-conda-env', '-r'])
        print job, task

        job.addChild(task)

    print job.asTcl()
    jobid = job.spool()
    print 'job sent to farm, id:', jobid


def get_batch_cmd(opts):
    cmds = []

    style = opts.style
    content = opts.content
    output_dir = opts.output_dir
    engine = opts.engine
    iterations = opts.iterations
    loss = opts.loss
    unsafe = opts.unsafe
    random = opts.random_style
    progressive = opts.progressive

    cmd = []
    cmd += ['nst']
    cmd += ['--content', content]
    cmd += ['--style', style]
    cmd += ['--output-dir', output_dir]
    cmd += ['--unsafe', unsafe]

    if iterations:
        cmd += ['--iterations', iterations]
    elif loss:
        cmd += ['--loss', loss]

    if random:
        cmd += ['--random', random]

    if progressive:
        cmd += ['--progressive', progressive]

    cmd += ['--engine', engine, ';']

    # cmd += ['setup-conda-env', '-r']

    cmds.append(cmd)
    return cmds