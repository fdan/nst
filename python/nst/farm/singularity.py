import os

try:
    import tractor.api.author
except ImportError:
    pass


class NstFarm(object):

    def __init__(self):
        self.from_content = True
        self.style = ''
        self.content = ''
        self.opt = ''
        self.out = ''
        self.frames = ''
        self.iterations = 500
        self.clayers = ['r41']
        self.cweights = ['1.0']
        self.slayers = []
        self.sweights = []
        self.smasks = []

    def send_to_farm(self):
        cmd = []

        cmd += ['singularity']
        cmd += ['exec']
        cmd += ['--nv']
        cmd += ['--bind']
        cmd += ['/mnt/ala']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/nst.sif']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
        cmd += ['--from-content', self.from_content]
        cmd += ['--style', self.style]
        cmd += ['--content', self.content]
        if self.opt:
            cmd += ['--opt', self.opt]
        cmd += ['--out', self.out]
        cmd += ['--iterations', self.iterations]
        cmd += ['--clayers', ':'.join([str(x) for x in self.clayers])]
        cmd += ['--cweights', ':'.join([str(x) for x in self.cweights])]
        cmd += ['--slayers', ':'.join([str(x) for x in self.slayers])]
        cmd += ['--sweights', ':'.join([str(x) for x in self.sweights])]

        if self.smasks:
            cmd += ['--smasks', ':'.join([str(x) for x in self.smasks])]

        user = os.getenv('USER')
        jobname = 'style_transfer_{0}'.format(user)

        job = tractor.api.author.Job()
        job.title = jobname

        envkey = []
        envkey += ['setenv PYTHONPATH=/mnt/ala/research/danielf/2021/git/nst/python']
        envkey += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth']

        job.envkey = envkey
        job.service = '(@.gpucount > 0)'
        # whistler, numbat, leech, addax, cicada, minnow, klipspringer, avocet, osprey, mosasaur, antelope, lobster, acouchi, sturgeon, shelduck, koala, trumpeter, ocelot, ladybug, roadrunner, quagga, opossum, xerus, ferret, piranha, ostrich, meerkat, snail, tortise

        # job.service = 'jerry' # best gpu we have currently
        job.tier = 'batch'
        job.atmost = 56

        processed_frames = self.eval_frames()
        for frame in processed_frames:
            frame_cmd = cmd + ['--frames', frame]
            frame_task_name = jobname + '_%04d' % frame
            job.newTask(title=frame_task_name, argv=frame_cmd)

        try:
            job.spool()
        except:
            print('job sent to farm')
        # print(job.asTcl())
        # jobid = job.spool()
        # print('job sent to farm, id:', jobid)

    def eval_frames(self):
        """
        Return a list of all frame numbers to be rendered.
        Step means every nth frame.
        """
        frames = []

        for token1 in self.frames.split(';'):

            # check for individual frames
            try:
                int(token1)
                frames.append(int(token1))

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
                frames += frame_range[::int(step)]

        # remove any repeated frames
        unique_frames = list(set(frames))
        unique_frames.sort()
        return unique_frames