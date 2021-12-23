import os
from copy import deepcopy

try:
    import tractor.api.author
except ImportError:
    pass


class NstFarm(object):

    bad_nodes = ['cicada', 'meerkat', 'leech']

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
        # smi_cmd = ['nvidia-smi', '--query-gpu=persistence_mode,display_mode,display_active', '--format=csv,noheader,nounits']

        cmd = []
        cmd += ['singularity']
        cmd += ['exec']
        cmd += ['--nv']
        cmd += ['--bind']
        cmd += ['/mnt/ala']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.7.1_cuda-11.0/nst.sif']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst']
        cmd += ['--from-content', self.from_content]
        cmd += ['--style', self.style]
        cmd += ['--engine', self.engine]
        cmd += ['--content', self.content]
        # if self.opt:
        #     cmd += ['--opt', self.opt]
        cmd += ['--out', self.out]
        cmd += ['--iterations', self.iterations]
        cmd += ['--clayers', ':'.join([str(x) for x in self.clayers])]
        cmd += ['--cweights', ':'.join([str(x) for x in self.cweights])]
        cmd += ['--slayers', ':'.join([str(x) for x in self.slayers])]
        cmd += ['--sweights', ':'.join([str(x) for x in self.sweights])]

        user = os.getenv('USER')
        jobname = 'style_transfer_{0}'.format(user)

        job = tractor.api.author.Job()
        job.title = jobname

        envkey = []
        envkey += ['setenv PYTHONPATH=/mnt/ala/research/danielf/2021/git/nst/python']
        envkey += ['setenv NST_VGG_MODEL=/mnt/ala/research/danielf/models/gatys_nst_vgg/vgg_conv.pth']

        job.envkey = envkey
        job.service = '(@.gpucount > 0) && !(%s)' % ' || '.join(self.bad_nodes)
        job.tier = 'batch'
        job.atmost = 56

        if not self.frames:
            job.newTask(title="style transfer", argv=cmd)

        else:
            ffmpeg_out = self.out.replace('####', '%%04d')
            out_dir = os.path.abspath(os.path.join(self.out, os.path.pardir))
            ffmpeg_cmd = ['ffmpeg', '-start_number', '1001', '-i', ffmpeg_out, '-c:v', 'libx264', '-crf', '1', '-y']
            ffmpeg_cmd += ['%s/nst.mp4' % out_dir]
            ffmpeg_task = job.newTask(title='h264_gen', argv=ffmpeg_cmd)

            processed_frames = self.eval_frames()
            for frame in processed_frames:

                cmd_ = deepcopy(cmd)

                if self.smasks:
                    self.smasks = [x.replace("####", "%04d" % frame) if x else None for x in self.smasks]
                    cmd_ += ['--smasks', ':'.join([str(x) for x in self.smasks])]

                if self.content:
                    self.content_ = self.content.replace("####", "%04d" % frame)
                    cmd_ += ['--content', self.content]

                if self.opt:
                    self.opt_image_ = self.opt.replace("####", "%04d" % frame)
                    cmd_ += ['--opt', self.opt]

                out_ = self.out.replace("####", "%04d" % frame)
                cmd_ += ['--out', out_]

                frame_cmd = cmd_ + ['--frames', frame]
                frame_task_name = jobname + '_%04d' % frame

                frame_task = job.newTask(title=frame_task_name, argv=frame_cmd)
                ffmpeg_task.addChild(frame_task)

        try:
            job.spool()
        except:
            print('job sent to farm')

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