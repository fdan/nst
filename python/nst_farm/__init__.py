import nst
import tractor.api.author

import os
import stat


class TrStyleImager(nst.StyleImager):

    def __init__(self):
        super(TrStyleImager, self).__init__()
        self.service_key = ''

    def write_cmd_file(self):
        cmd = []
        cmd += ['singularity']
        cmd += ['exec']
        cmd += ['--nv']
        cmd += ['--bind']
        cmd += ['/mnt/ala']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/environments/singularity/pytorch-1.10_cuda-11.4/nst.sif']
        cmd += ['/mnt/ala/research/danielf/2021/git/nst/bin/nst2']
        cmd += ['--load']
        cmd += [os.path.abspath(os.path.join(self.get_output_dir(), 'nst.yml'))]

        cmd_sh = '#! /usr/bin/bash\n\n' + ' '.join(cmd) + '\n'

        cmd_fp = os.path.abspath(os.path.join(self.get_output_dir(), 'cmd.sh'))

        with open(cmd_fp, mode="wt", encoding="utf-8") as file:
            file.write(cmd_sh)

        st = os.stat(cmd_fp)
        os.chmod(cmd_fp, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return cmd_fp

    def send_to_farm(self):
        cmd_fp = self.write_cmd_file()
        nst_yml = os.path.abspath(os.path.join(self.get_output_dir(), 'nst.yml'))
        self.save(nst_yml)

        job = tractor.api.author.Job()
        job.title = 'nst_job'
        envkey = []
        envkey += ['setenv PYTHONPATH=/home/13448206/git/nst/python:/home/13448206/git/tractor/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH']
        envkey += ['setenv TRACTOR_ENGINE=frank:5600']
        envkey += ['setenv NST_VGG_MODEL=/home/13448206/git/nst/bin/Models/vgg_conv.pth']
        envkey += ['setenv OCIO=/home/13448206/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio']
        job.envkey = envkey
        job.service = self.service_key
        job.tier = 'batch'
        job.atmost = 56

        cmd = [cmd_fp]

        job.newTask(title="style_transfer_test", argv=cmd)

        try:
            job.spool()
        except TypeError:
            pass