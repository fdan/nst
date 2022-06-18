import nst.oiio
import tractor.api.author

import os
import stat


class TrStyleImager(nst.oiio.StyleImager):

    def __init__(self, styles=None, content=None, engine='cpu'):
        super(TrStyleImager, self).__init__(styles, content, engine=engine)

    def write_cmd_file(self):
        cmd = []
        cmd += ['singularity']
        cmd += ['exec']
        cmd += ['--nv']
        cmd += ['--bind']
        cmd += ['/mnt/ala']
        cmd += ['/mnt/ala/research/danielf/git/nst/python/nst/oiio/environment/singularity/nst_oiio.sif']
        cmd += ['/mnt/ala/research/danielf/git/nst/bin/nst']

        cmd += ['--load']
        cmd += [os.path.abspath(os.path.join(self.get_output_dir(), 'nst.yml'))]

        cmd_sh = '#! /usr/bin/bash\n\n' + ' '.join(cmd) + '\n'

        out_dir = self.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        cmd_fp = os.path.abspath(os.path.join(out_dir, 'cmd.sh'))

        with open(cmd_fp, mode="wt", encoding="utf-8") as file:
            file.write(cmd_sh)

        st = os.stat(cmd_fp)
        os.chmod(cmd_fp, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return cmd_fp

    def send_to_farm(self, service_key, job_name='nst_job'):
        cmd_fp = self.write_cmd_file()
        nst_yml = os.path.abspath(os.path.join(self.get_output_dir(), 'nst.yml'))
        self.save(nst_yml)

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
