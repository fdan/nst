#! /usr/bin/bash

docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git ffdanff/nst-nuke:latest


