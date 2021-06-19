#! /usr/bin/bash

docker run --gpus all --rm -it -p 8888:8888 -v /home/$USER/git:/workspace/git --user $(id -u):$(id -g) --net=host ffdanff/pytorch-oiio:latest jupyter lab --notebook-dir /workspace/git/nst-temporal --allow-root
