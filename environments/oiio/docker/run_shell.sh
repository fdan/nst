#! /usr/bin/bash

#docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git --user $(id -u):$(id -g) --group-add users --net=host ffdanff/pytorch-oiio:v06
docker run --gpus all --rm -it -p 8887:8887 --volume /home/$USER/git:/workspace/git ffdanff/nst-oiio:v3
