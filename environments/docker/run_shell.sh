#! /usr/bin/bash

docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git --user $(id -u):$(id -g) --group-add users --net=host ffdanff/pytorch-oiio:latest
#docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git --net=host ffdanff/pytorch-oiio:latest
