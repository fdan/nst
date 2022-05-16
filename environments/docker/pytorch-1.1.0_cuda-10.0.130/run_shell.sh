#! /usr/bin/bash

docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git --user $(id -u):$(id -g) --group-add users --net=host ffdanff/pytorch-oiio:latest
#docker run --gpus all --rm -it -p 8888:8888 --volume /mnt/ala/research/danielf/2021/git:/workspace/git --user $(id -u):$(id -g) --group-add users --net=host ffdanff/pytorch-oiio:latest

