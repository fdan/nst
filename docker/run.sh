#! /usr/bin/bash

sudo docker run --gpus all --rm -it -p 8888:8888 -v /home/$USER/git:/workspace/git --net=host pytorch-oiio:latest jupyter lab --allow-root
