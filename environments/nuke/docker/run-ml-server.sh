#! /usr/bin/bash

docker run --gpus all --rm -it -p 8888:8888 --volume /home/$USER/git:/workspace/git ffdanff/nst-nuke:latest python3.8 /workspace/git/nst/python/nst/nuke/server/server.py 55555




