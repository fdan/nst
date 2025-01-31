FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    protobuf-compiler \
    libglib2.0-0 \
    libjpeg-dev \
    libopencv-dev \
    libopenexr-dev \
    libpng-dev \
    libsm-dev \
    vim \
    wget \
    emacs

# Install Python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3.8-dev && \
    rm -rf /var/lib/apt/lists/*

# Have aliases python3->python and pip3->pip
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install --no-cache-dir setuptools wheel && \
    python3.8 -m pip install --no-cache-dir \
    opencv-python \
    future \
    gast==0.2.2 \
    protobuf==3.0.0 \
    setuptools \
    pyyaml==3.13 \
    scikit-image \
    typing \
    imageio \
    OpenEXR

RUN python3.8 -m pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install --no-cache-dir cupy-cuda110
RUN python3.8 -m pip install --no-cache-dir cython
RUN python3.8 -m pip install --no-cache-dir kornia

# set env vars for nst
ENV PATH /workspace/git/nst/bin:$PATH
ENV PYTHONPATH /workspace/git/nst/python:$PYTHONPATH
ENV NST_VGG_MODEL /workspace/git/nst/bin/Models/vgg_conv.pth
ENV ML_SERVER_DIR /workspace/git/nst/python/nst/nuke/server
