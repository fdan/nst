FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y \
  build-essential \
  python3 \
  libopenimageio-dev \
  python3-openimageio \
  libopencolorio-dev \
  python3-pyopencolorio \
  protobuf-compiler \
  wget \
  git \
  ca-certificates \
  emacs

# Install Python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3.8-dev && \
    rm -rf /var/lib/apt/lists/*

# Have aliases python3->python and pip3->pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install --no-cache-dir setuptools wheel && \
    python3.8 -m pip install --no-cache-dir \
    future \
    gast==0.2.2 \
    protobuf==3.0.0 \
    setuptools \
    pyyaml==3.13 \
    scikit-image \
    typing \
    imageio

RUN python3.8 -m pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install --no-cache-dir cupy-cuda110
RUN python3.8 -m pip install --no-cache-dir cython
RUN python3.8 -m pip install --no-cache-dir kornia
RUN python3.8 -m pip install --no-cache-dir memory_profiler
RUN python3.8 -m pip install --no-cache-dir matplotlib
RUN python3.8 -m pip install --no-cache-dir jupyterlab

RUN mkdir -p /.local/share/jupyter/runtime/
RUN chmod -R 777 /.local
RUN mkdir -p /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
RUN chmod -R 777 /.jupyter
RUN touch /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
RUN echo "{\"theme\": \"JupyterLab Dark\"}" >> /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# set env vars for nst
ENV PATH /workspace/git/nst/bin:$PATH
ENV PYTHONPATH /workspace/git/nst/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH
ENV NST_VGG_MODEL /workspace/git/nst/bin/Models/vgg_conv.pth

# set $OCIO
ENV OCIO /workspace/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio

ENV MPLCONFIGDIR /mnt/ala/tmp