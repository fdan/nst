FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=Ampere

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
  unzip \
  emacs

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# Have aliases python3->python and pip3->pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install --no-cache-dir setuptools wheel && \
    python3.10 -m pip install --no-cache-dir \
    future \
    gast==0.2.2 \
    protobuf==3.0.0 \
    setuptools \
    pyyaml==3.13 \
    scikit-image \
    typing \
    imageio \
    numba

RUN python3.10 -m pip install --no-cache-dir torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.10 -m pip install --no-cache-dir cupy-cuda11x
RUN python3.10 -m pip install --no-cache-dir cython
RUN python3.10 -m pip install --no-cache-dir kornia==0.6.12
RUN python3.10 -m pip install --no-cache-dir opencv-python
RUN python3.10 -m pip install --no-cache-dir memory_profiler
RUN python3.10 -m pip install --no-cache-dir matplotlib
RUN python3.10 -m pip install --no-cache-dir jupyterlab
RUN python3.10 -m pip install --no-cache-dir pybind11
RUN python3.10 -m pip install --no-cache-dir ninja
RUN python3.10 -m pip install --no-cache-dir torch-lucent
RUN python3.10 -m pip install --no-cache-dir pytorch_histogram_matching

#RUN wget --quiet https://github.com/fdan/NeuralTextureSynthesis/archive/refs/heads/nomedia.zip && \
#    unzip nomedia.zip && \
#    cd NeuralTextureSynthesis-nomedia && \
#    python setup.py install

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

# necessary for pytorch extensions:
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib