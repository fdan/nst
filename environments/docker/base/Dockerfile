FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y \
  libopenjp2-7-dev \
  libsquish-dev \
  libraw-dev \
  libjpeg-turbo8-dev \
  libgif-dev \
  libdcmtk-dev \
  libtinyxml-dev \
  liblcms2-dev \
  libyaml-cpp-dev \
  unzip \
  wget \
  software-properties-common \
  pkg-config \
  cmake \
  git \
  libboost-all-dev \
  libturbojpeg0-dev \
  libopenjp2-7-dev \
  libjpeg-dev \
  libtiff-dev \
  libpng-dev \
  libilmbase-dev \
  libopenexr-dev \
  python-numpy \
  bzip2 \
  ca-certificates \
  libglib2.0-0 \
  libxext6 \
  libsm6 \
  libxrender1 \
  mercurial \
  emacs \
  subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda
RUN conda create -y --name pytorch
RUN conda install -n pytorch -y -c pytorch pytorch torchvision cudatoolkit=11 memory_profiler opencv matplotlib jupyterlab
RUN conda install -n pytorch -y pybind11
ENV PATH /opt/conda/envs/pytorch/bin:$PATH
RUN echo "source activate pytorch" >> ~/.bashrc
RUN echo "alias la='ls -lah'" >> ~/.bashrc
RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN /bin/bash -c "source activate pytorch"

RUN pip install protobuf

RUN mkdir -p /.local/share/jupyter/runtime/
RUN chmod -R 777 /.local
RUN mkdir -p /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
RUN chmod -R 777 /.jupyter
RUN touch /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
RUN echo "{\"theme\": \"JupyterLab Dark\"}" >> /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings



