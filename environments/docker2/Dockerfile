FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y \
  libboost-all-dev \
  libjpeg-dev \
  libtiff-dev \
  libpng-dev \
  libilmbase-dev \
  libopenexr-dev \
  pybind11-dev \
  python-numpy \
  wget \
  git \
  bzip2 \
  ca-certificates \
  libglib2.0-0 \
  libxext6 \
  libsm6 \
  libxrender1 \
  mercurial \
  emacs \
  subversion

# latest OIIO requires cmake-3.12+, ubuntu uses older version
RUN cd /tmp && \
  wget https://github.com/Kitware/CMake/archive/v3.12.4.tar.gz && \
  tar -xvf v3.12.4.tar.gz && \
  cd CMake-3.12.4 && \
  ./bootstrap && \
  make -j12 && \
  make install -j12

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda
RUN conda create -y --name pytorch
RUN conda install -n pytorch -y -c pytorch pytorch torchvision cudatoolkit=11 memory_profiler opencv matplotlib jupyterlab
ENV PATH /opt/conda/envs/pytorch/bin:$PATH
RUN echo "source activate pytorch" >> ~/.bashrc
RUN echo "alias la='ls -lah'" >> ~/.bashrc
RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN /bin/bash -c "source activate pytorch"

# compile OIIO pybindings using anaconda3 python
RUN cd /tmp && \
  wget https://github.com/OpenImageIO/oiio/archive/Release-2.1.17.0.tar.gz && \
  tar -xvf Release-2.1.17.0.tar.gz && \
  cd oiio-Release-2.1.17.0/ && \  
    mkdir build && \
  cd build && \
  /usr/local/bin/cmake -DUSE_PYTHON=1 -DOIIO_BUILD_TESTS=0 -DOIIO_BUILD_TOOLS=0 -DPYTHON_VERSION=3.7 -DCMAKE_INSTALL_PREFIX=/opt/conda/envs/pytorch .. && \
  make -j12 && \
  make install -j12

#  for nuke-ML-server:
RUN pip install protobuf

RUN mkdir -p /.local/share/jupyter/runtime/
RUN chmod -R 777 /.local
RUN mkdir -p /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
RUN chmod -R 777 /.jupyter
RUN touch /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
RUN echo "{\"theme\": \"JupyterLab Dark\"}" >> /.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# set env vars for nst
ENV PATH /workspace/git/nst/bin:$PATH
ENV PYTHONPATH /workspace/git/nst/python:$PYTHONPATH
ENV NST_VGG_MODEL /workspace/git/nst/bin/Models/vgg_conv.pth
