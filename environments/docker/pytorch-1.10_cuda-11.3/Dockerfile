FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y \
  cmake \
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
  libboost-all-dev \
  libjpeg-dev \
  libtiff-dev \
  libpng-dev \
  libilmbase-dev \
  libopenexr-dev \
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

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda
RUN conda create -y --name pytorch
RUN conda install -n pytorch -y -c pytorch pytorch=1.10 torchvision cudatoolkit=11 memory_profiler opencv matplotlib jupyterlab scikit-learn scikit-image imageio
RUN conda install -n pytorch -y pybind11
ENV PATH /opt/conda/envs/pytorch/bin:$PATH
RUN echo "source activate pytorch" >> ~/.bashrc
RUN echo "alias la='ls -lah'" >> ~/.bashrc
RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN /bin/bash -c "source activate pytorch"

# compile OCIO pybindings using anaconda3 python
RUN cd /tmp && \
  wget https://github.com/AcademySoftwareFoundation/OpenColorIO/archive/refs/heads/RB-1.1.zip && \
  unzip RB-1.1.zip && cd OpenColorIO-RB-1.1 && \
  mkdir build && \
  cd build && \
  /usr/bin/cmake -DPYTHON_LIBRARY=/opt/conda/envs/pytorch/lib -DPYTHON_INCLUDE_DIR=/opt/conda/envs/pytorch/include -DOCIO_BUILD_TESTS=OFF -DOCIO_BUILD_DOCS=OFF -DOCIO_BUILD_APPS=OFF -DCMAKE_INSTALL_PREFIX=/opt/ocio .. && \
  make -j12 && \
  make install -j12 && \
  cd /tmp && \
  rm -rf OpenColorIO-RB-1.1 && \
  rm RB-1.1.zip
  
# compile OIIO pybindings using anaconda3 python
RUN cd /tmp && \
  wget https://github.com/OpenImageIO/oiio/archive/refs/tags/v2.2.15.1.tar.gz && \
  tar -xvf v2.2.15.1.tar.gz && \
  cd oiio-2.2.15.1 && \
  mkdir build && \
  cd build && \
  /usr/bin/cmake -DUSE_PYTHON=1 -DOIIO_BUILD_TESTS=0 -DOIIO_BUILD_TOOLS=1 -DPYTHON_VERSION=3.7 -DCMAKE_INSTALL_PREFIX=/opt/oiio -DOPENCOLORIO_INCLUDE_PATH=/opt/ocio/include -DOPENCOLORIO_LIBRARY_PATH=/opt/ocio/lib .. && \
  make -j12 && \
  make install -j12 && \
  cd /tmp && \
  rm -rf oiio-2.2.15.1 && \
  rm v2.2.15.1.tar.gz

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
ENV PYTHONPATH /workspace/git/nst/python:/opt/oiio/lib/python3.7/site-packages:$PYTHONPATH
ENV NST_VGG_MODEL /workspace/git/nst/bin/Models/vgg_conv.pth

# set $OCIO
ENV OCIO /workspace/git/OpenColorIO-Configs-master/aces_1.0.3/config.ocio

ENV MPLCONFIGDIR /mnt/ala/tmp