
instructions

1. get address of ml server (hostname -I).  for singularity it will be the same as the host.  for docker, each container will have it's own.
2. launch ML server from docker or singularity
3. lauch a rez env for nuke containing the modified ml cient plugin (re-env nst).  launch nuke
4. try a local preview nst with either adam optimiser, or lower resolution
5. send some jobs to the farm via nuke

* where did I get to with nuke farm sub?


...

how to generate a new messageLive_pb2.py

this can be necessary if the protobuf version changes.  the docker image itself contains the protobuf compiler, use like so:

/workspace/git/nst/nuke/ml-client-live/proto# protoc --python_out=/workspace/git/nst/nuke/ml-client-live/proto/output messageLive.proto


...

BUILDING FOR UBUNTU 20.04

https://github.com/TheFoundryVisionmongers/nuke-ML-server/issues/15

NOTE: the nuke install dir must be on $LD_LIBRARY_PATH

** INSTALL GCC-6 **
1. to the file /etc/apt/sources.list add the line:
deb http://dk.archive.ubuntu.com/ubuntu/ bionic main universe

2. sudo apt update
autp apt install g++-6 gcc-6

** BUILD PROTOBUF **
https://github.com/protocolbuffers/protobuf/blob/v3.5.1/src/README.md

wget https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.5.1.tar.gz
tar -xvf v3.5.1.tar.gz
cd protobuf-3.5.1
./configure CXX=g++-6 CC=gcc-6 CPP=cpp-6 CXXCPP=cpp-6 CXXFLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'
make
make check
sudo make install
sudo ldconfig

** cmake flags for nst **
in CMakeLists.txt

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)

** RUN NUKE **

export NUKE_PATH=/home/dan/git/nst/nuke/build/ml-client-live/

** PORT FORWARDING **

in a terminal:
ssh -L 12345:serverIP:55555 username@host

in nuke mlc:
server: 127.0.0.1
port: 12345


