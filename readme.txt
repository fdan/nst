
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


