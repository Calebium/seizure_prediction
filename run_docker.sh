#!/usr/bin/env bash

DOCKER_NAME=${1:-doc_pytorch}    # If arg1 does not exist use doc_pytorch as default
HOST_PORT=${2:-8888}             # If arg2 does not exist use port 8888 as default
DOCKER_TAG=${3:-20200429033528}  # If arg3 does not exist use tag 20200429033528 as default

DOCKER_USER=$(id -un)

# docker run -p option format:  -p [host_port]:[container_port]

docker run \
       --runtime=nvidia \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
       --rm -it \
       --name $DOCKER_NAME \
       -e DOCKER_NAME=$DOCKER_NAME \
       -v /home/$DOCKER_USER/data1/:/home/$DOCKER_USER/data1/ \
       -v /home/$DOCKER_USER/data1_shared/EEG_Data/:/home/$DOCKER_USER/EEG_Data/ \
       -p $HOST_PORT:8888 \
       $DOCKER_USER/docker-pytorch-1-4:$DOCKER_TAG
