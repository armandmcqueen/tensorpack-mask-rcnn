#!/usr/bin/env bash

IMAGE_NAME=tensorpack-mask-rcnn
IMAGE_TAG=dev
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} . --build-arg CACHEBUST=$(date +%s)