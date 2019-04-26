#!/usr/bin/env bash


echo "Running docker image tensorpack-mask-rcnn:bai"
echo ""



nvidia-docker run -it  -v ~/data:/data -v ~/logs:/logs tensorpack-mask-rcnn:bai