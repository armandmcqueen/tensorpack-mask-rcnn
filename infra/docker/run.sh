#!/usr/bin/env bash

BRANCH_NAME=${1:-"master"}

echo "Running docker image tensorpack-mask-rcnn:dev-${BRANCH_NAME}"
echo ""



nvidia-docker run -it  -v ~/data:/data -v ~/logs:/logs tensorpack-mask-rcnn:dev-${BRANCH_NAME}