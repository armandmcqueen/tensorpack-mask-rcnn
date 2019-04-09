#!/usr/bin/env bash

BRANCH_NAME=${1:-"master"}

echo "Building docker image tensorpack-mask-rcnn:dev-${BRANCH_NAME}"
echo ""



docker build -t tensorpack-mask-rcnn:dev-${BRANCH_NAME} .. --build-arg CACHEBUST=$(date +%s) --build-arg BRANCH_NAME=${BRANCH_NAME}