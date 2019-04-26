#!/usr/bin/env bash

BRANCH_NAME=${1:-"master"}

# The BRANCH_NAME refers to the git pull that happens inside of the Dockerfile
echo "Building docker image tensorpack-mask-rcnn:bai"
echo ""



docker build -t tensorpack-mask-rcnn:bai .. --build-arg CACHEBUST=$(date +%s) --build-arg BRANCH_NAME=bai