#!/usr/bin/env bash

docker build -t tensorpack-mask-rcnn:dev .. --build-arg CACHEBUST=$(date +%s)