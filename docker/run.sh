#!/usr/bin/env bash

nvidia-docker run -it  -v ~/data:/data -v ~/logs:/logs tensorpack-mask-rcnn:dev