#!/usr/bin/env bash

source activate tensorflow_p36

pip install --upgrade pip
pip install ujson
pip install opencv-python
pip install pycocotools
pip install tensorpack
pip install --ignore-installed numpy==1.14.5