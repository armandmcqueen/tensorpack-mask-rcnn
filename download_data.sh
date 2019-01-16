#!/usr/bin/env bash

DATA_DIR=/home/ubuntu/data

mkdir -p $DATA_DIR
aws s3 cp s3://armand-ajay-workshop/mask-rcnn/sagemaker/input/train $DATA_DIR --recursive

wget -O $DATA_DIR/pretrained-models/ImageNet-R50-AlignPadding.npz http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz
