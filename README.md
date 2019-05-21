# Mask RCNN

Performance focused implementation of Mask RCNN based on the [Tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

### Overview

Mask RCNN code with a batch dimension for training, FP16 training, custom TF ops. Some features of the original Tensorpack have been stripped away:

- We only support FPN-based training
- We do not have support for Cascade RCNN

### Status

Training on N GPUs (V100s) with a per-gpu batch size of M = NxM training

Training converges to target accuracy for configurations from 8x1 up to 32x4 training. 32x4 training is unreliable, with NaN losses ~50% of the experiments. 

Training throughput is substantially improved from original Tensorpack code.


### Notes

- Requires a custom TF binary - available under GitHub releases (custom ops and fix for bug introduced in TF 1.13

### Tensorpack todo

- Port TensorSpec changes.

### Tensorpack fork point

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6
