# Mask RCNN

Performance focused implementation of Mask RCNN based on the [Tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

### Overview

This implementation of Mask RCNN is focused on increasing training throughput without sacrificing any accuracy. We do this by training with a batch size > 1 per GPU using FP16 and two custom TF ops. 

### Status

Training on N GPUs (V100s) with a per-gpu batch size of M = NxM training

Training converges to target accuracy for configurations from 8x1 up to 32x4 training. Training throughput is substantially improved from original Tensorpack code.

### Notes

- Running this codebase requires a custom TF binary - available under GitHub releases (custom ops and fix for bug introduced in TF 1.13
- We give some details the codebase and optimizations in `CODEBASE.md`

### Tensorpack todo

- Port TensorSpec changes.

### Tensorpack fork point

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6
