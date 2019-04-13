# Mask RCNN

Performance focused implementation of Mask RCNN based on the [Tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

### Overview

Code with a batch dimension for training. Converges with batch size = 1. Convergence issue with bs > 1.

Requires a custom TF binary - available under GitHub releases.




### Tensorpack todo

- Port TensorSpec changes.

### Tensorpack fork point

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6