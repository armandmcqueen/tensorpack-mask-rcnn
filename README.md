# Mask RCNN

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6

## Folders

##### MaskRCNN

Converting code to batch. Work in progress, actively being worked on and broken.

##### MaskRCNN-no-batch

Original code with batch size = 1 img/GPU. Code should always run successfully - branch to make changes.


## pycocotools

Use NVIDIA optimized library from MLPerf

```
git clone -b v0.1 https://github.com/NVIDIA/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && pip install -e .
```

