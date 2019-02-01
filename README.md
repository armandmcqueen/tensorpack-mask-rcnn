# Mask RCNN

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6

## pycocotools

Use NVIDIA optimized library from MLPerf

```
git clone -b v0.1 https://github.com/NVIDIA/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && pip install -e .
```

- Add to cocoapi/PythonAPI/coco.py:

```
try:
    import ujson as json
except ImportError:
    import json
```