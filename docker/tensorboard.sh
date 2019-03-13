#!/usr/bin/env bash

# ssh -L 127.0.0.1:6006:127.0.0.1:6006 ubuntu@XXXXXXXX

tensorboard --logdir=live:~/logs/train_log,old:~/old_logs