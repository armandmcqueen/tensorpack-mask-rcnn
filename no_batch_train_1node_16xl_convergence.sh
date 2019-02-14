#!/usr/bin/env bash

# Set timestamp, virtualenv, and logging directory
TS=`date +'%Y%m%d_%H%M%S'`
VENV=tensorflow_p36
LOG_DIR=/home/ubuntu/logs/train_log_${TS}

# Create log directory
mkdir -p ${LOG_DIR}

# Write current branch and commit hash to log directory
git branch | grep \* | awk '{print $2}' > ${LOG_DIR}/git_info
git log | head -1 >> ${LOG_DIR}/git_info

# Copy this script into logging directory
cp `basename $0` ${LOG_DIR}

# Execute training job
HOROVOD_TIMELINE=/home/ubuntu/logs/htimeline.json \
HOROVOD_CYCLE_TIME=0.5 \
HOROVOD_FUSION_THRESHOLD=67108864 \
/home/ubuntu/anaconda3/envs/${VENV}/bin/mpirun -np 8 -H localhost:8 \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH -x PATH \
-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \
--output-filename /home/ubuntu/logs/mpirun_logs \
/home/ubuntu/anaconda3/envs/${VENV}/bin/python3 /home/ubuntu/tensorpack-mask-rcnn/MaskRCNN-no-batch/train.py \
--logdir ${LOG_DIR} \
--perf \
--throughput_log_freq 2000 \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/home/ubuntu/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.STEPS_PER_EPOCH=15000 \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
BACKBONE.WEIGHTS=/home/ubuntu/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAINER=horovod
