#!/usr/bin/env bash

# Set timestamp and logging directory, begin writing to it.
TS=`date +'%Y%m%d_%H%M%S'`
LOG_DIR=/home/ubuntu/logs/train_log_${TS}
#LOG_DIR=/tmp/logs
mkdir -p ${LOG_DIR}
exec &> >(tee ${LOG_DIR}/nohup.out)

# Print evaluated script commands
set -x

# Set VENV
VENV=${CONDA_DEFAULT_ENV}

# Write current branch and commit hash to log directory
git branch | grep \* | awk '{print $2}' > ${LOG_DIR}/git_info
git log | head -1 >> ${LOG_DIR}/git_info
git diff >> ${LOG_DIR}/git_info

# Copy this script into logging directory
cp `basename $0` ${LOG_DIR}

# Record environment variables
env > ${LOG_DIR}/env.txt

# Record python libaries
pip freeze > ${LOG_DIR}/requirements.txt

# Record tensorflow shared object linkages (CUDA version?)
ldd /home/ubuntu/anaconda3/envs/${VENV}/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so > ${LOG_DIR}/tf_so_links.txt

# Execute training job
#HOROVOD_TIMELINE=${LOG_DIR}/htimeline.json \
#-x TENSORPACK_FP16 \
#TENSORPACK_FP16=1 \
HOROVOD_CYCLE_TIME=0.5 \
HOROVOD_FUSION_THRESHOLD=67108864 \
/home/ubuntu/anaconda3/envs/${VENV}/bin/mpirun -np 8 -H localhost:8 \
-wdir /home/ubuntu/tensorpack-mask-rcnn \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH -x PATH \
-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \
--output-filename ${LOG_DIR}/mpirun_logs \
/home/ubuntu/anaconda3/envs/${VENV}/bin/python3 -m MaskRCNN_no_batch.train \
--logdir ${LOG_DIR} \
--perf \
--throughput_log_freq 2000 \
--images_per_step 8 \
--summary_period 25 \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/home/ubuntu/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
BACKBONE.WEIGHTS=/home/ubuntu/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAINER=horovod
