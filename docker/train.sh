#!/usr/bin/env bash
NUM_GPU=${1:-1}
IMAGES_PER_GPU=${2:-1}
let IMAGES_PER_STEP=${IMAGES_PER_GPU}*${NUM_GPU}



echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "IMAGES_PER_GPU: ${IMAGES_PER_GPU}"
echo "IMAGES_PER_STEP: ${IMAGES_PER_STEP}"
echo ""


HOROVOD_TIMELINE=/tensorpack-mask-rcnn/logs/htimeline.json \
HOROVOD_CYCLE_TIME=0.5 \
HOROVOD_FUSION_THRESHOLD=67108864 \
/usr/local/bin/mpirun -np ${NUM_GPU} \
--H localhost:${NUM_GPU} \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH -x PATH \
-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \
--output-filename /tensorpack-mask-rcnn/logs/mpirun_logs \
/usr/local/bin/python3 /tensorpack-mask-rcnn/MaskRCNN/train.py \
--logdir /tensorpack-mask-rcnn/logs/train_log \
--perf \
--throughput_log_freq 1 \
--images_per_step ${IMAGES_PER_STEP} \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
BACKBONE.WEIGHTS=/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAINER=horovod \
TRAIN.BATCH_SIZE_PER_GPU=${IMAGES_PER_GPU}
