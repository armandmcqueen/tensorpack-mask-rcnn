#!/usr/bin/env bash
NUM_GPU=${1:-1}
SUMMARY_PERIOD=${2:-0}

let STEPS_PER_EPOCH=120000/${NUM_GPU}

echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "STEPS_PER_EPOCH: ${STEPS_PER_EPOCH}"
echo "SUMMARY_PERIOD: ${SUMMARY_PERIOD}"
echo ""


# --use_nobatch_pipeline

TENSORPACK_FP16=1 \
HOROVOD_TIMELINE=/logs/htimeline.json \
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
--output-filename /logs/mpirun_logs \
/usr/local/bin/python3 /tensorpack-mask-rcnn/MaskRCNN_no_batch_convergence/train.py \
--logdir /logs/train_log \
--perf \
--fp16 \
--summary_period $SUMMARY_PERIOD \
--throughput_log_freq 2000 \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.STEPS_PER_EPOCH=${STEPS_PER_EPOCH} \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
TRAIN.EVAL_PERIOD=12 \
BACKBONE.WEIGHTS=/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAINER=horovod