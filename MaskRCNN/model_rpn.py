# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.models import Conv2D, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from config import config as cfg
from model_box import clip_boxes
from utils.mixed_precision import mixed_precision_scope

@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors, fp16=False):
    """
    Returns:
        label_logits: BS x fH x fW x NA
        box_logits: BS x (NAx4) x fH x fW
    """
    # featuremap = print_runtime_shape("featuremap", featuremap, prefix="rpn_head")
    prefix = "rpn_head"

    if fp16:
        featuremap = tf.cast(featuremap, tf.float16)

    with mixed_precision_scope(mixed=fp16):
        with argscope(Conv2D, data_format='channels_first',
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
            hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu)

            label_logits = Conv2D('class', hidden, num_anchors, 1)
            box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
            # BS, NA(*4), im/16, im/16 (NCHW)

            # label_logits = print_runtime_shape("label_logits", label_logits, prefix=prefix)
            # box_logits = print_runtime_shape("box_logits", box_logits, prefix=prefix)


            label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # BS x fH x fW x NA

            # shp = tf.shape(box_logits)  # BS x (NAx4) x fH x fW
            # box_logits = print_runtime_shape("box_logits", box_logits, prefix="rpn_head")
            # box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # BS x fH x fW x (NAx4)
            # box_logits = tf.reshape(box_logits, tf.stack([shp[0], shp[2], shp[3], num_anchors, 4]))  # BS x fH x fW x NA x 4

    if fp16:
        label_logits = tf.cast(label_logits, tf.float32)
        box_logits = tf.cast(box_logits, tf.float32)

    return label_logits, box_logits

############################################################################################

@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """
    Args:
        anchor_labels: fHxfWxNA
        anchor_boxes: fHxfWxNAx4, encoded
        label_logits:  fHxfWxNA
        box_logits: fHxfWxNAx4

    Returns:
        label_loss, box_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    # with tf.name_scope('label_metrics'):
    #     valid_label_prob = tf.nn.sigmoid(valid_label_logits)
    #     summaries = []
    #     with tf.device('/cpu:0'):
    #         for th in [0.5, 0.2, 0.1]:
    #             valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
    #             nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
    #             pos_prediction_corr = tf.count_nonzero(
    #                 tf.logical_and(
    #                     valid_label_prob > th,
    #                     tf.equal(valid_prediction, valid_anchor_labels)),
    #                 dtype=tf.int32)
    #             placeholder = 0.5   # A small value will make summaries appear lower.
    #             recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
    #             recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
    #             precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
    #             precision = tf.where(tf.equal(nr_pos_prediction, 0),
    #                                  placeholder, precision, name='precision_th{}'.format(th))
    #             summaries.extend([precision, recall])
    #     add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter
    placeholder = 0.
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits)
    label_loss = tf.reduce_sum(label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / cfg.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')

    # add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]



