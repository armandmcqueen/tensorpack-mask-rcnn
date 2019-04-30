# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.models import Conv2D, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from config import config as cfg
from model_box import clip_boxes
from utils.mixed_precision import mixed_precision_scope
from performance import print_buildtime_shape, print_runtime_shape, print_runtime_tensor
import time

@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors, fp16=False):
    """
    The RPN head that takes the feature map from the FPN and outputs bounding box logits.

    For every pixel on the feature maps, there are a certain number of anchors.
    The output will be:
    label logits: indicate whether there is an object for a certain anchor in one pixel
    box logits: The encoded box logits from fast-rccn paper https://arxiv.org/abs/1506.01497
                page 5, in order to be consistent with the ground truth encoded boxes

    Args:
        featuremap: feature map for a single FPN layer, i.e. one from P23456, BS x NumChannel x H x W
        channel: # channels of the feature map, scalar, default 256
        num_anchors(NA): # of anchors for each pixel in the current feature map, scalar, default 3
    Returns:
        label_logits: BS x H x W x NA
        box_logits: BS x (NA * 4) x H x W, encoded
    """
    if fp16:
        featuremap = tf.cast(featuremap, tf.float16)

    with mixed_precision_scope(mixed=fp16):
        with argscope(Conv2D, data_format='channels_first',
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
            hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu)
            # BS x N_channels x H x W
            label_logits = Conv2D('class', hidden, num_anchors, 1)
            # BS x NA x H x W
            box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
            # BS x (NA*4) x H x W

            label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # BS x H x W x NA

    if fp16:
        label_logits = tf.cast(label_logits, tf.float32)
        box_logits = tf.cast(box_logits, tf.float32)

    return label_logits, box_logits


@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """
    Calculate the rpn loss for one FPN layer for a single image.
    The ground truth(GT) anchor labels and anchor boxes has been preprocessed to fit
    the dimensions of FPN feature map. The GT boxes are encoded from fast-rcnn paper
    https://arxiv.org/abs/1506.01497 page 5.

    Args:
        anchor_labels: GT anchor labels, H x W x NA
        anchor_boxes: GT boxes for each anchor, H x W x NA x 4, encoded
        label_logits: label logits from the rpn head, H x W x NA
        box_logits: box logits from the rpn head, H x W x NA x 4
    Returns:
        label_loss, box_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
#        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
#        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
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
    label_loss = tf.where(tf.equal(tf.size(valid_anchor_labels), 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / cfg.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(tf.size(pos_anchor_boxes), 0), placeholder, box_loss, name='box_loss')

    # add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]
