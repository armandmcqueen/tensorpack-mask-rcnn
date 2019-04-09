# -*- coding: utf-8 -*-

import itertools
import numpy as np
import tensorflow as tf

from tensorpack.models import Conv2D, FixedUnPooling, MaxPooling, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.tower import get_current_tower_context

from basemodel import GroupNorm
from config import config as cfg
from model_box import roi_align
from model_rpn import generate_rpn_proposals, rpn_losses, rpn_losses_batch_iterative
from utils.box_ops import area as tf_area
from utils.mixed_precision import mixed_precision_scope


@layer_register(log_shape=True)
def fpn_model(features, fp16=False):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features
    num_channel = cfg.FPN.NUM_CHANNEL

    use_gn = cfg.FPN.NORM == 'GN'

    def upsample2x(name, x):
        dtype_str = 'float16' if fp16 else 'float32'
        return FixedUnPooling(
            name, x, 2, unpool_mat=np.ones((2, 2), dtype=dtype_str),
            data_format='channels_first')

        # tf.image.resize is, again, not aligned.
        # with tf.name_scope(name):
        #     shape2d = tf.shape(x)[2:]
        #     x = tf.transpose(x, [0, 2, 3, 1])
        #     x = tf.image.resize_nearest_neighbor(x, shape2d * 2, align_corners=True)
        #     x = tf.transpose(x, [0, 3, 1, 2])
        #     return x
    
    with mixed_precision_scope(mixed=fp16):
      with argscope(Conv2D, data_format='channels_first',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=1.)):
        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1)
                    for i, c in enumerate(features)]
        if use_gn:
            lat_2345 = [GroupNorm('gn_c{}'.format(i + 2), c) for i, c in enumerate(lat_2345)]
        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3)
                 for i, c in enumerate(lat_sum_5432[::-1])]
        if use_gn:
            p2345 = [GroupNorm('gn_p{}'.format(i + 2), c) for i, c in enumerate(p2345)]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first', padding='VALID')
        
        if fp16:
            return [tf.cast(l, tf.float32) for l in p2345] + [tf.cast(p6, tf.float32)]

        return p2345 + [p6]

@under_name_scope()
def fpn_map_rois_to_levels_batch(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes (nx4):

    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes[:,1:]))
    level = tf.cast(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),   # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


@under_name_scope()
def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes (nx4):

    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.cast(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),   # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


@under_name_scope()
def multilevel_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            boxes_on_featuremap = boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

            # roi_feature_maps = tf.roi_align(featuremap,
            #                                 boxes,
            #                                 pooled_height=resolution,
            #                                 pooled_width=resolution,
            #                                 spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[i],
            #                                 sampling_ratio=2)
            # all_rois.append(roi_feature_maps)

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois




@under_name_scope(name_scope="multilevel_roi_align")
def multilevel_roi_align_tf_op(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels_batch(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            #boxes_on_featuremap = boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            #all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

#            if boxes.shape.dims[1].value == 4:                                                               # REMOVE WHEN COMPLETELY BATCHIFIED
#                boxes = tf.concat((tf.zeros([tf.shape(boxes)[0], 1], dtype=tf.float32), boxes), axis=1)      # REMOVE WHEN COMPLETELY BATCHIFIED

            # coordinate system fix for boxes
            boxes = tf.concat((boxes[:,:1], boxes[:,1:] - 0.5*cfg.FPN.ANCHOR_STRIDES[i]), axis=1) 

            roi_feature_maps = tf.roi_align(featuremap,
                                            boxes,
                                            pooled_height=resolution,
                                            pooled_width=resolution,
                                            spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[i],
                                            sampling_ratio=2)
            all_rois.append(roi_feature_maps)

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois


def multilevel_rpn_losses_batch_fixed_single_image(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Args:
        multilevel_anchors: #lvl RPNAnchors
        multilevel_label_logits: #lvl tensors of shape HxWxA
        multilevel_box_logits: #lvl tensors of shape HxWxAx4
    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('single_image_rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl], multilevel_box_logits[lvl],
                name_scope='level{}'.format(lvl + 2))
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2])
        total_box_loss = tf.add_n(losses[1::2])
    return [total_label_loss, total_box_loss]


def multilevel_rpn_losses(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Args:
        multilevel_anchors: #lvl RPNAnchors
        multilevel_label_logits: #lvl tensors of shape HxWxA
        multilevel_box_logits: #lvl tensors of shape HxWxAx4

    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl], multilevel_box_logits[lvl],
                name_scope='level{}'.format(lvl + 2))
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2], name='label_loss')
        total_box_loss = tf.add_n(losses[1::2], name='box_loss')
        add_moving_summary(total_label_loss, total_box_loss)
    return [total_label_loss, total_box_loss]





def multilevel_rpn_losses_batch(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Args:
        multilevel_anchors: #lvl RPNAnchors (batch formulation)
        multilevel_label_logits: #lvl tensors of shape BS x H x W x A
        multilevel_box_logits: #lvl tensors of shape BS x (A*4) x H x W

    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]

            label_loss, box_loss = rpn_losses_batch_iterative(
                    anchors.gt_labels,
                    anchors.encoded_gt_boxes(),
                    multilevel_label_logits[lvl],
                    multilevel_box_logits[lvl],
                    name_scope='level{}'.format(lvl + 2),
                    print_anchor_tensors=lvl == 0)
            #
            # label_loss, box_loss = rpn_losses_batch(
            #     anchors.gt_labels,
            #     anchors.encoded_gt_boxes(),
            #     multilevel_label_logits[lvl],
            #     multilevel_box_logits[lvl],
            #     name_scope='level{}'.format(lvl + 2),
            #     print_anchor_tensors=lvl==0)
            #

            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2], name='label_loss')
        total_box_loss = tf.add_n(losses[1::2], name='box_loss')
        add_moving_summary(total_label_loss, total_box_loss)
    return [total_label_loss, total_box_loss]





@under_name_scope()
def generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d):
    """
    Args:
        multilevel_pred_boxes: #lvl HxWxAx4 boxes
        multilevel_label_logits: #lvl tensors of shape HxWxA

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_pred_boxes) == num_lvl
    assert len(multilevel_label_logits) == num_lvl

    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]

                proposal_boxes, proposal_scores = generate_rpn_proposals(
                    tf.reshape(pred_boxes_decoded, [-1, 4]),
                    tf.reshape(multilevel_label_logits[lvl], [-1]),
                    image_shape2d, fpn_nms_topk)
                all_boxes.append(proposal_boxes)
                all_scores.append(proposal_scores)

        proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
        proposal_scores = tf.concat(all_scores, axis=0)  # n
        # Here we are different from Detectron.
        # Detectron picks top-k within the batch, rather than within an image. However we do not have a batch.
        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)
    else:
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]
                all_boxes.append(tf.reshape(pred_boxes_decoded, [-1, 4]))
                all_scores.append(tf.reshape(multilevel_label_logits[lvl], [-1]))
        all_boxes = tf.concat(all_boxes, axis=0)
        all_scores = tf.concat(all_scores, axis=0)
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            all_boxes, all_scores, image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if training else cfg.RPN.TEST_POST_NMS_TOPK)

    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), \
        tf.stop_gradient(proposal_scores, name='scores')


@under_name_scope()
def generate_fpn_proposals_batch_tf_op(multilevel_anchor_boxes,
        multilevel_box_logits, multilevel_label_logits, orig_image_dims):
    """
    Args:
        multilevel_box_logits:      #lvl [ BS x (NAx4) x H x W ] boxes
        multilevel_label_logits:    #lvl [ BS x H x W x A ] tensors
        orig_image_dimensions: Original (prepadding) image dimensions (h,w,c)   BS x 3

    Returns:
        boxes: K x 5 float
        scores:  (#lvl x BS x K) vector       (logits)
    """
    prefix = "model_fpn.generate_fpn_proposals_batch_tf_op"
    bug_prefix = "GEN_PROPOSALS_BUG fpn"
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_label_logits) == num_lvl
    orig_images_hw = orig_image_dims[:, :2]


    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK
        for lvl in range(num_lvl):
            with tf.name_scope(f'Lvl{lvl}'):
                im_info = tf.cast(orig_images_hw, tf.float32)
                # h, w

                label_logits = multilevel_label_logits[lvl]
                # label_logits = print_runtime_shape(f'label_logits, lvl{lvl}', label_logits, prefix=bug_prefix)
                #scores = tf.transpose(label_logits, [0, 3, 1, 2])
                scores = label_logits

                #box_logits = multilevel_box_logits[lvl] # N(A4)HW
                box_logits = tf.transpose(multilevel_box_logits[lvl],[0, 2, 3, 1])



                bbox_deltas = box_logits
                # bbox_deltas = print_runtime_shape(f'bbox_deltas (pre-reshape), lvl {lvl}', bbox_deltas, prefix=bug_prefix)



                single_level_anchor_boxes = multilevel_anchor_boxes[lvl]
                shp = tf.shape(single_level_anchor_boxes)
                single_level_anchor_boxes = tf.reshape(single_level_anchor_boxes, (-1, 4))
                #print("single_level_anchor_boxes", single_level_anchor_boxes)
                """

                area = cfg.RPN.ANCHOR_SIZES[lvl] ** 2
                anchor_list = []
                for ratio in cfg.RPN.ANCHOR_RATIOS:
                    # ratio = h/w
                    # h = w x ratio

                    # area = h x w
                    # h = area / w

                    # w x ratio = area / w
                    # w = sqrt (area / ratio)

                    # h = w * ratio

                    w = (area / ratio) ** 0.5
                    h = w * ratio
                    x1 = -1 * (w / 2.)
                    y1 = -1 * (h / 2.)
                    x2 = -x1
                    y2 = -y1

                    anchor = tf.constant([x1, y1, x2, y2], dtype=tf.float32)
                    anchor_list.append(anchor)
                # print(anchor_list)
                anchors = tf.stack(anchor_list)

                """


                # https://caffe2.ai/docs/operators-catalogue.html#generateproposals


                rois, rois_probs = tf.generate_bounding_box_proposals(scores,
                                                                   bbox_deltas,
                                                                   im_info,
                                                                   single_level_anchor_boxes,
                                                                   spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[lvl],
                                                                   pre_nms_topn=fpn_nms_topk,
                                                                   post_nms_topn=fpn_nms_topk,
                                                                   nms_threshold=cfg.RPN.PROPOSAL_NMS_THRESH,
                                                                   min_size=cfg.RPN.MIN_SIZE)

                # rois_probs = print_runtime_shape(f'rois_probs, lvl {lvl}', rois_probs, prefix=bug_prefix)
                all_boxes.append(rois)
                all_scores.append(rois_probs)


        proposal_boxes = tf.concat(all_boxes, axis=0)  # (#lvl x BS) x K x 5
        proposal_boxes = tf.reshape(proposal_boxes, [-1, 5])        # (#lvl x BS x K) x 5

        proposal_scores = tf.concat(all_scores, axis=0)  # (#lvl x BS) x K
        proposal_scores = tf.reshape(proposal_scores, [-1])         # (#lvl x BS x 5) vector

        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)

    else:
        raise RuntimeError("Only level-wise predictions are supported with batches")

    return tf.stop_gradient(proposal_boxes, name='boxes'), \
        tf.stop_gradient(proposal_scores, name='scores')
