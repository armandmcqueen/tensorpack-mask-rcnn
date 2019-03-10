# -*- coding: utf-8 -*-

import tensorflow as tf


from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.tower import get_current_tower_context

from MaskRCNN.config import config as cfg





@under_name_scope()
def clip_boxes_workaround(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes



@under_name_scope()
def generate_rpn_proposals_workaround(boxes, scores, img_shape,
                           pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        boxes: nx4 float dtype, the proposal boxes. Decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    topk = tf.minimum(pre_nms_topk, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes_workaround(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > cfg.RPN.MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=post_nms_topk,
        iou_threshold=cfg.RPN.PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)
    proposal_scores = tf.gather(topk_valid_scores, nms_indices)
    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')


@under_name_scope()
def generate_fpn_proposals_workaround(
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

                proposal_boxes, proposal_scores = generate_rpn_proposals_workaround(
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
        proposal_boxes, proposal_scores = generate_rpn_proposals_workaround(
            all_boxes, all_scores, image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if training else cfg.RPN.TEST_POST_NMS_TOPK)

    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), \
        tf.stop_gradient(proposal_scores, name='scores')


