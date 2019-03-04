# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.models import Conv2D, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
from tensorpack.tfutils.summary import add_moving_summary


STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .config import config as cfg
    from .model_box import clip_boxes, clip_boxes_batch
    from .perf import print_runtime_shape, print_buildtime_shape, runtime_print, print_runtime_tensor
    from .utils.mixed_precision import mixed_precision_scope
else:
    from MaskRCNN.config import config as cfg
    from MaskRCNN.model_box import clip_boxes, clip_boxes_batch
    from MaskRCNN.perf import print_runtime_shape, print_buildtime_shape, runtime_print, print_runtime_tensor
    from MaskRCNN.utils.mixed_precision import mixed_precision_scope



@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors, fp16=False):
    """
    Returns:
        label_logits: BS x fH x fW x NA
        box_logits: BS x fH x fW x NA x 4
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

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(
                        valid_label_prob > th,
                        tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32)
                placeholder = 0.5   # A small value will make summaries appear lower.
                recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
                recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
                precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
                precision = tf.where(tf.equal(nr_pos_prediction, 0),
                                     placeholder, precision, name='precision_th{}'.format(th))
                summaries.extend([precision, recall])
        add_moving_summary(*summaries)

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

    add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]




@under_name_scope()
# TODO: [Armand] I'm not convinced that this code is correct.
def rpn_losses_batch(anchor_labels, anchor_boxes, label_logits, box_logits, print_anchor_tensors=False):
    """
    Args:
        anchor_labels: BS x fH x fW x NA
        anchor_boxes: BS x fH x fW x NA x 4, encoded
        label_logits: BS x fH x fW x NA
        box_logits: BS x fH x fW x NA x 4

    Returns:
        label_loss, box_loss
    """

    prefix = "rpn_losses_batch"

    # box_logits = print_runtime_shape("box_logits", box_logits, prefix=prefix)
    if print_anchor_tensors:
        anchor_labels = print_runtime_tensor("anchor_labels", anchor_labels, prefix=prefix, summarize=-1)
        anchor_boxes = print_runtime_tensor("anchor_boxes", anchor_boxes, prefix=prefix, summarize=-1)

    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))  # (?, ?, ?, ?)
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))         # (?, ?, ?, ?)

        # pos_mask = print_runtime_shape("pos_mask", pos_mask, prefix=prefix)

        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.
        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)    # (?, )

        print_buildtime_shape("valid_mask", valid_mask, prefix=prefix)
        print_buildtime_shape("pos_mask", pos_mask, prefix=prefix)
        print_buildtime_shape("valid_anchor_labels", valid_anchor_labels, prefix=prefix)

    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)          # (?, )
    print_buildtime_shape("valid_label_logits", valid_label_logits, prefix=prefix)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        print_buildtime_shape("valid_label_prob", valid_label_prob, prefix=prefix)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(
                        valid_label_prob > th,
                        tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32)
                placeholder = 0.5   # A small value will make summaries appear lower.
                recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
                recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
                precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
                precision = tf.where(tf.equal(nr_pos_prediction, 0),
                                     placeholder, precision, name='precision_th{}'.format(th))
                summaries.extend([precision, recall])
        add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter
    placeholder = 0.
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits)
    label_loss = tf.reduce_sum(label_loss) * (1. / (cfg.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)

    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / (cfg.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')

    box_loss = runtime_print("[TODO Armand] Not convinced that box and label loss are correct. Worried that "
                             "we are including anchors on top of padding/on the image boundary in loss function which, "
                             "according to the Faster RCNN paper, will lead to non-convergence", box_loss)

    add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]





@under_name_scope()
def rpn_losses_batch_iterative(anchor_labels, anchor_boxes, label_logits, box_logits, print_anchor_tensors=False):
    """
    Args:
        anchor_labels: BS x H x W x NA
        anchor_boxes: BS x H x W x NA x 4, encoded
        label_logits: BS x H x W x NA
        box_logits: BS x (A*4) x H x W



    Returns:
        label_loss, box_loss
    """

    prefix = "rpn_losses_batch_iterative"

    shp = tf.shape(box_logits)
    box_logits = tf.reshape(box_logits, [shp[0], shp[2], shp[3], -1, 4]) # BS x H x W x A x 4


    # box_logits = print_runtime_shape("box_logits", box_logits, prefix=prefix)
    for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
        #si_anchor_labels = anchor_labels[i, :, :, :]
        with tf.device('/cpu:0'):
            valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
            pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
            nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
            nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
            # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

            valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
        valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

        with tf.name_scope('label_metrics'):
            valid_label_prob = tf.nn.sigmoid(valid_label_logits)
            summaries = []
            with tf.device('/cpu:0'):
                for th in [0.5, 0.2, 0.1]:
                    valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                    nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                    pos_prediction_corr = tf.count_nonzero(
                            tf.logical_and(
                                    valid_label_prob > th,
                                    tf.equal(valid_prediction, valid_anchor_labels)),
                            dtype=tf.int32)
                    placeholder = 0.5  # A small value will make summaries appear lower.
                    recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
                    recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
                    precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
                    precision = tf.where(tf.equal(nr_pos_prediction, 0),
                                         placeholder, precision, name='precision_th{}'.format(th))
                    summaries.extend([precision, recall])
            add_moving_summary(*summaries)

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

        add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]



@under_name_scope()
def generate_rpn_proposals(boxes, scores, img_shape,
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
    topk_boxes = clip_boxes(topk_boxes, img_shape)

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
def generate_rpn_proposals_batch(boxes, scores, prepadding_dims,
                           pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        boxes:  [ BS x N x 4 ] float dtype, the proposal boxes. Decoded to floatbox already
        scores: [ BS x N ] float, the logits
        prepadding_dims: BS x 2, height and width of image prior to padding (scaled to feature map)
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: BS x K x 5       (5 = batchindex, floatbox)
        scores: BS x K
    """

    print_buildtime_shape("gen_rpn_prop.prepadding_dims", prepadding_dims)



    assert boxes.shape.ndims == 3, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk
    topk = tf.minimum(pre_nms_topk, tf.size(scores))


    bs = tf.shape(boxes)[0]

    b_pre_nms_topk = tf.tile([topk], [bs])
    b_post_nms_topk = tf.tile([post_nms_topk], [bs])
    b_indices = tf.range(bs)
    b_indices = tf.cast(b_indices, tf.float32)

    out = tf.map_fn(single_image_generate_rpn_proposals,
                    [boxes, scores, prepadding_dims, b_indices, b_pre_nms_topk, b_post_nms_topk],
                    dtype=(tf.float32, tf.float32, tf.int32),
                    back_prop=False,
                    name="mapfn_generate_fpn_proposals",
                    infer_shape=False)



    padded_proposal_boxes, padded_proposal_scores, actual_post_nms_topks = out


    all_actual_boxes = []
    all_actual_scores = []
    for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
        single_image_actual_boxes = padded_proposal_boxes[i]
        single_image_actual_scores = padded_proposal_scores[i]

        all_actual_boxes.append(single_image_actual_boxes)
        all_actual_scores.append(single_image_actual_scores)

    proposal_boxes = tf.concat(all_actual_boxes, axis=0)
    proposal_scores = tf.concat(all_actual_scores, axis=0)


    print_buildtime_shape("map_fn.out.proposal_boxes", proposal_boxes)
    print_buildtime_shape("map_fn.out.proposal_scores", proposal_scores)
    return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')




def single_image_generate_rpn_proposals(input_tensors):
    boxes, scores, img_shape, bs_idx, pre_nms_topk, post_nms_topk = input_tensors

    pre_nms_topk = tf.minimum(pre_nms_topk, tf.size(scores))
    orig_post_nms_topk = post_nms_topk
    actual_post_nms_topk = tf.minimum(post_nms_topk, tf.size(scores))

    print(actual_post_nms_topk)

    print_buildtime_shape("mapfn.boxes", boxes)
    print_buildtime_shape("mapfn.scores", scores)
    print_buildtime_shape("mapfn.img_shape", img_shape)
    print_buildtime_shape("mapfn.bs_idx", bs_idx)
    print_buildtime_shape("mapfn.pre_nms_topk", pre_nms_topk)
    print_buildtime_shape("mapfn.post_nms_topk", actual_post_nms_topk)

    # boxes = print_runtime_shape("mapfn.boxes", boxes)
    # scores = print_runtime_shape("mapfn.scores", scores)
    # img_shape = print_runtime_shape("mapfn.img_shape", img_shape)
    # pre_nms_topk = print_runtime_shape("mapfn.pre_nms_topk", pre_nms_topk)
    # actual_post_nms_topk = print_runtime_shape("mapfn.post_nms_topk", actual_post_nms_topk)

    topk_scores, topk_indices = tf.nn.top_k(scores, k=pre_nms_topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)

    # topk_boxes = print_runtime_shape("mapfn.topk_boxes", topk_boxes)
    topk_boxes = clip_boxes_batch(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))                        # K x 2 x 2
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)     # K x 1 x 2
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)                    # K x 2
    valid = tf.reduce_all(wbhb > cfg.RPN.MIN_SIZE, axis=1)  # n,                    #
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=actual_post_nms_topk,
        iou_threshold=cfg.RPN.PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)   # K x 4
    proposal_scores = tf.gather(topk_valid_scores, nms_indices) # K



    proposal_boxes = tf.pad(proposal_boxes, [[0, 0], [1, 0]], 'constant', constant_values=bs_idx)   # K x 5


    # Pad out to orig_post_nms_top_k so that map_fn is happy

    required_padding = orig_post_nms_topk - actual_post_nms_topk
    proposal_boxes = tf.pad(proposal_boxes, [[0, required_padding], [0, 0]], 'constant', constant_values=bs_idx)  # orig_post_nms_topk x 5
    proposal_scores = tf.pad(proposal_scores, [[0, required_padding]], 'constant', constant_values=bs_idx)  # orig_post_nms_topk vector

    # boxes = print_runtime_shape("mapfn.boxes", boxes)


    # tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return proposal_boxes, proposal_scores, actual_post_nms_topk
