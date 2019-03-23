# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf

from tensorpack.models import Conv2D, FullyConnected, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized_method

from basemodel import GroupNorm
from config import config as cfg
from model_box import decode_bbox_target, encode_bbox_target
from utils.box_ops import pairwise_iou, pairwise_iou_batch
from utils.mixed_precision import mixed_precision_scope

#from non_max_suppression_custom import non_max_suppression_custom

@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)


@under_name_scope(name_scope="proposal_metrics")
def proposal_metrics_batch(per_image_ious):
    """
    Add summaries for RPN proposals.
    Args:
        per_image_ios: list of len batch_size: nxm, #proposal x #gt
    """
    prefix="proposal_metrics_batch"

    # find best roi for each gt, for summary only
    thresholds = [0.3, 0.5]
    best_ious = []
    mean_best_ious = []
    recalls = {}
    for th in thresholds:
        recalls[th] = []

    for batch_index, iou in enumerate(per_image_ious):
        best_iou = tf.reduce_max(iou, axis=0)
        best_ious.append(best_iou)

        mean_best_ious.append(tf.reduce_mean(best_iou))

        # summaries = [mean_best_iou]
        with tf.device('/cpu:0'):
            for th in thresholds:
                recall = tf.truediv(
                    tf.count_nonzero(best_iou >= th),
                    tf.size(best_iou, out_type=tf.int64))
                recalls[th].append(recall)

    all_mean_best_ious = tf.stack(mean_best_ious)
    mean_of_mean_best_iou = tf.reduce_mean(all_mean_best_ious, name='best_iou_per_gt')
    summaries = [mean_of_mean_best_iou]
    for th in thresholds:
        recall = tf.reduce_mean(tf.stack(recalls[th]), name='recall_iou{}'.format(th))
        summaries.append(recall)

    add_moving_summary(*summaries)

@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some boxes from all proposals for training.
    #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        A BoxProposals instance.
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)
    # stop the gradient -- they are meant to be training targets
    return BoxProposals(
        tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'),
        tf.stop_gradient(ret_labels, name='sampled_labels'),
        tf.stop_gradient(fg_inds_wrt_gt))




@under_name_scope(name_scope="sample_fast_rcnn_targets")
def sample_fast_rcnn_targets_batch(boxes, gt_boxes, gt_labels, orig_gt_counts, batch_size):
    """
    Sample some boxes from all proposals for training.
    #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals.
    Args:
        boxes: (#lvl x BS x K) x 5 region proposals. [batch_index, floatbox] aka Nx5
        gt_boxes: BS x MaxGT x 4, floatbox
        gt_labels: BS x MaxGT, int32
        orig_gt_counts: BS   # The number of ground truths in the data. Use to unpad gt_labels and gt_boxes
    Returns:
        sampled_boxes: tx5 floatbox, the rois
        sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    prefix = "sample_fast_rcnn_targets_batch"

    # Examine proposal boxes generated by RPN

    # boxes = print_runtime_tensor("rois", boxes)
    boxes = tf.identity(boxes, name="dump_rois")
    gt_boxes = tf.identity(gt_boxes, name="dump_gt_boxes")
    gt_labels = tf.identity(gt_labels, name="dump_gt_labels")



    per_image_ious = pairwise_iou_batch(boxes, gt_boxes, orig_gt_counts, batch_size=batch_size) # list of len BS [N x M]

    per_image_ious[0] = tf.identity(per_image_ious[0], name="dump_per_image_ious")

    # per_image_ious[0] = print_runtime_tensor("per_image_ious (batch_idx=0):", per_image_ious[0], prefix=prefix)
    proposal_metrics_batch(per_image_ious)


    ious = []
    best_iou_inds = []
    for i in range(batch_size):
        image_ious = per_image_ious[i]
        gt_count = orig_gt_counts[i]

        # gt_count = print_runtime_tensor("gt_count", gt_count)

        single_image_gt_boxes = gt_boxes[i, :gt_count, :]

        single_image_gt_boxes = tf.identity(single_image_gt_boxes, name=f'dump_single_image_gt_boxes-{i}')

        # single_image_gt_boxes = print_runtime_tensor("single_image_gt_boxes", single_image_gt_boxes)
        single_image_gt_boxes = tf.pad(single_image_gt_boxes, [[0,0], [1,0]], mode="CONSTANT", constant_values=i)
        boxes = tf.concat([boxes, single_image_gt_boxes], axis=0)

        iou = tf.concat([image_ious, tf.eye(gt_count)], axis=0)  # (N+M) x M

        iou = tf.identity(iou, name=f'dump_iou-{i}')

        best_iou_ind = tf.argmax(iou, axis=1)   # A vector with the index of the GT with the highest IOU,
                                                # (length #proposals (N+M), values all in 0~m-1)

        best_iou_ind = tf.identity(best_iou_ind, name=f'dump_best_iou_ind-{i}')
        ious.append(iou)
        best_iou_inds.append(best_iou_ind)

    def sample_fg_bg(iou):
        """
        Sample rows from the iou so that:
            - you have the correct ratio of fg/bg,
            - The total number of sampled rows matches FRCNN.BATCH_PER_IM (unless there are insufficient rows)
        FG/BG is determined based on whether the proposal has an IOU with a GT that crosses the FG_THRESH
        Args:
            iou: (N+M) x M
        Returns:
            fg_inds: List of rows indices (0:N+M) from iou that are fg and have iou at least FRCNN.FG_THRESH
            bg_inds: List of rows indices (0:N+M) from iou that are bg
        """
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH # N+M vector

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds))
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]
        # fg_inds = fg_inds[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds))
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]
        # bg_inds = bg_inds[:num_bg]


        return num_fg, num_bg, fg_inds, bg_inds





    all_ret_boxes = []
    all_ret_labels = []
    all_fg_inds_wrt_gt = []
    num_bgs = []
    num_fgs = []
    for i in range(batch_size):
        # ious[i] = print_runtime_tensor("ious[i]", ious[i], prefix=prefix)
        num_fg, num_bg, fg_inds, bg_inds = sample_fg_bg(ious[i])

        num_fg = tf.identity(num_fg, name=f'dump_num_fg-{i}')
        num_bg = tf.identity(num_bg, name=f'dump_num_bg-{i}')

        num_fgs.append(num_fg)
        num_bgs.append(num_bg)

        # fg_inds = print_runtime_tensor("fg_inds", fg_inds, prefix=prefix)
        # bg_inds = print_runtime_tensor("bg_inds", bg_inds, prefix=prefix)



        best_iou_ind = best_iou_inds[i]


        fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # num_fg

        fg_inds_wrt_gt = tf.identity(fg_inds_wrt_gt, name=f'dump_fg_inds_wrt_gt-{i}')
        all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes

        all_indices = tf.identity(all_indices, name=f'dump_all_indices-{i}')

        box_mask_for_image = tf.equal(boxes[:, 0], i) # Extract boxes for a single image so we can apply all_indices as mask

        # print_buildtime_shape(f'boxes, btch_idx={i}', boxes, prefix=prefix)
        # print_buildtime_shape(f'box_mask_for_image, btch_idx={i}', box_mask_for_image, prefix=prefix)
        single_images_row_indices = tf.squeeze(tf.where(box_mask_for_image), axis=1)
        # print_buildtime_shape(f'single_images_row_indices, btch_idx={i}', single_images_row_indices, prefix=prefix)
        single_image_boxes = tf.gather(boxes, single_images_row_indices) # ?x5
        # print_buildtime_shape(f'single_image_boxes, btch_idx={i}', single_image_boxes, prefix=prefix)
        # single_image_boxes = single_image_boxes[:, 1:] # ?x4
        # print_buildtime_shape(f'single_image_boxes, batch_idx column removed, btch_idx={i}', single_image_boxes, prefix=prefix)

        single_image_ret_boxes = tf.gather(single_image_boxes, all_indices)  # ?x5

        # print_buildtime_shape(f'single_image_ret_boxes (unpadded), btch_idx={i}', single_image_ret_boxes, prefix=prefix)
        # single_image_ret_boxes = tf.pad(single_image_ret_boxes, [[0, 0], [1, 0]], constant_values=i) # ?x5

        # print_buildtime_shape(f'single_image_ret_boxes, btch_idx={i}', single_image_ret_boxes, prefix=prefix)

        all_ret_boxes.append(single_image_ret_boxes)

        gt_count = orig_gt_counts[i]
        single_image_gt_labels = gt_labels[i, 0:gt_count]   # Vector of length #gts

        single_image_ret_labels = tf.concat(
            [tf.gather(single_image_gt_labels, fg_inds_wrt_gt),
             tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)

        all_ret_labels.append(single_image_ret_labels)
        all_fg_inds_wrt_gt.append(fg_inds_wrt_gt)

    total_num_fgs = tf.add_n(num_fgs, name="num_fg")
    total_num_bgs = tf.add_n(num_bgs, name="num_bg")
    add_moving_summary(total_num_fgs, total_num_bgs)

    # total_num_fgs = print_runtime_tensor("total_num_fgs", total_num_fgs)

    ret_boxes = tf.concat(all_ret_boxes, axis=0)    # ? x 5
    ret_labels = tf.concat(all_ret_labels, axis=0)  # ? vector



    # stop the gradient -- they are meant to be training targets
    sampled_boxes = tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes')
    box_labels = tf.stop_gradient(ret_labels, name='sampled_labels')
    gt_id_for_each_fg = [tf.stop_gradient(fg_inds_wrt_gt) for fg_inds_wrt_gt in all_fg_inds_wrt_gt]

    return sampled_boxes, box_labels, gt_id_for_each_fg



@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4

    Returns:
        cls_logits: N x num_class classification logits
        reg_logits: N x num_classx4 or Nx2x4 if class agnostic
    """
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4), name='output_box')

    return classification, box_regression




@layer_register(log_shape=True)
def fastrcnn_outputs_batch(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4

    Returns:
        cls_logits: N x num_class classification logits
        reg_logits: N x num_classx4 or Nx2x4 if class agnostic
    """
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, [-1, num_classes_for_box, 4], name='output_box')
    return classification, box_regression



@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgxCx4 or nfgx1x4 if class agnostic

    Returns:
        label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(num_fg), fg_labels], axis=1)  # #fgx2
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)
    else:
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.cast(tf.equal(prediction, labels), tf.float32)  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.cast(tf.shape(labels)[0], tf.float32), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy,
                       fg_accuracy, false_negative, tf.cast(num_fg, tf.float32, name='num_fg_label'))
    return [label_loss, box_loss]


@under_name_scope()
def fastrcnn_predictions(boxes, scores):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#classx4 floatbox in float32
        scores: nx#class

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert scores.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob, out_type=tf.int64)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        #selection = non_max_suppression_custom(
            #box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.gather(ids, selection)

        if get_tf_version_tuple() >= (1, 13):
            sorted_selection = tf.sort(selection, direction='ASCENDING')
            mask = tf.sparse.SparseTensor(indices=tf.expand_dims(sorted_selection, 1),
                                          values=tf.ones_like(sorted_selection, dtype=tf.bool),
                                          dense_shape=output_shape)
            mask = tf.sparse.to_dense(mask, default_value=False)
        else:
            # this function is deprecated by TF
            sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
            mask = tf.sparse_to_dense(
                sparse_indices=sorted_selection,
                output_shape=output_shape,
                sparse_values=True,
                default_value=False)
        return mask

    # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
    buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]
    masks = tf.map_fn(f, (scores, boxes), dtype=tf.bool,
                      parallel_iterations=1 if buggy_tf else 10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    scores = tf.boolean_mask(scores, masks)

    # filter again by sorting scores
    topk_scores, topk_indices = tf.nn.top_k(
        scores,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(scores)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)

    final_scores = tf.identity(topk_scores, name='scores')
    final_labels = tf.add(cat_ids, 1, name='labels')
    final_ids = tf.stack([cat_ids, box_ids], axis=1, name='all_ids')
    final_boxes = tf.gather_nd(boxes, final_ids, name='boxes')
    return final_boxes, final_scores, final_labels


"""
FastRCNN heads for FPN:
"""


@layer_register(log_shape=True)
def fastrcnn_2fc_head(feature, fp16=False):
    """
    Args:
        feature (any shape):

    Returns:
        2D head feature
    """
    dim = cfg.FPN.FRCNN_FC_HEAD_DIM
    if fp16:
        feature = tf.cast(feature, tf.float16)

    with mixed_precision_scope(mixed=fp16):
        init = tf.variance_scaling_initializer(dtype=tf.float16 if fp16 else tf.float32)
        hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
        hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)

    if fp16:
        hidden = tf.cast(hidden, tf.float32)

    return hidden


@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_convs, norm=None):
    """
    Args:
        feature (NCHW):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'

    Returns:
        2D head feature
    """
    assert norm in [None, 'GN'], norm
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal' if get_tf_version_tuple() >= (1, 12) else 'normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.FRCNN_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, cfg.FPN.FRCNN_FC_HEAD_DIM,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
    return l


def fastrcnn_4conv1fc_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


def fastrcnn_4conv1fc_gn_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, norm='GN', **kwargs)


class BoxProposals(object):
    """
    A structure to manage box proposals and their relations with ground truth.
    """
    def __init__(self, boxes, labels=None, fg_inds_wrt_gt=None):
        """
        Args:
            boxes: Nx4
            labels: N, each in [0, #class), the true label for each input box
            fg_inds_wrt_gt: #fg, each in [0, M)

        The last four arguments could be None when not training.
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)

    @memoized_method
    def fg_inds(self):
        """ Returns: #fg indices in [0, N-1] """
        return tf.reshape(tf.where(self.labels > 0), [-1], name='fg_inds')

    @memoized_method
    def fg_boxes(self):
        """ Returns: #fg x4"""
        return tf.gather(self.boxes, self.fg_inds(), name='fg_boxes')

    @memoized_method
    def fg_labels(self):
        """ Returns: #fg"""
        return tf.gather(self.labels, self.fg_inds(), name='fg_labels')






class FastRCNNHeadBatch(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """
    def __init__(self,
                 box_logits,
                 label_logits,
                 bbox_regression_weights,
                 proposal_batch_idx_map,
                 prepadding_gt_counts,
                 proposal_boxes):
        """
        Args:
            proposals: BoxProposalsBatch (boxes=Nx5)
            box_logits: Nx#classx4 or Nx1x4, the output of the head
            label_logits: Nx#class, the output of the head
            gt_boxes: BS x MaxGTs x 4
            bbox_regression_weights: a 4 element tensor
            proposal_batch_idx_map: N element vector with batch index from that BoxProposal.box
        """
        self.box_logits = box_logits
        self.label_logits = label_logits

        self.bbox_regression_weights = bbox_regression_weights
        self.proposal_batch_idx_map = proposal_batch_idx_map
        self.prepadding_gt_counts = prepadding_gt_counts

        self.proposal_boxes = proposal_boxes

        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1

        self.training_info_available = False


    def add_training_info(self,
                          gt_boxes,
                          proposal_labels,
                          proposal_fg_inds,
                          proposal_fg_boxes,
                          proposal_fg_labels,
                          proposal_gt_id_for_each_fg):

        self.gt_boxes = gt_boxes
        self.proposal_labels = proposal_labels
        self.proposal_fg_inds = proposal_fg_inds
        self.proposal_fg_boxes = proposal_fg_boxes
        self.proposal_fg_labels = proposal_fg_labels
        self.proposal_gt_id_for_each_fg = proposal_gt_id_for_each_fg

        self.training_info_available = True



    @memoized_method
    def losses(self, batch_size_per_gpu, shortcut=False):

        assert self.training_info_available, "In order to calculate losses, we need to know GT info, but " \
                                             "add_training_info was never called"

        if shortcut:
            proposal_label_loss = tf.cast(tf.reduce_mean(self.proposal_labels), dtype=tf.float32)
            proposal_boxes_loss = tf.cast(tf.reduce_mean(self.proposal_boxes), dtype=tf.float32)
            proposal_fg_boxes_loss = tf.cast(tf.reduce_mean(self.proposal_fg_boxes), dtype=tf.float32)
            gt_box_loss = tf.cast(tf.reduce_mean(self.gt_boxes), dtype=tf.float32)

            bbox_reg_loss = tf.cast(tf.reduce_mean(self.bbox_regression_weights), dtype=tf.float32)
            label_logit_loss = tf.cast(tf.reduce_mean(self.label_logits), dtype=tf.float32)

            total_loss = proposal_label_loss + proposal_boxes_loss + proposal_fg_boxes_loss + gt_box_loss \
                         + bbox_reg_loss + label_logit_loss
            return [total_loss]

        all_labels = []
        all_label_logits = []
        all_encoded_fg_gt_boxes = []
        all_fg_box_logits = []
        for i in range(batch_size_per_gpu):

            single_image_fg_inds_wrt_gt = self.proposal_gt_id_for_each_fg[i]

            single_image_gt_boxes = self.gt_boxes[i, :self.prepadding_gt_counts[i], :] # NumGT x 4
            gt_for_each_fg = tf.gather(single_image_gt_boxes, single_image_fg_inds_wrt_gt) # NumFG x 4
            single_image_fg_boxes_indices = tf.where(tf.equal(self.proposal_fg_boxes[:, 0], i))
            single_image_fg_boxes_indices = tf.squeeze(single_image_fg_boxes_indices, axis=1)

            single_image_fg_boxes = tf.gather(self.proposal_fg_boxes, single_image_fg_boxes_indices) # NumFG x 5
            single_image_fg_boxes = single_image_fg_boxes[:, 1:]  # NumFG x 4

            encoded_fg_gt_boxes = encode_bbox_target(gt_for_each_fg, single_image_fg_boxes) * self.bbox_regression_weights

            single_image_box_indices = tf.squeeze(tf.where(tf.equal(self.proposal_boxes[:, 0], i)), axis=1)
            single_image_labels = tf.gather(self.proposal_labels, single_image_box_indices) # Vector len N
            single_image_label_logits = tf.gather(self.label_logits, single_image_box_indices)

            single_image_box_logits = tf.gather(self.box_logits, single_image_box_indices)

            single_image_fg_box_logits = tf.gather(single_image_box_logits, single_image_fg_boxes_indices)

            all_labels.append(single_image_labels)
            all_label_logits.append(single_image_label_logits)
            all_encoded_fg_gt_boxes.append(encoded_fg_gt_boxes)
            all_fg_box_logits.append(single_image_fg_box_logits)



        return fastrcnn_losses(
            tf.concat(all_labels, axis=0),
            tf.concat(all_label_logits, axis=0),
            tf.concat(all_encoded_fg_gt_boxes, axis=0),
            tf.concat(all_fg_box_logits, axis=0)
        )


    # ------ NOTHING HERE HAS BEEN BATCHIFIED CAREFULLY. --------
    @memoized_method
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        nobatch_proposal_boxes = self.proposal_boxes[:, 1:]
        anchors = tf.tile(tf.expand_dims(nobatch_proposal_boxes, 1),
                          [1, cfg.DATA.NUM_CLASS, 1])  # N x #class x 4
        decoded_boxes = decode_bbox_target(
                self.box_logits / self.bbox_regression_weights,
                anchors
        )
        return decoded_boxes


    @memoized_method
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    # ------ -------------------------------------------- --------



class FastRCNNHead(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """
    def __init__(self, proposals, box_logits, label_logits, gt_boxes, bbox_regression_weights):
        """
        Args:
            proposals: BoxProposals
            box_logits: Nx#classx4 or Nx1x4, the output of the head
            label_logits: Nx#class, the output of the head
            gt_boxes: Mx4
            bbox_regression_weights: a 4 element tensor
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)
        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1

    @memoized_method
    def fg_box_logits(self):
        """ Returns: #fg x ? x 4 """
        return tf.gather(self.box_logits, self.proposals.fg_inds(), name='fg_box_logits')

    @memoized_method
    def losses(self):
        encoded_fg_gt_boxes = encode_bbox_target(
            tf.gather(self.gt_boxes, self.proposals.fg_inds_wrt_gt),
            self.proposals.fg_boxes()) * self.bbox_regression_weights
        return fastrcnn_losses(
            self.proposals.labels, self.label_logits,
            encoded_fg_gt_boxes, self.fg_box_logits()
        )

    @memoized_method
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposals.boxes, 1),
                          [1, cfg.DATA.NUM_CLASS, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights,
            anchors
        )
        return decoded_boxes

    @memoized_method
    def decoded_output_boxes_for_true_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.proposals.labels)

    @memoized_method
    def decoded_output_boxes_for_predicted_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.predicted_labels())

    @memoized_method
    def decoded_output_boxes_for_label(self, labels):
        assert not self._bbox_class_agnostic
        indices = tf.stack([
            tf.range(tf.size(labels, out_type=tf.int64)),
            labels
        ])
        needed_logits = tf.gather_nd(self.box_logits, indices)
        decoded = decode_bbox_target(
            needed_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    @memoized_method
    def decoded_output_boxes_class_agnostic(self):
        """ Returns: Nx4 """
        assert self._bbox_class_agnostic
        box_logits = tf.reshape(self.box_logits, [-1, 4])
        decoded = decode_bbox_target(
            box_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    @memoized_method
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    @memoized_method
    def predicted_labels(self):
        """ Returns: N ints """
        return tf.argmax(self.label_logits, axis=1, name='predicted_labels')