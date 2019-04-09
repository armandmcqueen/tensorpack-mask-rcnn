#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import itertools
import numpy as np
import os
import shutil
import cv2
import six
assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack import *
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.summary import add_moving_summary

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
    from .dataset import DetectionDataset
    from .config import finalize_configs, config as cfg
    from .data import get_all_anchors, get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow, get_batch_train_dataflow
    from .eval import DetectionResult, predict_image, multithread_predict_dataflow, EvalCallback
    from .model_box import RPNAnchors, clip_boxes, clip_boxes_batch, crop_and_resize, roi_align, crop_and_resize_from_batch_codebase
    from .model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses, \
        generate_fpn_proposals_batch_tf_op, multilevel_roi_align_tf_op, multilevel_rpn_losses_batch, multilevel_rpn_losses_batch_fixed_single_image
    from .model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions_batch, fastrcnn_predictions, sample_fast_rcnn_targets, \
        sample_fast_rcnn_targets_batch, fastrcnn_outputs_batch, FastRCNNHeadBatch
    from .model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head
    from .model_rpn import generate_rpn_proposals, rpn_head, rpn_losses, rpn_head_withbatch
    from .viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall
    from .performance import ThroughputTracker, print_runtime_shape, print_runtime_tensor, print_runtime_tensor_loose_branch, summarize_tensor
else:

    import model_frcnn
    import model_mrcnn
    from basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
    from dataset import DetectionDataset
    from config import finalize_configs, config as cfg
    from data import get_all_anchors, get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow, get_batch_train_dataflow
    from eval import DetectionResult, predict_image, multithread_predict_dataflow, EvalCallback
    from model_box import RPNAnchors, clip_boxes, clip_boxes_batch, crop_and_resize, roi_align, crop_and_resize_from_batch_codebase
    from model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses, \
        generate_fpn_proposals_batch_tf_op, multilevel_roi_align_tf_op, multilevel_rpn_losses_batch, multilevel_rpn_losses_batch_fixed_single_image
    from model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions_batch, fastrcnn_predictions, sample_fast_rcnn_targets, \
        sample_fast_rcnn_targets_batch, fastrcnn_outputs_batch, FastRCNNHeadBatch
    from model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head
    from model_rpn import generate_rpn_proposals, rpn_head, rpn_losses, rpn_head_withbatch
    from viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall
    from performance import ThroughputTracker, print_runtime_shape, print_runtime_tensor, print_runtime_tensor_loose_branch, summarize_tensor

BATCH_SIZE_PLACEHOLDER = 1 # Some pieces of batch code rely on batch size global arg. In convergence codebase, this is a constant


BATCH_ENABLE = True 


try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

def runtime_print(message, trigger_tensor):
    print_op = tf.print(message)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)

def print_runtime_tensor_loose_branch(name, tensor, prefix=None, summarize=-1, trigger_tensor=None):
    assert trigger_tensor is not None

    s = "[runtime_tensor_freehanging_branch] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)



def print_runtime_shape(name, tensor, prefix=None):
    s = "[runtime_shape] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += f'{name}: '
    return runtime_print([s, tf.shape(tensor)], tensor)



def print_runtime_tensor(name, tensor, prefix=None, summarize=-1):
    s = "[runtime_tensor] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)



class DetectionModel(ModelDesc):
    def __init__(self, fp16):
        self.fp16 = fp16

    def preprocess(self, image):
    #    image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    @property
    def training(self):
        return get_current_tower_context().is_training

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """

        if BATCH_ENABLE:
            out = ['output/batch_indices', 'output/boxes', 'output/scores', 'output/labels']
        else:
            out = ['output/boxes', 'output/scores', 'output/labels']

        if cfg.MODE_MASK:
            out.append('output/masks')
        return ['images', 'orig_image_dims'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        #########################################################################################################
        #if BATCH_DATA_PIPELINE_EVAL:
        #########################################################################################################
            #inputs["anchor_labels_lvl2"] = tf.squeeze(inputs["anchor_labels_lvl2"], axis=0)
            #inputs["anchor_boxes_lvl2"] = tf.squeeze(inputs["anchor_boxes_lvl2"], axis=0)
            #inputs["anchor_labels_lvl3"] = tf.squeeze(inputs["anchor_labels_lvl3"], axis=0)
            #inputs["anchor_boxes_lvl3"] = tf.squeeze(inputs["anchor_boxes_lvl3"], axis=0)
            #inputs["anchor_labels_lvl4"] = tf.squeeze(inputs["anchor_labels_lvl4"], axis=0)
            #inputs["anchor_boxes_lvl4"] = tf.squeeze(inputs["anchor_boxes_lvl4"], axis=0)
            #inputs["anchor_labels_lvl5"] = tf.squeeze(inputs["anchor_labels_lvl5"], axis=0)
            #inputs["anchor_boxes_lvl5"] = tf.squeeze(inputs["anchor_boxes_lvl5"], axis=0)
            #inputs["anchor_labels_lvl6"] = tf.squeeze(inputs["anchor_labels_lvl6"], axis=0)
            #inputs["anchor_boxes_lvl6"] = tf.squeeze(inputs["anchor_boxes_lvl6"], axis=0)
            #inputs["images"] = tf.squeeze(inputs["images"], axis=0)
            #inputs["orig_image_dims"] = tf.squeeze(inputs["orig_image_dims"], axis=0)
            #inputs["orig_gt_counts"] = tf.squeeze(inputs["orig_gt_counts"], axis=0)
            #inputs["gt_labels"] = tf.squeeze(inputs["gt_labels"], axis=0)
            #inputs["gt_boxes"] = tf.squeeze(inputs["gt_boxes"], axis=0)
            #inputs["gt_masks"] = tf.squeeze(inputs["gt_masks"], axis=0)
        #########################################################################################################

        image = self.preprocess(inputs['images'])     # 1CHW

        features = self.backbone(image)
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        proposal_boxes, rpn_losses = self.rpn(image, features, anchor_inputs, inputs['orig_image_dims'])  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(image, features, proposal_boxes, targets, inputs)

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost



class ResNetFPNModel(DetectionModel):
    def __init__(self, fp16):
        super(ResNetFPNModel, self).__init__(fp16)

    # TODO: Batchify
    def inputs(self):

        ######################################################################################################
        if BATCH_ENABLE:
        ######################################################################################################

            ret = [
                tf.placeholder(tf.string, (None,), 'filenames'), # N length vector of filenames
                tf.placeholder(tf.float32, (None, None, None, 3), 'images'),  # N x H x W x C
                tf.placeholder(tf.int32, (None, 3), 'orig_image_dims')  # N x 3(image dims - hwc)
            ]
            num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
            for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
                ret.extend([
                    tf.placeholder(tf.int32, (None, None, None, num_anchors),  # N x H x W x NumAnchors
                                'anchor_labels_lvl{}'.format(k + 2)),
                    tf.placeholder(tf.float32, (None, None, None, num_anchors, 4),  # N x H x W x NumAnchors x 4
                                'anchor_boxes_lvl{}'.format(k + 2))])
            ret.extend([
                tf.placeholder(tf.float32, (None, None, 4), 'gt_boxes'),  # N x MaxNumGTs x 4
                tf.placeholder(tf.int64, (None, None), 'gt_labels'),  # all > 0        # N x MaxNumGTs
                tf.placeholder(tf.int32, (None,), 'orig_gt_counts')  # N
            ])

            if cfg.MODE_MASK:
                ret.append(
                        tf.placeholder(tf.uint8, (None, None, None, None), 'gt_masks')  # N x MaxNumGTs x H x W
                )
        else:
            ret = [ tf.placeholder(tf.float32, (None, None, 3), 'images') ]
            num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
            for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
                ret.extend([
                        tf.placeholder(tf.int32, (None, None, num_anchors),
                                    'anchor_labels_lvl{}'.format(k + 2)),
                        tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                                    'anchor_boxes_lvl{}'.format(k + 2))])
            ret.extend([
                tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
                tf.placeholder(tf.int64, (None,), 'gt_labels')])  # all > 0
            if cfg.MODE_MASK:
                ret.append(
                    tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
                )   # NR_GT x height x width
        #######################################################################################################

        return ret

    def slice_feature_and_anchors(self, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_slice_lvl{}'.format(i)):
                anchors[i] = anchors[i].narrow_to(p23456[i])

    def backbone(self, image):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS, fp16=self.fp16)
        print("c2345", c2345)
        p23456 = fpn_model('fpn', c2345, fp16=self.fp16)
        return p23456


    def rpn(self, image, features, inputs, orig_image_dims):
        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

        if BATCH_ENABLE:
            image_shape2d = orig_image_dims[:,:2]
            #image_shape2d = print_runtime_tensor('image_shape2d', image_shape2d)
        else:
            image_shape2d = tf.shape(image)[2:] 
    #    image_shape2d = tf.expand_dims(tf.shape(image)[2:], 0)     # h,w

        all_anchors_fpn = get_all_anchors_fpn()

        #multilevel_anchors = [RPNAnchors(
        #    all_anchors_fpn[i],
        #    tf.squeeze(inputs['anchor_labels_lvl{}'.format(i + 2)], axis=0),
        #    tf.squeeze(inputs['anchor_boxes_lvl{}'.format(i + 2)], axis=0)) for i in range(len(all_anchors_fpn))]
        #
        #self.slice_feature_and_anchors(features, multilevel_anchors)


        # Multi-Level RPN Proposals
        #########################################################################################################
        if BATCH_ENABLE:
        #########################################################################################################
            rpn_outputs = []
            for pi in features:
                label_logits, box_logits = rpn_head_withbatch('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS), fp16=self.fp16)

                # Operation the same transpose as in the no batch code
                #label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA
                #shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
                #box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
                #box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], -1, 4]))  # fHxfWxNAx4

                rpn_outputs.append((label_logits, box_logits))

            multilevel_label_logits = [k[0] for k in rpn_outputs]
            multilevel_box_logits = [k[1] for k in rpn_outputs]

#            image_shape2d = tf.expand_dims(image_shape2d, 0)
            proposal_boxes, proposal_scores = generate_fpn_proposals_batch_tf_op(all_anchors_fpn, multilevel_box_logits,
                                                                                 multilevel_label_logits,
                                                                                 image_shape2d)

            ############# UNBATCHING FOR RPN LOSS ##########################
            #multilevel_label_logits_unbatched = [tf.squeeze(k, 0) for k in multilevel_label_logits]

            #multilevel_box_logits_out = []
            #for box_logits in multilevel_box_logits:
            #    shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
            #    box_logits_out = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
            #    box_logits_out = tf.reshape(box_logits_out, tf.stack([shp[2], shp[3], -1, 4]))  # fHxfWxNAx4
            #    multilevel_box_logits_out.append(box_logits_out)
            #multilevel_box_logits_unbatched  = multilevel_box_logits_out     
            ############################################################### 
 
        else:
            rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS), fp16=self.fp16)
                           for pi in features]
            multilevel_label_logits = [k[0] for k in rpn_outputs]
            multilevel_box_logits = [k[1] for k in rpn_outputs]

            multilevel_pred_boxes = [anchor.decode_logits(logits)
                                     for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]
            proposal_boxes, proposal_scores = generate_fpn_proposals(
                multilevel_pred_boxes, multilevel_label_logits, image_shape2d)

           
        #########################################################################################################




        if self.training:
            #########################################################################################################
            if BATCH_ENABLE:
            #########################################################################################################

                # TODO: Remove this when merging batch blocks, begin
                #batch_input_multilevel_anchor_labels = [tf.expand_dims(inputs['anchor_labels_lvl{}'.format(i + 2)], axis=0) for i in range(len(all_anchors_fpn))]
                #batch_input_multilevel_anchor_boxes = [tf.expand_dims(inputs['anchor_boxes_lvl{}'.format(i + 2)], axis=0) for i in range(len(all_anchors_fpn))]
                #batch_input_multilevel_label_logits = [tf.expand_dims(m, axis=0) for m in multilevel_label_logits]
                #batch_input_multilevel_box_logits = [tf.expand_dims(m, axis=0) for m in multilevel_box_logits]
                # TODO: Remove this when merging batch blocks, end

                batch_input_multilevel_anchor_labels = [inputs['anchor_labels_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]
                batch_input_multilevel_anchor_boxes = [inputs['anchor_boxes_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]
                batch_input_multilevel_label_logits = multilevel_label_logits 

                batch_input_multilevel_box_logits = []
                for box_logits in multilevel_box_logits:
                    shp = tf.shape(box_logits)  # Nx(NAx4)xfHxfW
                    box_logits_t = tf.transpose(box_logits, [0, 2, 3, 1])  # NxfHxfWx(NAx4)
                    box_logits_t = tf.reshape(box_logits_t, tf.stack([shp[0], shp[2], shp[3], -1, 4]))  # NxfHxfWxNAx4
                    batch_input_multilevel_box_logits.append(box_logits_t)

                rpn_losses = []
                for i in range(BATCH_SIZE_PLACEHOLDER):
                    orig_image_hw = orig_image_dims[i, :2]
                    si_all_anchors_fpn = get_all_anchors_fpn()
                    si_multilevel_box_logits = [box_logits[i, :, :, :, :] for box_logits in batch_input_multilevel_box_logits]
                    si_multilevel_label_logits = [label_logits[i, :, :, :] for label_logits in batch_input_multilevel_label_logits]
                    si_multilevel_anchor_labels = [anchor_labels[i, :, :, :] for anchor_labels in batch_input_multilevel_anchor_labels]
                    si_multilevel_anchors_boxes = [anchor_boxes[i, :, :, :, :] for anchor_boxes in batch_input_multilevel_anchor_boxes]

                    si_multilevel_anchors = [RPNAnchors(si_all_anchors_fpn[j],
                                                        si_multilevel_anchor_labels[j],
                                                        si_multilevel_anchors_boxes[j])
                                                        for j in range(len(features))]

                    # Given the original image dims, find what size each layer of the FPN feature map would be (follow FPN padding logic)
                    mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)  # the image is padded so that it is a multiple of this (32 with default config).
                    orig_image_hw_after_fpn_padding = tf.ceil(tf.cast(orig_image_hw, tf.float32) / mult) * mult
                    si_multilevel_anchors_narrowed = []
                    for lvl, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
                        featuremap_dims_float = orig_image_hw_after_fpn_padding / float(stride)
                        featuremap_dims = tf.cast(tf.math.floor(featuremap_dims_float + 0.5), tf.int32)  # Fix bankers rounding

                        narrowed_anchors_for_lvl = si_multilevel_anchors[lvl].narrow_to_featuremap_dims(featuremap_dims)
                        si_multilevel_anchors_narrowed.append(narrowed_anchors_for_lvl)

                    si_losses = multilevel_rpn_losses_batch_fixed_single_image(si_multilevel_anchors_narrowed,
                                                                               si_multilevel_label_logits,
                                                                               si_multilevel_box_logits)
                    rpn_losses.extend(si_losses)

                with tf.name_scope('rpn_losses'):
                    total_label_loss = tf.add_n(rpn_losses[::2], name='label_loss')
                    total_box_loss = tf.add_n(rpn_losses[1::2], name='box_loss')
                    add_moving_summary(total_label_loss, total_box_loss)
                    losses = [total_label_loss, total_box_loss]

                # losses[0] = print_runtime_tensor("batch_total_label_loss", losses[0])
                # losses[1] = print_runtime_tensor("batch_total_box_loss", losses[1])
            else:
                no_batch_losses = multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
                losses = no_batch_losses

            #########################################################################################################
        else:
            losses = []

        return proposal_boxes, losses

    def roi_heads(self, image, features, proposal_boxes, targets, inputs):

        if BATCH_ENABLE:
            image_shape2d = inputs['orig_image_dims'][:,:2]      # BSx2 (h&w)
            #image_shape2d = print_runtime_tensor('image_shape2d', image_shape2d)
        else:
            image_shape2d = tf.shape(image)[2:]     # h,w
    #    image_shape2d = tf.expand_dims(tf.shape(image)[2:], 0)     # h,w   

        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        #############################################################################################################
        if BATCH_ENABLE:
        #############################################################################################################
#            prepadding_gt_counts = tf.expand_dims(tf.shape(gt_labels)[0], axis=0)  # 1 x NumGT
            prepadding_gt_counts = inputs['orig_gt_counts'] 

            if self.training:
                # BATCH_SAMPLE_TARGETS
                #input_gt_boxes = tf.expand_dims(gt_boxes, axis=0)
                #input_gt_labels = tf.expand_dims(gt_labels, axis=0)
                #input_proposal_boxes = tf.pad(proposal_boxes, [[0,0], [1,0]], constant_values=0)
                input_proposal_boxes = proposal_boxes
                input_gt_boxes = gt_boxes
                input_gt_labels = gt_labels

                proposal_boxes, proposal_labels, proposal_gt_id_for_each_fg = sample_fast_rcnn_targets_batch(
                        input_proposal_boxes,
                        input_gt_boxes,
                        input_gt_labels,
                        prepadding_gt_counts,
                        batch_size=BATCH_SIZE_PLACEHOLDER)

                #proposal_boxes = proposal_boxes[:, 1:]

                # TODO: Remove this after mask blocks have been modified to work with no BoxProposals
                # It's needed for mask block functionality but is not used within this code block and
                # should be able to be safely deleted once the mask blocks have been converted.
#                proposals = BoxProposals(proposal_boxes, proposal_labels, proposal_gt_id_for_each_fg[0])
                proposals = BoxProposals(proposal_boxes, proposal_labels, proposal_gt_id_for_each_fg)


            # BATCH_ROI_ALIGN_BOX
            roi_feature_fastrcnn = multilevel_roi_align_tf_op(features[:4], proposal_boxes, 7)

            # COMMON CODE
            fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
            head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn, fp16=self.fp16)

            # BATCH_FAST_RCNN_OUTPUTS
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs_batch('fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)

            # BATCH_FAST_RCNN_LOSSES
            # Convert nobatch tensors to batch tensors
            regression_weights = tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)
#            batch_indices_for_rois = tf.zeros(tf.shape(proposal_boxes)[0])
#            proposal_boxes = tf.pad(proposal_boxes, [[0,0],[1,0]])
            # END Convert nobatch tensors to batch tensors
            batch_indices_for_rois = proposal_boxes[:,0]


            fastrcnn_head = FastRCNNHeadBatch(fastrcnn_box_logits,
                                              fastrcnn_label_logits,
                                              regression_weights,
                                              batch_indices_for_rois,
                                              prepadding_gt_counts,
                                              proposal_boxes)
            if self.training:
                # Convert nobatch tensors to batch tensors
                # proposal_labels = proposals.labels
                # proposal_gt_id_for_each_fg = [proposals.fg_inds_wrt_gt]
                proposal_fg_inds = tf.reshape(tf.where(proposal_labels > 0), [-1])
                proposal_fg_boxes = tf.gather(proposal_boxes, proposal_fg_inds)
                proposal_fg_labels = tf.gather(proposal_labels, proposal_fg_inds)
                # END Convert nobatch tensors to batch tensors

                fastrcnn_head.add_training_info(input_gt_boxes,
                                                proposal_labels,
                                                proposal_fg_inds,
                                                proposal_fg_boxes,
                                                proposal_fg_labels,
                                                proposal_gt_id_for_each_fg)

                all_losses = fastrcnn_head.losses(BATCH_SIZE_PLACEHOLDER)

        else: # BATCH_BOX_CLASS_HEAD = False

            proposals = BoxProposals(proposal_boxes)

            if self.training:
                # NO BATCH_SAMPLE_TARGETS
                proposals = sample_fast_rcnn_targets(proposal_boxes, gt_boxes, gt_labels)

            # NO BATCH_ROI_ALIGN_BOX
            roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7)

            # COMMON CODE
            fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
            head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn, fp16=self.fp16)

            # NO BATCH_FAST_RCNN_OUTPUTS
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)

            # NO BATCH_FAST_RCNN_LOSSES
            fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                    gt_boxes, tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
            if self.training:
                all_losses = fastrcnn_head.losses()
        ##############################################################################################################


        if self.training:
            if cfg.MODE_MASK:
                gt_masks = targets[2]
                # maskrcnn loss

                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)


                ##########################################################################################################
                if BATCH_ENABLE:
                ##########################################################################################################

                    roi_feature_maskrcnn = multilevel_roi_align_tf_op(
                        features[:4], proposals.fg_boxes(), 14,
                        name_scope='multilevel_roi_align_mask')

                    mask_logits = maskrcnn_head_func(
                            'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, fp16=self.fp16)   # #fg x #cat x 28 x 28

#                    prepadding_gt_counts = tf.expand_dims(tf.shape(gt_labels)[0], axis=0) # 1 x NumGT
#                    proposal_fg_boxes = tf.pad(proposals.fg_boxes(), [[0, 0], [1,0]], constant_values=0) # NumFG x 5
                    proposal_fg_labels = proposals.fg_labels() # vector of length NumFG
#                    proposal_gt_id_for_each_fg = [proposals.fg_inds_wrt_gt] # list [ vectors length NumFG ]
#                    orig_image_dims = [image_shape2d]       # list [ 2 ]
#                    gt_masks = tf.expand_dims(gt_masks, axis=0) # 1 x NumGT x H x W


                    prepadding_gt_counts = inputs['orig_gt_counts']
                    orig_image_dims = image_shape2d #inputs['orig_image_dims'][:,:2]      # BSx2 tensor

                    proposal_fg_boxes = proposals.fg_boxes()
                    proposal_gt_id_for_each_fg = proposals.fg_inds_wrt_gt

                    per_image_target_masks_for_fg = []
                    per_image_fg_labels = []
                    for i in range(BATCH_SIZE_PLACEHOLDER):

                        single_image_gt_count = prepadding_gt_counts[i]
                        single_image_gt_masks = gt_masks[i, :single_image_gt_count, :, :] # NumGT x H x w (maybe? might have length 1 dim at beginning)
                        single_image_fg_indices = tf.squeeze(tf.where(tf.equal(proposal_fg_boxes[:, 0], i)), axis=1)
                        single_image_fg_boxes = tf.gather(proposal_fg_boxes, single_image_fg_indices)[:, 1:]
                        single_image_fg_labels = tf.gather(proposal_fg_labels, single_image_fg_indices)
                        single_image_fg_inds_wrt_gt = proposal_gt_id_for_each_fg[i]

                        print(type(single_image_fg_inds_wrt_gt))
                        assert isinstance(single_image_fg_inds_wrt_gt, tf.Tensor)

                        single_image_gt_masks = tf.expand_dims(single_image_gt_masks, axis=1)

                        single_image_target_masks_for_fg = crop_and_resize_from_batch_codebase(single_image_gt_masks,
                                                                           single_image_fg_boxes,
                                                                           single_image_fg_inds_wrt_gt,
                                                                           28,
                                                                           orig_image_dims[i],
                                                                           pad_border=False,
                                                                           verbose_batch_index=i)  # fg x 1x28x28
                        per_image_fg_labels.append(single_image_fg_labels)
                        per_image_target_masks_for_fg.append(single_image_target_masks_for_fg)

                    target_masks_for_fg = tf.concat(per_image_target_masks_for_fg, axis=0)

                    proposal_fg_labels = tf.concat(per_image_fg_labels, axis=0)
                else:

                    roi_feature_maskrcnn = multilevel_roi_align(
                            features[:4], proposals.fg_boxes(), 14,
                            name_scope='multilevel_roi_align_mask')

                    mask_logits = maskrcnn_head_func(
                            'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, fp16=self.fp16)   # #fg x #cat x 28 x 28

                    target_masks_for_fg = crop_and_resize(
                        tf.expand_dims(gt_masks, 1),
                        proposals.fg_boxes(),
                        proposals.fg_inds_wrt_gt, 28,
                        pad_border=False)  # fg x 1x28x28

                    proposal_fg_labels = proposals.fg_labels() # Need to do this so that BATCH_MASK_LOSS can use fg_labels
                                                               # without worrying about whether BATCH_CROP_AND_RESIZE_MASK is True
                ##########################################################################################################

                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')

                mask_loss = maskrcnn_loss(mask_logits, proposal_fg_labels, target_masks_for_fg)

                all_losses.append(mask_loss)
            return all_losses
        else:

            if BATCH_ENABLE:
                decoded_boxes, batch_ids = fastrcnn_head.decoded_output_boxes_batch()
            else:
                decoded_boxes = fastrcnn_head.decoded_output_boxes()

            if BATCH_ENABLE:
                decoded_boxes = clip_boxes_batch(decoded_boxes, image_shape2d, tf.cast(batch_ids, dtype=tf.int32), name='fastrcnn_all_boxes')
            else:
                decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')

            if BATCH_ENABLE:
                final_boxes, final_scores, final_labels, box_ids = fastrcnn_predictions_batch(decoded_boxes, label_scores, name_scope='output')
            else: 
                final_boxes, final_scores, final_labels = fastrcnn_predictions(decoded_boxes, label_scores, name_scope='output')
            
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, fp16=self.fp16)   # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')
  

                if BATCH_ENABLE:
                    #proposal_boxes = tf.concat((tf.zeros([tf.shape(proposals.boxes)[0], 1], dtype=tf.float32), proposals.boxes), axis=1)      # REMOVE WHEN COMPLETELY BATCHIFIED
                    tf.gather(proposal_boxes[:,0], box_ids, name='output/batch_indices')
            return []


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['images', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['images'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_gpu = cfg.TRAIN.NUM_GPUS
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_gpu))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_gpu)
            for k in range(num_gpu)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DetectionDataset().eval_or_save_inference_results(all_results, dataset, output)


def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("output.png", viz)
    logger.info("Inference output written to output.png")
    tpviz.interactive_imshow(viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--fp16', help="Train backbone in FP16", action="store_true")

    #################################################################################################################
    # Performance investigation arguments
    parser.add_argument('--perf', help="Enable performance investigation mode", action="store_true")
    parser.add_argument('--throughput_log_freq', help="In perf investigation mode, code will print throughput after every throughput_log_freq steps as well as after every epoch", type=int, default=100)
    parser.add_argument('--images_per_step', help="Number of images in a minibatch (total, not per GPU)", type=int, default=8)
    parser.add_argument('--num_total_images', help="Number of images in an epoch. = images_per_steps * steps_per_epoch (differs slightly from the total number of images).", type=int, default=120000)

    parser.add_argument('--tfprof', help="Enable tf profiller", action="store_true")
    parser.add_argument('--tfprof_start_step', help="Step to enable tf profiling", type=int, default=15005)
    parser.add_argument('--tfprof_end_step', help="Step after which tf profiling will be disabled", type=int, default=15010)

    parser.add_argument('--summary_period', help="Write summary events periodically at this interval. Setting it to 0 writes at at the end of an epoch.", type=int, default=0)

    #################################################################################################################




    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel(args.fp16)
    DetectionDataset()  # initialize the config with information from our dataset

    if args.visualize or args.evaluate or args.predict:
        assert tf.test.is_gpu_available()
        assert args.load
        finalize_configs(is_training=False)

        if args.predict or args.visualize:
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        if args.visualize:
            do_visualize(MODEL, args.load)
        else:
            predcfg = PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.load),
                input_names=MODEL.get_inference_tensor_names()[0],
                output_names=MODEL.get_inference_tensor_names()[1])
            if args.predict:
                do_predict(OfflinePredictor(predcfg), args.predict)
            elif args.evaluate:
                assert args.evaluate.endswith('.json'), args.evaluate
                do_evaluate(predcfg, args.evaluate)
    else:
        is_horovod = cfg.TRAINER == 'horovod'
        if is_horovod:
            hvd.init()
            logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

        if not is_horovod or hvd.rank() == 0:
            logger.set_logger_dir(args.logdir, 'd')

        finalize_configs(is_training=True)
        stepnum = cfg.TRAIN.STEPS_PER_EPOCH

        # warmup is step based, lr is epoch based
        init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
        warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
        warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
        lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

        factor = 8. / cfg.TRAIN.NUM_GPUS
        for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
            mult = 0.1 ** (idx + 1)
            lr_schedule.append(
                (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
        logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
        logger.info("LR Schedule (epochs, value): " + str(lr_schedule))

        #####################################################################
        if BATCH_ENABLE:
        #####################################################################
            train_dataflow = get_batch_train_dataflow(BATCH_SIZE_PLACEHOLDER)
        else:
            train_dataflow = get_train_dataflow()
        #####################################################################

        # This is what's commonly referred to as "epochs"
        total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
        logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

        bsize = BATCH_SIZE_PLACEHOLDER if BATCH_ENABLE else 0

        callbacks = [
            PeriodicCallback(
                ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                every_k_epochs=20),
            # linear warmup
            ScheduledHyperParamSetter(
                'learning_rate', warmup_schedule, interp='linear', step_based=True),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            PeakMemoryTracker(),
            EstimatedTimeLeft(median=True),
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
        ] + [
            EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir, bsize)
            for dataset in cfg.DATA.VAL
        ]
        if not is_horovod:
            callbacks.append(GPUUtilizationTracker())

        if args.perf:
            callbacks.append(ThroughputTracker(BATCH_SIZE_PLACEHOLDER*cfg.TRAIN.NUM_GPUS,
                                               args.num_total_images,
                                               trigger_every_n_steps=args.throughput_log_freq,
                                               log_fn=logger.info))

            if args.tfprof:
                # We only get tf profiling chrome trace on rank==0
                if hvd.rank() == 0:
                    callbacks.append(EnableCallbackIf(
                        GraphProfiler(dump_tracing=True, dump_event=True),
                        lambda self: self.trainer.global_step >= args.tfprof_start_step and self.trainer.global_step <= args.tfprof_end_step))

        if is_horovod and hvd.rank() > 0:
            session_init = None
        else:
            if args.load:
                session_init = get_model_loader(args.load)
            else:
                session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

        #session_config = tf.ConfigProto(device_count={'GPU': 1})
        #session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        #callbacks.append(DumpTensors([
        #    'group0/block2/output:0',
        #    'group1/block3/output:0',
        #    'group2/block5/output:0',
        #    'group3/block2/output:0'
        #]))

        traincfg = TrainConfig(
            model=MODEL,
            data=QueueInput(train_dataflow),
            callbacks=callbacks,
            extra_callbacks=[
               MovingAverageSummary(),
               ProgressBar(),
               MergeAllSummaries(period=args.summary_period),
               RunUpdateOps()
            ],
            steps_per_epoch=stepnum,
            max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
            session_init=session_init,
            session_config=None,
            starting_epoch=cfg.TRAIN.STARTING_EPOCH
        )


        if is_horovod:
            trainer = HorovodTrainer(average=False)
        else:
            # nccl mode appears faster than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
        launch_train_with_config(traincfg, trainer)
