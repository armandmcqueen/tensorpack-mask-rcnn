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
    from .basemodel import image_preprocess, resnet_fpn_backbone
    from .dataset import DetectionDataset
    from .config import finalize_configs, config as cfg
    from .data import get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow
    from .eval import DetectionResult, predict_image, multithread_predict_dataflow
    from .model_box import RPNAnchors, clip_boxes, crop_and_resize
    from .model_fpn import fpn_model, multilevel_roi_align, multilevel_rpn_losses_batch, multilevel_roi_align_batch, \
        generate_fpn_proposals_batch_tf_op, multilevel_rpn_losses_batch_shortcut
    from .model_frcnn import fastrcnn_predictions, sample_fast_rcnn_targets_batch, fastrcnn_outputs_batch, FastRCNNHeadBatch
    from .model_mrcnn import maskrcnn_loss
    from .model_rpn import rpn_head
    from .viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall
    from .perf import print_runtime_shape, print_buildtime_shape, runtime_print, print_runtime_tensor, \
        print_runtime_tensor_loose_branch, ThroughputTracker
    #import .model_frcnn
    #import .model_mrcnn
else:
    from MaskRCNN.basemodel import image_preprocess, resnet_fpn_backbone
    from MaskRCNN.dataset import DetectionDataset
    from MaskRCNN.config import finalize_configs, config as cfg
    from MaskRCNN.data import get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow
    from MaskRCNN.eval import DetectionResult, predict_image, multithread_predict_dataflow
    from MaskRCNN.model_box import RPNAnchors, clip_boxes, crop_and_resize
    from MaskRCNN.model_fpn import fpn_model, multilevel_roi_align, multilevel_rpn_losses_batch, multilevel_roi_align_batch, \
        generate_fpn_proposals_batch_tf_op, multilevel_rpn_losses_batch_shortcut
    from MaskRCNN.model_frcnn import fastrcnn_predictions, sample_fast_rcnn_targets_batch, fastrcnn_outputs_batch, FastRCNNHeadBatch
    from MaskRCNN.model_mrcnn import maskrcnn_loss
    from MaskRCNN.model_rpn import rpn_head
    from MaskRCNN.viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall
    from MaskRCNN.perf import print_runtime_shape, print_buildtime_shape, runtime_print, print_runtime_tensor, \
        print_runtime_tensor_loose_branch, ThroughputTracker
    import MaskRCNN.model_frcnn as model_frcnn
    import MaskRCNN.model_mrcnn as model_mrcnn

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

USE_RPN_LOSS_SHORTCUT = False
USE_FASTRCNN_LOSS_SHORTCUT = False
USE_MASK_LOSS_SHORTCUT = False




class ResNetFPNModel(ModelDesc):

    def __init__(self, batch_size, fp16=False):
        self.batch_size = batch_size
        self.fp16 = fp16

    def preprocess(self, image):
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
        out = ['output/boxes', 'output/scores', 'output/labels']
        if cfg.MODE_MASK:
            out.append('output/masks')
        return ['images'], out

    def build_graph(self, *inputs):
        prefix = "train.build_graph"

        inputs = dict(zip(self.input_names, inputs))
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        prepadding_dims = inputs['orig_image_dims']

        images = self.preprocess(inputs['images'])  # NCHW

        # Print the filenames and dimensions for images in the batch
        # images = print_runtime_tensor_loose_branch("filenames", inputs["filenames"], prefix=prefix, trigger_tensor=images)
        # prepadding_dims = print_runtime_tensor("prepadding_dims", prepadding_dims, prefix=prefix)

        features = self.backbone(images)


        proposal_boxes, rpn_losses = self.rpn(images, features, anchor_inputs, prepadding_dims)

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(images, features, proposal_boxes, targets, inputs['orig_gt_counts'], prepadding_dims)

        if self.training:
            wd_cost = regularize_cost(
                    '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

            rpn_label_loss, rpn_box_loss = rpn_losses
            fr_label_loss, fr_box_loss, mask_loss = head_losses

            wd_cost = print_runtime_tensor("wd_cost", wd_cost, prefix="train.py")
            rpn_label_loss = print_runtime_tensor("rpn_label_loss", rpn_label_loss, prefix="train.py")
            rpn_box_loss = print_runtime_tensor("rpn_box_loss", rpn_box_loss, prefix="train.py")
            fr_label_loss = print_runtime_tensor("fr_label_loss", fr_label_loss, prefix="train.py")
            fr_box_loss = print_runtime_tensor("fr_box_loss", fr_box_loss, prefix="train.py")
            mask_loss = print_runtime_tensor("mask_loss", mask_loss, prefix="train.py")

            head_losses = [fr_label_loss, fr_box_loss, mask_loss]
            rpn_losses = [rpn_label_loss, rpn_box_loss]


            total_cost = tf.add_n(
                    rpn_losses + head_losses + [wd_cost], 'total_cost')

            total_cost = print_runtime_tensor("total_cost", total_cost, prefix="train.py")

            add_moving_summary(total_cost, wd_cost)
            return total_cost

    def inputs(self):

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


        return ret

    def slice_feature_and_anchors_batch(self, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_batch_slice_lvl{}'.format(i)):
                anchors[i] = anchors[i].narrow_to_batch(p23456[i])

    def backbone(self, images):
        c2345 = resnet_fpn_backbone(images, cfg.BACKBONE.RESNET_NUM_BLOCKS, fp16=self.fp16)
        p23456 = fpn_model('fpn', c2345, fp16=self.fp16)
        return p23456

    def rpn(self, images, features, inputs, orig_image_dims):
        """
        Returns:
            proposal_boxes: Nx5 (batch_ind, x1, y1, x2, y2)
            rpn_losses: list of tf.float32
        """
        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)


        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS), fp16=self.fp16)
                       for pi in features]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs] # BS x (NAx4) x fH x fW



        proposal_boxes, proposal_scores = generate_fpn_proposals_batch_tf_op(multilevel_box_logits,
                                                                             multilevel_label_logits,
                                                                             orig_image_dims)





        ############################################################################################
        # Old way of handling anchors. Might be able to simplify. Only used by rpn_losses
        ############################################################################################
        all_anchors_fpn = get_all_anchors_fpn()  # For a single image. List, with anchors for each level
        batched_all_anchors_fpn = []

        for i, all_anchors_on_level in enumerate(all_anchors_fpn):
            batched_all_anchors_on_level = tf.stack([all_anchors_on_level for _ in range(self.batch_size)])
            batched_all_anchors_fpn.append(batched_all_anchors_on_level)

        multilevel_anchors = [RPNAnchors(
                batched_all_anchors_fpn[i],
                inputs['anchor_labels_lvl{}'.format(i + 2)],
                inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]
        self.slice_feature_and_anchors_batch(features, multilevel_anchors)
        ############################################################################################

        if self.training:

            if USE_RPN_LOSS_SHORTCUT:
                losses = multilevel_rpn_losses_batch_shortcut(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
            else:
                losses = multilevel_rpn_losses_batch(multilevel_anchors, multilevel_label_logits, multilevel_box_logits)

        else:
            losses = []

        return proposal_boxes, losses







    def roi_heads(self, images, features, proposal_boxes, targets, prepadding_gt_counts, orig_image_dims):
        prefix = "train.roi_heads"
        assert len(features) == 5, "Features have to be P23456!"

        gt_boxes, gt_labels, *_ = targets
        image_shape2d = tf.shape(images)[2:]  # h,w

        if self.training:
            proposal_boxes, proposal_labels, proposal_gt_id_for_each_fg = sample_fast_rcnn_targets_batch(
                proposal_boxes,
                gt_boxes,
                gt_labels,
                prepadding_gt_counts,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU)

            proposal_fg_inds = tf.reshape(tf.where(proposal_labels > 0), [-1])
            proposal_fg_boxes = tf.gather(proposal_boxes, proposal_fg_inds)
            proposal_fg_labels = tf.gather(proposal_labels, proposal_fg_inds)




        roi_feature_fastrcnn, batch_indices_for_rois = multilevel_roi_align_batch(features[:4], proposal_boxes, 7, runtime_tag="box")



        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn, fp16=self.fp16)

        print_buildtime_shape("head_feature", head_feature, prefix=prefix)
        # head_feature = print_runtime_shape(f'head_feature', head_feature, prefix=prefix)


        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs_batch('fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)

        # fastrcnn_label_logits = print_runtime_shape(f'fastrcnn_label_logits', fastrcnn_label_logits, prefix=prefix)
        # fastrcnn_box_logits = print_runtime_shape(f'fastrcnn_box_logits', fastrcnn_box_logits, prefix=prefix)

        regression_weights = tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32, name="bbox_regression_weights")

        fastrcnn_head = FastRCNNHeadBatch(fastrcnn_box_logits,
                                          fastrcnn_label_logits,
                                          gt_boxes,
                                          regression_weights,
                                          batch_indices_for_rois,
                                          prepadding_gt_counts,
                                          proposal_boxes,
                                          proposal_labels,
                                          proposal_fg_inds,
                                          proposal_fg_boxes,
                                          proposal_fg_labels,
                                          proposal_gt_id_for_each_fg)





        if self.training:
            print("self.training == True")
            # Using fastrcnn loss shortcut
            all_losses = fastrcnn_head.losses(shortcut=USE_FASTRCNN_LOSS_SHORTCUT)

            if cfg.MODE_MASK:
                gt_masks = targets[2]

                # maskrcnn loss
                roi_feature_maskrcnn, _ = multilevel_roi_align_batch(
                        features[:4], proposal_fg_boxes, 14,
                        runtime_tag=f'mask',
                        verbose=True,
                        name_scope='multilevel_roi_align_mask')


                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                        'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, fp16=self.fp16)  # BS x #fg x #cat x 28 x 28


                per_image_target_masks_for_fg = []
                per_image_fg_labels = []
                for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                    prefix = f'roi_heads, batch_idx {i}'
                    single_image_gt_masks = gt_masks[i, :, :, :]
                    single_image_fg_indices = tf.squeeze(tf.where(tf.equal(proposal_fg_boxes[:, 0], i)), axis=1)
                    single_image_fg_boxes = tf.gather(proposal_fg_boxes, single_image_fg_indices)[:, 1:]
                    single_image_fg_labels = tf.gather(proposal_fg_labels, single_image_fg_indices)
                    single_image_fg_inds_wrt_gt = proposal_gt_id_for_each_fg[i]

                    single_image_gt_masks = tf.expand_dims(single_image_gt_masks, axis=1)

                    single_image_target_masks_for_fg = crop_and_resize(single_image_gt_masks,
                                                                             single_image_fg_boxes,
                                                                             single_image_fg_inds_wrt_gt,
                                                                             28,
                                                                             orig_image_dims[i],
                                                                             pad_border=False,
                                                                            verbose_batch_index=i)  # fg x 1x28x28

                    print_buildtime_shape("single_image_image_target_masks_for_fg", single_image_target_masks_for_fg, prefix=prefix)

                    # single_image_target_masks_for_fg = print_runtime_shape(f'single_image_target_masks_for_fg {i}', single_image_target_masks_for_fg, prefix=prefix)

                    per_image_fg_labels.append(single_image_fg_labels)
                    per_image_target_masks_for_fg.append(single_image_target_masks_for_fg)


                target_masks_for_fg = tf.concat(per_image_target_masks_for_fg, axis=0)
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')

                fg_labels = tf.concat(per_image_fg_labels, axis=0)


                all_losses.append(maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg, shortcut=USE_MASK_LOSS_SHORTCUT))
            return all_losses
        else:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BATCH BOUNDARY (1)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NOT YET BATCHIFIED
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                    decoded_boxes, label_scores, name_scope='output')
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                        'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, fp16=self.fp16)  # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')
            return []
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NOT YET BATCHIFIED









def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()  # we don't visualize mask stuff
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
    parser.add_argument('--fp16', help="Train in FP16", action="store_true")

    #################################################################################################################
    # Performance investigation arguments
    parser.add_argument('--perf', help="Enable performance investigation mode", action="store_true")
    parser.add_argument('--throughput_log_freq', help="In perf investigation mode, code will print throughput after "
                                                      "every throughput_log_freq steps as well as after every epoch",
                        type=int, default=1)
    parser.add_argument('--images_per_step', help="Number of images in a minibatch (total, not per GPU)",
                        type=int, default=1)
    parser.add_argument('--num_total_images', help="Number of images in an epoch. = images_per_steps * steps_per_epoch "
                                                   "(differs slightly from the total number of images).",
                        type=int, default=120000)

    parser.add_argument('--tfprof', help="Enable tf profiller", action="store_true")
    parser.add_argument('--tfprof_start_step', help="Step to enable tf profiling", type=int, default=15005)
    parser.add_argument('--tfprof_end_step', help="Step after which tf profiling will be disabled",
                        type=int, default=15010)

    parser.add_argument('--summary_period', help="Write summary events periodically at this interval. Setting it to 0 "
                                                 "writes at the end of an epoch. Increasing frequency hurt throughput",
                        type=int, default=0)


    #################################################################################################################

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)



    DetectionDataset()  # initialize the config with information from our dataset

    if args.visualize or args.evaluate or args.predict:
        assert tf.test.is_gpu_available()
        assert args.load
        finalize_configs(is_training=False)
        MODEL = ResNetFPNModel(cfg.TRAIN.BATCH_SIZE_PER_GPU, args.fp16)

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
        MODEL = ResNetFPNModel(cfg.TRAIN.BATCH_SIZE_PER_GPU, args.fp16)
        print("Batch size per GPU: " + str(cfg.TRAIN.BATCH_SIZE_PER_GPU))
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
        # train_dataflow = get_train_dataflow(cfg.TRAIN.BATCH_SIZE_PER_GPU)
        train_dataflow = get_train_dataflow(cfg.TRAIN.BATCH_SIZE_PER_GPU)
        # This is what's commonly referred to as "epochs"
        total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
        logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

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
                        SessionRunTimeout(60000).set_chief_only(True),  # 1 minute timeout
                    ]

        # callbacks.extend([EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
        #                 for dataset in cfg.DATA.VAL])

        # callbacks.append(DumpTensors([
        #     "oxbow_batch_input_proposal_boxes:0",
        #     "oxbow_batch_input_gt_boxes:0",
        #     "oxbow_batch_input_gt_labels:0",
        #     "oxbow_batch_input_prepadding_gt_counts:0",
        #     "oxbow_batch_output_proposal_boxes:0",
        #     "oxbow_batch_output_proposal_labels:0",
        #     "oxbow_batch_output_proposal_gt_id_for_each_fg:0"
        # ]))


        if not is_horovod:
            callbacks.append(GPUUtilizationTracker())

        if args.perf:
            callbacks.append(ThroughputTracker(args.images_per_step,
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
                starting_epoch=cfg.TRAIN.STARTING_EPOCH
        )
        if is_horovod:
            trainer = HorovodTrainer(average=False)
        else:
            # nccl mode appears faster than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
        launch_train_with_config(traincfg, trainer)
