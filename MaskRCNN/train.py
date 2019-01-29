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

import model_frcnn
import model_mrcnn
from basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from dataset import DetectionDataset
from config import finalize_configs, config as cfg
from data import get_all_anchors, get_all_anchors_fpn, get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, predict_image, multithread_predict_dataflow, EvalCallback
from model_box import RPNAnchors, clip_boxes, crop_and_resize, roi_align
from model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses
from model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, sample_fast_rcnn_targets
from model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head
from model_rpn import generate_rpn_proposals, rpn_head, rpn_losses
from viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


# Checks at graph_build time
def check_shape(name, tensor):
    print("[tshape] "+str(name)+": " + str(tensor.shape))

def print_runtime_shape(name, tensor):
    return tf.print("[runtime_shape] "+name+": "+str(tf.shape(tensor)))

class DetectionModel(ModelDesc):
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

    # def build_graph(self, *inputs):
    #     inputs = dict(zip(self.input_names, inputs))
    #
    #     image = self.preprocess(inputs['image'])     # 1CHW
    #
    #     features = self.backbone(image)
    #     anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
    #     proposals, rpn_losses = self.rpn(image, features, anchor_inputs)  # inputs?
    #
    #     targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
    #     head_losses = self.roi_heads(image, features, proposals, targets)
    #
    #     if self.training:
    #         wd_cost = regularize_cost(
    #             '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
    #         total_cost = tf.add_n(
    #             rpn_losses + head_losses + [wd_cost], 'total_cost')
    #         add_moving_summary(total_cost, wd_cost)
    #         return total_cost

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        images = self.preprocess(inputs['images'])     # NCHW

        check_shape('images', images)
        print_runtime_shape("Images", images)

        features = self.backbone(images)

        print("Features")
        print(type(features))
        for i, f in enumerate(features):
            t_name = "Feature["+str(i)+"].shape: "
            check_shape(t_name, f)

        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<0
        proposals, rpn_losses = self.rpn(images, features, anchor_inputs, inputs['orig_image_dims'])  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]


        head_losses = self.roi_heads(images, features, proposals, targets)

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost






class ResNetFPNModel(DetectionModel):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, None, 3), 'images'),            # N x H x W x C
            tf.placeholder(tf.int32, (None, 3), 'orig_image_dims')                  # N x 3(image dims - hwc)
        ]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, None, num_anchors),           # N x H x W x NumAnchors
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, None, num_anchors, 4),      # N x H x W x NumAnchors x 4
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, None, 4), 'gt_boxes'),                # N x MaxNumGTs x 4
            tf.placeholder(tf.int64, (None, None), 'gt_labels'),   # all > 0        # N x MaxNumGTs
            tf.placeholder(tf.int32, (None,), 'orig_gt_counts')                     # N
        ])


        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None, None), 'gt_masks')      # N x MaxNumGTs x H x W
            )
        return ret


    # def slice_feature_and_anchors(self, p23456, anchors, orig_image_dims):
    #     orig_image_dims_hw = orig_image_dims[:2]    # Remove irrelevant channel dimension
    #     for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
    #         with tf.name_scope('FPN_slice_lvl{}'.format(i)):
    #             orig_image_scale_factor = 1.0 / stride
    #             projection_dims = orig_image_dims_hw * orig_image_scale_factor
    #             anchors[i] = anchors[i].narrow_to(p23456[i], projection_dims)

    def backbone(self, images):
        c2345 = resnet_fpn_backbone(images, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        p23456 = fpn_model('fpn', c2345)
        return p23456


    def rpn(self, images, features, inputs, orig_image_dims):
        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

        image_shape2d = tf.shape(images)[2:]     # h,w
        all_anchors_fpn = get_all_anchors_fpn() # For a single image



        # check_shape("all_anchors_fpn", all_anchors_fpn)
        print(type(all_anchors_fpn))
        print("len: "+str(len(all_anchors_fpn)))

        batched_all_anchors_fpn = []
        for all_anchors_on_level in all_anchors_fpn:
            batched_all_anchors_on_level = tf.stack([all_anchors_on_level for i in range(self.batch_size)])
            batched_all_anchors_fpn.append(batched_all_anchors_on_level)



        multilevel_anchors = [RPNAnchors(
            batched_all_anchors_fpn[i],    # RPNAnchors needs a list of all_anchors for each image as later we will reduce the number of anchors on a per-image basis to match varying original image dimensions
            inputs['anchor_labels_lvl{}'.format(i + 2)],
            inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]


        # We skip narrowing the anchor set because we would just need to pad it again immediately.
        # self.slice_feature_and_anchors(features, multilevel_anchors, orig_image_dims)

        # Allow padding to flow until later


        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
                       for pi in features]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]



        for lvl, lvl_label_logits in enumerate(multilevel_label_logits):
            lvl_box_logits = multilevel_box_logits[lvl]
            check_shape("lvl " + str(lvl) + " box_logits", lvl_box_logits)
            check_shape("lvl " + str(lvl) + " label_logits", lvl_label_logits)


        multilevel_pred_boxes = [anchor.decode_logits(logits)
                                 for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]

        for lvl, lvl_pred_boxes in enumerate(multilevel_pred_boxes):
            check_shape("lvl "+str(lvl)+" pred_box", lvl_pred_boxes)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UNBATCH

        multilevel_anchors =      [RPNAnchors(b_anchors.boxes[0, :, :, :, :],
                                              b_anchors.gt_labels[0, :, :, :],
                                              b_anchors.gt_boxes[0, :, :, :, :]) for b_anchors in multilevel_anchors]
        multilevel_pred_boxes =   [b_pred_boxes[0, :, :, :, :] for b_pred_boxes in multilevel_pred_boxes]
        multilevel_box_logits =   [b_box_logits[0, :, :, :, :] for b_box_logits in multilevel_box_logits]
        multilevel_label_logits = [b_label_logits[0, :, :, :] for b_label_logits in multilevel_label_logits]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<1

        proposal_boxes, proposal_scores = generate_fpn_proposals(
            multilevel_pred_boxes, multilevel_label_logits, image_shape2d)

        if self.training:
            losses = multilevel_rpn_losses(
                multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
        else:
            losses = []

        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, images, features, proposals, targets):

        gt_boxes, gt_labels, *_ = targets

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UNBATCH

        images = images[0, :, :, :]
        gt_boxes = gt_boxes[0, :, :]
        gt_labels = gt_labels[0, :]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UNBATCH

        image_shape2d = tf.shape(images)[2:]     # h,w
        assert len(features) == 5, "Features have to be P23456!"



        if self.training:
            proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7)

        head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
            'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)
        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                     gt_boxes, tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))


        if self.training:
            all_losses = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                gt_masks = targets[2]

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UNBATCH

                gt_masks = gt_masks[0, :, :, :]

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UNBATCH

                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    features[:4], proposals.fg_boxes(), 14,
                    name_scope='multilevel_roi_align_mask')
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks, 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 28,
                    pad_border=False)  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))
            return all_losses
        else:
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
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')
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

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    batch_size=2

    MODEL = ResNetFPNModel(batch_size)
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
        train_dataflow = get_train_dataflow(batch_size)
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
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
        ] + [
            EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
            for dataset in cfg.DATA.VAL
        ]
        if not is_horovod:
            callbacks.append(GPUUtilizationTracker())

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
