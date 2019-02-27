# -*- coding: utf-8 -*-
# File: box_ops.py

import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope


######################################################################################################################
# Checks at graph_build time
def print_buildtime_shape(name, tensor, prefix=None):
    if prefix is not None:
        prefix = f' [{prefix}]'
    else:
        prefix = ""

    print(f'[buildtime_shape]{prefix} {name}: {tensor.shape}')




def print_runtime_shape(name, tensor):
    s = "[runtime_shape] "+name+": "+str(tf.shape(tensor))
    return runtime_print(s, tensor)



# A method if you want tf.print to behave like tf.Print (i.e. the 'print' exists as an op in the computation graph)
"""
some_tensor = tf.op(some_other_tensor)
some_tensor = runtime_print("String to print", some_tensor)
"""
def runtime_print(message, trigger_tensor):
    print_op = tf.print(message)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)



def print_runtime_tensor(name, tensor, prefix=None, summarize=-1):
    s = "[runtime_tensor] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)

######################################################################################################################



@under_name_scope()
def flatten_gt_boxes(padded_gt_boxes, gt_counts):
    """
    Args:
        padded_gt_boxes: BS x MaxNumGTs x 4
        gt_counts: BS vector. The actual number of GTs for an image to know how much of MaxNumGTs is padding

    Returns:
        TotalNumGTs x 5         (batch_index, box)
    """

    batch_indices = tf.range(padded_gt_boxes.get_shape()[0])
    return tf.map_fn(flatten_single_image_boxes, [padded_gt_boxes, gt_counts, batch_indices])




def flatten_single_image_boxes(inputs):
    padded_gt_boxes, num_gts, batch_idx = inputs
    gt_boxes = padded_gt_boxes[0:num_gts, :]  # N x 4
    gt_boxes = tf.pad(gt_boxes,
                      paddings=[[0, 0], [1, 0]],
                      constant_values=batch_idx)
    return gt_boxes

"""
This file is modified from
https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py
"""


@under_name_scope()
def area(boxes):
    """
    Args:
      boxes: nx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def area_batch(boxes):
    """
    Args:
      boxes: nx5 floatbox

    Returns:
      n
    """
    prefix="tf_area_batch"
    print_buildtime_shape("boxes (raw)", boxes, prefix=prefix)
    boxes = boxes[:, 1:]
    print_buildtime_shape("boxes (processed)", boxes, prefix=prefix)
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))







@under_name_scope()
def pairwise_iou_batch(proposal_boxes, gt_boxes, orig_gt_counts, batch_size):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx5                             (batch_index, x1, y1, x2, t2)
      boxlist2: BS x MaxNumGTs x 4
      orig_gt_counts: BS

    Returns:
        list of length BS, each element is output of pairwise_iou: N x M
        (where N is number of boxes for image and M is number of GTs for image)
    """

    prefix = "pairwise_iou_batch"

    # For each image index, extract a ?x4 boxlist and gt_boxlist

    per_images_iou = []
    for batch_idx in range(batch_size):

        box_mask_for_image = tf.equal(proposal_boxes[:, 0], batch_idx)

        single_image_boxes = tf.boolean_mask(proposal_boxes, box_mask_for_image)
        single_image_boxes = single_image_boxes[:, 1:]
        single_image_gt_boxes = gt_boxes[batch_idx, 0:orig_gt_counts[batch_idx], :]
        single_image_iou = pairwise_iou(single_image_boxes, single_image_gt_boxes)

        per_images_iou.append(single_image_iou)

    return per_images_iou

