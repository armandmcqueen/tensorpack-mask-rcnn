from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import ops

import tensorflow as tf

ops.NotDifferentiable('NonMaxSuppressionCustom')

nms_custom_lib_abspath = os.path.join(tf.resource_loader.get_data_files_path(),
                                      'non_max_suppression_custom_op.so')
nms_custom = tf.load_op_library(nms_custom_lib_abspath)
                               

def non_max_suppression_custom(boxes,
                               scores,
                               max_output_size,
                               iou_threshold=0.5,
                               score_threshold=float('-inf'),
                               name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
  """
  with ops.name_scope(name, 'non_max_suppression_custom'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    return nms_custom.non_max_suppression_custom(boxes, scores, max_output_size,
                                                 iou_threshold, score_threshold)    
    #return gen_image_ops.non_max_suppression_v3(boxes, scores, max_output_size,
                                                #iou_threshold, score_threshold)


with tf.Session() as sess:
  x = non_max_suppression_custom(tf.constant([[1.1, 2.1, 2.1, 2.1], 
                                              [1.3, 2.4, 1.6, 2.4]]), 
                                 tf.constant([1.0, 0.5]), 
                                 tf.constant(1), 
                                 0.8, 0.8)
  z = sess.run([x])
  print(z)

