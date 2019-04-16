import tensorflow as tf
from tensorpack.tfutils.tower import get_current_tower_context
from config import config as cfg


# TODO: consider padding/unpadding

def serialize_backbone(fn, input_tensor, batch_size):
   single_tensors = tf.split(input_tensor, batch_size) 

   out_tensors_lvl_list = []
   for tensor in single_tensors:
       out_tensors_lvl_list.append(fn(tensor))

   return [tf.concat([out_tensors_lvl_list[i][lvl] for i in range(len(single_tensors))], axis=0) for lvl in range(len(out_tensors_lvl_list[0]))]


def serialize_rpn(fn, images, features, anchor_inputs, orig_image_dims, training, batch_size):

    serialized_proposals = []
    serialized_scores = []
    serialized_rpn_losses = []

    for i in range(batch_size): 
        single_image_features = [feature[i:(i+1)] for feature in features]
        single_image_unpadded = images[i:(i+1), :orig_image_dims[i,0], :orig_image_dims[i,1], :]
        single_image_anchors = {k: v[i:(i+1)] for k,v in anchor_inputs.items()} 

        single_image_proposals, single_image_rpn_losses, single_image_proposal_scores = fn(single_image_unpadded, single_image_features, anchor_inputs, orig_image_dims[i:(i+1)], batch_size=1)

        serialized_proposals.append(single_image_proposals) 
        serialized_scores.append(single_image_proposal_scores) 
        serialized_rpn_losses.append(single_image_rpn_losses) 

    # recombine proposals into batch -- need to be careful with batch indices, since they are all zero coming out of the rpn block
    for i in range(batch_size):
        serialized_proposals[i] = tf.concat((i+serialized_proposals[i][:, :1], serialized_proposals[i][:, 1:]), axis=1)

    boxes = tf.concat(serialized_proposals, axis=0)
    scores = tf.concat(serialized_scores, axis=0)

    # filter topk boxes across the batch
    training = get_current_tower_context().is_training
    fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK*batch_size if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK

    proposal_topk = tf.minimum(tf.size(scores), fpn_nms_topk)
    proposal_scores, topk_indices = tf.nn.top_k(scores, k=proposal_topk, sorted=False)
    proposal_boxes = tf.gather(boxes, topk_indices)


    # combine losses
    if training:
        label_losses = tf.stack([losses[0] for losses in serialized_rpn_losses])
        box_losses = tf.stack([losses[1] for losses in serialized_rpn_losses])

        rpn_losses = [tf.reduce_mean(label_losses), tf.reduce_mean(box_losses)]
    else:
        rpn_losses = []

    return proposal_boxes, rpn_losses
