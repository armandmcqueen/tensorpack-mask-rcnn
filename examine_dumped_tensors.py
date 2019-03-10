import numpy as np


def summarize_step(fpath):
    tensor_dict = np.load(fpath)
    num_gt = tensor_dict["sample_fast_rcnn_targets/dump_gt_boxes:0"].shape[1]
    num_fg = tensor_dict["sample_fast_rcnn_targets/dump_num_fg-0:0"]
    print(fpath)
    print("num_gt", num_gt)
    print("num_fg", num_fg)
    print("non GT FG", num_fg-num_gt)

    image_shape = tensor_dict["dump_orig_image_dims:0"]
    print("image_shape", image_shape)


    rois = tensor_dict["sample_fast_rcnn_targets/dump_rois:0"] # (6509, 5)

    max_x = 0
    max_y = 0

    num_real = 0
    num_sizeable = 0
    for roi in rois.tolist():
        # print(roi)
        _, x1, y1, x2, y2 = roi
        w = x2 - x1
        h = y2 = y1
        real = w > 0 and h > 0
        area = h*w

        if y2 > max_y:
            max_y = y2

        if x2 > max_x:
            max_x = x2

        if real:
            num_real += 1

            if area > 20:
                num_sizeable += 1

    print("real_rois", num_real)
    print("sizeable_rois", num_sizeable)
    print("max_x", max_x)
    print("max_y", max_y)

    anchor_boxes = tensor_dict["dump_anchor_boxes_lvl2:0"]
    print(anchor_boxes)




if __name__ == '__main__':

    for i in range(104):
        npz_path = f'logs/train_log/DumpTensor-{i+1}.npz'
        print("----------------------------------------------------")


        summarize_step(npz_path)


# sample_fast_rcnn_targets/dump_rois:0 (6509, 5)
# sample_fast_rcnn_targets/dump_gt_boxes:0 (1, 3, 4)
# sample_fast_rcnn_targets/dump_gt_labels:0 (1, 3)
# sample_fast_rcnn_targets/dump_per_image_ious:0 (6509, 3)
# sample_fast_rcnn_targets/dump_single_image_gt_boxes-0:0 (3, 4)
# sample_fast_rcnn_targets/dump_iou-0:0 (6512, 3)
# sample_fast_rcnn_targets/dump_best_iou_ind-0:0 (6512,)
# sample_fast_rcnn_targets/dump_num_fg-0:0 ()
# sample_fast_rcnn_targets/dump_fg_inds_wrt_gt-0:0 (3,)
# sample_fast_rcnn_targets/dump_all_indices-0:0 (512,)
# dump_wd_cost:0 ()
# dump_rpn_label_loss:0 ()
# dump_rpn_box_loss:0 ()
# dump_mask_loss:0 ()
# dump_fr_label_loss:0 ()
# dump_fr_box_loss:0 ()
# dump_total_cost:0 ()
# dump_orig_image_dims:0 (1, 3)
# dump_anchor_labels_lvl2:0 (1, 336, 336, 3)
# dump_anchor_labels_lvl3:0 (1, 168, 168, 3)
# dump_anchor_labels_lvl4:0 (1, 84, 84, 3)
# dump_anchor_labels_lvl5:0 (1, 42, 42, 3)
# dump_anchor_labels_lvl6:0 (1, 21, 21, 3)
# dump_anchor_boxes_lvl2:0 (1, 336, 336, 3, 4)
# dump_anchor_boxes_lvl3:0 (1, 168, 168, 3, 4)
# dump_anchor_boxes_lvl4:0 (1, 84, 84, 3, 4)
# dump_anchor_boxes_lvl5:0 (1, 42, 42, 3, 4)
# dump_anchor_boxes_lvl6:0 (1, 21, 21, 3, 4)