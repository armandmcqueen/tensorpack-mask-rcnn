#! /usr/local/bin/python3

import numpy as np


def summarize_step(fpath):
    d = np.load(fpath)

    filenames = d["inputs_filenames:0"]
    images = d["inputs_images:0"]

    orig_image_dims = d["inputs_orig_image_dims:0"]
    orig_gt_counts = d["inputs_orig_gt_counts:0"]

    gt_boxes = d["inputs_gt_boxes:0"]
    gt_labels = d["inputs_gt_labels:0"]
    gt_masks = d["inputs_gt_masks:0"]


    anchor_labels_lvl2 = d["inputs_anchor_labels_lvl2:0"]
    anchor_boxes_lvl2 = d["inputs_anchor_boxes_lvl2:0"]
    anchor_labels_lvl3 = d["inputs_anchor_labels_lvl3:0"]
    anchor_boxes_lvl3 = d["inputs_anchor_boxes_lvl3:0"]
    anchor_labels_lvl4 = d["inputs_anchor_labels_lvl4:0"]
    anchor_boxes_lvl4 = d["inputs_anchor_boxes_lvl4:0"]
    anchor_labels_lvl5 = d["inputs_anchor_labels_lvl5:0"]
    anchor_boxes_lvl5 = d["inputs_anchor_boxes_lvl5:0"]
    anchor_labels_lvl6 = d["inputs_anchor_labels_lvl6:0"]
    anchor_boxes_lvl6 = d["inputs_anchor_boxes_lvl6:0"]

    print(filenames)
    print(orig_image_dims)
    print(orig_gt_counts)
    print(gt_boxes)
    print(gt_labels)



if __name__ == '__main__':

    for i in range(10):
        batch_npz_path = f'test_data/batch/DumpTensor-{i+1}.npz'
        nobatch_npz_path = f'test_data/nobatch/DumpTensor-{i+1}.npz'

        print("----------------------------------------------------")
        print("BATCH")
        summarize_step(batch_npz_path)

        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("NOBATCH")
        summarize_step(nobatch_npz_path)





