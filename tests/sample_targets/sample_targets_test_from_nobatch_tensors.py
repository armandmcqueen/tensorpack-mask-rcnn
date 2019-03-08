import os
import numpy as np
import tensorflow as tf

from MaskRCNN.model_frcnn import sample_fast_rcnn_targets_batch


class BatchSampleTargets(tf.test.TestCase):
    def testSampleTargets1(self):
        self.assertFastRCNNLoss("DumpTensor-1.npz")

    def testSampleTargets2(self):
        self.assertFastRCNNLoss("DumpTensor-2.npz")

    def testSampleTargets3(self):
        self.assertFastRCNNLoss("DumpTensor-3.npz")

    def testSampleTargets4(self):
        self.assertFastRCNNLoss("DumpTensor-4.npz")

    def testSampleTargets5(self):
        self.assertFastRCNNLoss("DumpTensor-5.npz")

    def testSampleTargets6(self):
        self.assertFastRCNNLoss("DumpTensor-6.npz")

    def testSampleTargets7(self):
        self.assertFastRCNNLoss("DumpTensor-7.npz")

    def testSampleTargets8(self):
        self.assertFastRCNNLoss("DumpTensor-8.npz")

    def testSampleTargets9(self):
        self.assertFastRCNNLoss("DumpTensor-9.npz")

    def testSampleTargets10(self):
        self.assertFastRCNNLoss("DumpTensor-10.npz")

    def testSampleTargets11(self):
        self.assertFastRCNNLoss("DumpTensor-11.npz")

    def testSampleTargets12(self):
        self.assertFastRCNNLoss("DumpTensor-12.npz")

    def testSampleTargets13(self):
        self.assertFastRCNNLoss("DumpTensor-13.npz")





    
    def assertSampleTargets(self, filename):
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "../sample_targets_test_data/nobatch")
        with self.test_session(use_gpu=True) as sess:
            path = os.path.join(dirname, filename)
            d = np.load(path)

            for k, v in d.items():
                print(k, v.shape)
            print("----------------------")


            ################################################################################################
            # Inputs from No Batch Codebase
            ################################################################################################
            proposal_boxes = d["oxbow_nobatch_input_proposal_boxes:0"]
            gt_boxes = d["oxbow_nobatch_input_gt_boxes:0"]
            gt_labels = d["oxbow_nobatch_input_gt_labels:0"]
            nobatch_output_proposal_boxes = d["oxbow_nobatch_output_proposal_boxes:0"]
            nobatch_output_proposal_labels = d["oxbow_nobatch_output_proposal_labels:0"]
            nobatch_output_proposal_fg_inds_wrt_gt = d["oxbow_nobatch_output_proposal_fg_inds_wrt_gt:0"]




            ## Missing tensors
            prepadding_gt_counts = np.asarray([gt_boxes.shape[0]])

            ################################################################################################
            # Convert Tensors to What Batch Codebase Expects
            ################################################################################################

            proposal_boxes = np.pad(proposal_boxes, [[0,0], [1, 0]], mode='constant', constant_values=0)
            gt_boxes = gt_boxes[np.newaxis, :]
            gt_labels = gt_labels[np.newaxis, :]

            nobatch_output_proposal_boxes = np.pad(nobatch_output_proposal_boxes, [[0, 0], [1, 0]], mode='constant', constant_values=0)
            nobatch_output_proposal_fg_inds_wrt_gt = nobatch_output_proposal_fg_inds_wrt_gt[np.newaxis, :]



            ################################################################################################
            # Run Through Batch Codebase
            ################################################################################################

            print("proposal_boxes (converted)", proposal_boxes.shape)
            print("gt_boxes (converted)", gt_boxes.shape)
            print("gt_labels (converted)", gt_labels.shape)
            print("prepadding_gt_counts (converted)", prepadding_gt_counts.shape)

            output_proposal_boxes, output_proposal_labels, output_proposal_gt_id_for_each_fg = sample_fast_rcnn_targets_batch(
                    proposal_boxes,
                    gt_boxes,
                    gt_labels,
                    prepadding_gt_counts,
                    batch_size=1)






            ################################################################################################
            # Compare nobatch and batch outputs
            ################################################################################################

            # actual_output_boxes = sorted(output_proposal_boxes.eval().tolist())
            # expected_output_boxes = sorted(nobatch_output_proposal_boxes.tolist())
            #
            # print(len(actual_output_boxes))
            # print(len(expected_output_boxes))
            #
            # for i, actual_box in enumerate(actual_output_boxes):
            #     expected_box = expected_output_boxes[i]
            #     print(f'-------------------{i}--------------')
            #     print(actual_box)
            #     print(expected_box)




            self.assertEqual(sorted(output_proposal_boxes.eval().tolist()),
                             sorted(nobatch_output_proposal_boxes.tolist()))

            self.assertEqual(sorted(output_proposal_labels.eval().tolist()),
                             sorted(nobatch_output_proposal_labels.tolist()))

            for i, t in enumerate(output_proposal_gt_id_for_each_fg):
                self.assertEqual(sorted(t.eval().tolist()),
                                 sorted(nobatch_output_proposal_fg_inds_wrt_gt[i].tolist()))



