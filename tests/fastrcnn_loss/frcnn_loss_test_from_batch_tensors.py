import os
import numpy as np
import tensorflow as tf

# from MaskRCNN.model_frcnn import FastRCNNHeadBatch
from MaskRCNN_no_batch.model_frcnn import FastRCNNHead, BoxProposals

class NoBatchFastRCNNLoss(tf.test.TestCase):
    def testFastRCNNLoss1(self):
        self.assertFastRCNNLoss("DumpTensor-1.npz")

    def testFastRCNNLoss2(self):
        self.assertFastRCNNLoss("DumpTensor-2.npz")

    def testFastRCNNLoss3(self):
        self.assertFastRCNNLoss("DumpTensor-3.npz")

    def testFastRCNNLoss4(self):
        self.assertFastRCNNLoss("DumpTensor-4.npz")
    
    def testFastRCNNLoss5(self):
        self.assertFastRCNNLoss("DumpTensor-5.npz")
    
    def testFastRCNNLoss6(self):
        self.assertFastRCNNLoss("DumpTensor-6.npz")
    
    def testFastRCNNLoss7(self):
        self.assertFastRCNNLoss("DumpTensor-7.npz")
    
    def testFastRCNNLoss8(self):
        self.assertFastRCNNLoss("DumpTensor-8.npz")
    
    def testFastRCNNLoss9(self):
        self.assertFastRCNNLoss("DumpTensor-9.npz")
    
    def testFastRCNNLoss10(self):
        self.assertFastRCNNLoss("DumpTensor-10.npz")
    
    def testFastRCNNLoss11(self):
        self.assertFastRCNNLoss("DumpTensor-11.npz")
    
    def testFastRCNNLoss12(self):
        self.assertFastRCNNLoss("DumpTensor-12.npz")

    def testFastRCNNLoss13(self):
        self.assertFastRCNNLoss("DumpTensor-13.npz")

    def testFastRCNNLoss14(self):
        self.assertFastRCNNLoss("DumpTensor-14.npz")

    def testFastRCNNLoss15(self):
        self.assertFastRCNNLoss("DumpTensor-15.npz")

    def testFastRCNNLoss16(self):
        self.assertFastRCNNLoss("DumpTensor-16.npz")

    
    def assertFastRCNNLoss(self, filename):
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "../fastrcnn_loss_test_data/batch")
        with self.test_session(use_gpu=True) as sess:
            path = os.path.join(dirname, filename)
            d = np.load(path)
                
            # for k, v in d.items():
            #     print(k, v.shape)

            ################################################################################################
            # Inputs from Batch Codebase
            ################################################################################################


            box_logits = d["unit_test_fastrcnn_box_logits:0"]
            label_logits = d["unit_test_fastrcnn_label_logits:0"]
            gt_boxes = d["unit_test_gt_boxes:0"]
            regression_weights = d["unit_test_regression_weights:0"]
            batch_indices_for_rois = d["unit_test_batch_indices_for_rois:0"]
            prepadding_gt_counts = d["unit_test_prepadding_gt_counts:0"]
            proposal_boxes = d["unit_test_proposal_boxes:0"]
            proposal_labels = d["unit_test_proposal_labels:0"]
            proposal_fg_inds = d["unit_test_proposal_fg_inds:0"]
            proposal_fg_boxes = d["unit_test_proposal_fg_boxes:0"]
            proposal_fg_labels = d["unit_test_proposal_fg_labels:0"]
            proposal_gt_id_for_each_fg = d["unit_test_proposal_gt_id_for_each_fg:0"]
            batch_label_loss = d["unit_test_fast_rcnn_label_loss:0"]
            batch_box_loss= d["unit_test_fast_rcnn_box_loss:0"]


            ################################################################################################
            # Run Through Non-Batch Codebase
            ################################################################################################
            nonbatch_proposal_boxes = proposal_boxes[:, 1:]
            nonbatch_gt_boxes = gt_boxes[0, :, :]
            nonbatch_proposal_gt_id_for_each_fg = proposal_gt_id_for_each_fg[0, :]
            box_proposals = BoxProposals(nonbatch_proposal_boxes, proposal_labels, nonbatch_proposal_gt_id_for_each_fg)
            fastrcnn_head = FastRCNNHead(box_proposals, box_logits, label_logits, nonbatch_gt_boxes, regression_weights)
            label_loss, box_loss = fastrcnn_head.losses()






            ################################################################################################
            # Compare nobatch and batch outputs
            ################################################################################################

            self.assertEqual(label_loss.eval(), batch_label_loss)
            self.assertEqual(box_loss.eval(), batch_box_loss)



            #self.assertAlmostEqual(label_loss.eval(), d['rpn_losses/label_loss:0'], places=8)
            #self.assertAlmostEqual(box_loss.eval(), d['rpn_losses/box_loss:0'], places=8)

