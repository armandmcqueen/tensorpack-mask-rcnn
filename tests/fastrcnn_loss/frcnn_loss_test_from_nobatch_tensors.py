import os
import numpy as np
import tensorflow as tf

from MaskRCNN.model_frcnn import FastRCNNHeadBatch
# from MaskRCNN_no_batch.model_frcnn import FastRCNNHead, BoxProposals

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



    
    def assertFastRCNNLoss(self, filename):
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "../fastrcnn_loss_test_data/nobatch")
        with self.test_session(use_gpu=True) as sess:
            path = os.path.join(dirname, filename)
            d = np.load(path)

            # for k, v in d.items():
            #     print(k, v.shape)

            ################################################################################################
            # Inputs from No Batch Codebase
            ################################################################################################
            proposal_boxes = d["oxbow_nobatch_input_proposal_boxes:0"]
            proposal_labels = d["oxbow_nobatch_input_proposal_labels:0"]
            proposal_fg_boxes = d["oxbow_nobatch_input_proposal_fg_boxes:0"]
            proposal_fg_inds = d["oxbow_nobatch_input_proposal_fg_inds:0"]
            proposal_fg_labels = d["oxbow_nobatch_input_proposal_fg_labels:0"]
            proposal_gt_id_for_each_fg = d["oxbow_nobatch_input_proposal_fg_inds_wrt_gt:0"]
            label_logits = d["oxbow_nobatch_input_fastrcnn_label_logits:0"]
            box_logits = d["oxbow_nobatch_input_fastrcnn_box_logits:0"]
            gt_boxes = d["oxbow_nobatch_input_gt_boxes:0"]
            regression_weights = d["oxbow_nobatch_input_regression_weights:0"]
            nobatch_label_loss = d["oxbow_nobatch_output_label_loss:0"]
            nobatch_box_loss= d["oxbow_nobatch_output_box_loss:0"]


            ## Missing tensors
            batch_indices_for_rois = np.asarray([0 for _ in range(proposal_labels.size)])
            prepadding_gt_counts = np.asarray([gt_boxes.shape[0]])

            ################################################################################################
            # Convert Tensors to What Batch Codebase Expects
            ################################################################################################

            proposal_boxes = np.pad(proposal_boxes, [[0,0], [1, 0]], mode='constant', constant_values=0)
            proposal_fg_boxes = np.pad(proposal_fg_boxes, [[0, 0], [1, 0]], mode='constant', constant_values=0)
            proposal_gt_id_for_each_fg = proposal_gt_id_for_each_fg[np.newaxis, :]
            gt_boxes = gt_boxes[np.newaxis, :]

            # print("proposal_boxes", proposal_boxes.shape)
            # print("proposal_fg_boxes", proposal_fg_boxes.shape)
            # print("proposal_gt_id_for_each_fg", proposal_gt_id_for_each_fg.shape)
            # print("gt_boxes", gt_boxes.shape)


            ################################################################################################
            # Run Through Batch Codebase
            ################################################################################################

            fastrcnn_head = FastRCNNHeadBatch(box_logits,
                                              label_logits,
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


            all_losses = fastrcnn_head.losses()
            label_loss, box_loss = all_losses




            ################################################################################################
            # Compare nobatch and batch outputs
            ################################################################################################

            self.assertEqual(label_loss.eval(), nobatch_label_loss)
            self.assertEqual(box_loss.eval(), nobatch_box_loss)



            #self.assertAlmostEqual(label_loss.eval(), d['rpn_losses/label_loss:0'], places=8)
            #self.assertAlmostEqual(box_loss.eval(), d['rpn_losses/box_loss:0'], places=8)

