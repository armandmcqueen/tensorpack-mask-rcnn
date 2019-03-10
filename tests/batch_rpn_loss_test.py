import os
import numpy as np
import tensorflow as tf

from MaskRCNN.model_box import RPNAnchors
from MaskRCNN.model_fpn import multilevel_rpn_losses_batch

class BatchMultiLevelRPNLossTest(tf.test.TestCase):

    def testMultilevelRpnLoss1(self):
        self.assertMultilevelRpnLoss("DumpTensor-1.npz")

    def testMultilevelRpnLoss2(self):
        self.assertMultilevelRpnLoss("DumpTensor-2.npz")

    def testMultilevelRpnLoss3(self):
        self.assertMultilevelRpnLoss("DumpTensor-3.npz")

    def testMultilevelRpnLoss4(self):
        self.assertMultilevelRpnLoss("DumpTensor-4.npz")

    def testMultilevelRpnLoss5(self):
        self.assertMultilevelRpnLoss("DumpTensor-5.npz")

    def testMultilevelRpnLoss6(self):
        self.assertMultilevelRpnLoss("DumpTensor-6.npz")

    def testMultilevelRpnLoss7(self):
        self.assertMultilevelRpnLoss("DumpTensor-7.npz")

    def testMultilevelRpnLoss8(self):
        self.assertMultilevelRpnLoss("DumpTensor-8.npz")

    def testMultilevelRpnLoss9(self):
        self.assertMultilevelRpnLoss("DumpTensor-9.npz")

    def testMultilevelRpnLoss10(self):
        self.assertMultilevelRpnLoss("DumpTensor-10.npz")

    def testMultilevelRpnLoss11(self):
        self.assertMultilevelRpnLoss("DumpTensor-11.npz")

    def testMultilevelRpnLoss13(self):
        self.assertMultilevelRpnLoss("DumpTensor-12.npz")

    def assertMultilevelRpnLoss(self, filename):
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "rpn_loss_test_data")
        with self.test_session(use_gpu=True) as sess:
            path = os.path.join(dirname, filename)
            d = np.load(path)
                
            #for k, v in d.items():
                #print(k, v.shape)

            multilevel_anchors = [
                RPNAnchors(
                    boxes=d['FPN_slice_lvl0/narrow_to/Slice:0'][np.newaxis, :],
                    gt_labels=d['FPN_slice_lvl0/narrow_to/Slice_1:0'][np.newaxis, :],
                    gt_boxes=d['FPN_slice_lvl0/narrow_to/Slice_2:0'][np.newaxis, :]
                ),
                RPNAnchors(
                    boxes=d['FPN_slice_lvl1/narrow_to/Slice:0'][np.newaxis, :],
                    gt_labels=d['FPN_slice_lvl1/narrow_to/Slice_1:0'][np.newaxis, :],
                    gt_boxes=d['FPN_slice_lvl1/narrow_to/Slice_2:0'][np.newaxis, :]
                ),
                RPNAnchors(
                    boxes=d['FPN_slice_lvl2/narrow_to/Slice:0'][np.newaxis, :],
                    gt_labels=d['FPN_slice_lvl2/narrow_to/Slice_1:0'][np.newaxis, :],
                    gt_boxes=d['FPN_slice_lvl2/narrow_to/Slice_2:0'][np.newaxis, :]
                ),
                RPNAnchors(
                    boxes=d['FPN_slice_lvl3/narrow_to/Slice:0'][np.newaxis, :],
                    gt_labels=d['FPN_slice_lvl3/narrow_to/Slice_1:0'][np.newaxis, :],
                    gt_boxes=d['FPN_slice_lvl3/narrow_to/Slice_2:0'][np.newaxis, :]
                ),
                RPNAnchors(
                    boxes=d['FPN_slice_lvl4/narrow_to/Slice:0'][np.newaxis, :],
                    gt_labels=d['FPN_slice_lvl4/narrow_to/Slice_1:0'][np.newaxis, :],
                    gt_boxes=d['FPN_slice_lvl4/narrow_to/Slice_2:0'][np.newaxis, :]
                )
            ]

            multilevel_label_logits = [
                d['rpn/Squeeze:0'][np.newaxis, :],
                d['rpn_1/Squeeze:0'][np.newaxis, :],
                d['rpn_2/Squeeze:0'][np.newaxis, :],
                d['rpn_3/Squeeze:0'][np.newaxis, :],
                d['rpn_4/Squeeze:0'][np.newaxis, :],
            ]

            multilevel_box_logits = [
                # Collapse last 2 dims.  Move last dim to first axis.  Add batch axis
                d['rpn/Reshape:0'].reshape(*d['rpn/Reshape:0'].shape[:2], -1).transpose(2, 0, 1)[np.newaxis, :],
                d['rpn_1/Reshape:0'].reshape(*d['rpn_1/Reshape:0'].shape[:2], -1).transpose(2, 0, 1)[np.newaxis, :],
                d['rpn_2/Reshape:0'].reshape(*d['rpn_2/Reshape:0'].shape[:2], -1).transpose(2, 0, 1)[np.newaxis, :],
                d['rpn_3/Reshape:0'].reshape(*d['rpn_3/Reshape:0'].shape[:2], -1).transpose(2, 0, 1)[np.newaxis, :],
                d['rpn_4/Reshape:0'].reshape(*d['rpn_4/Reshape:0'].shape[:2], -1).transpose(2, 0, 1)[np.newaxis, :],
            ]

            label_loss, box_loss = multilevel_rpn_losses_batch(
                multilevel_anchors, 
                multilevel_label_logits, 
                multilevel_box_logits)

            #self.assertEqual(label_loss.eval(), d['rpn_losses/label_loss:0'])
            #self.assertEqual(box_loss.eval(), d['rpn_losses/box_loss:0'])
            self.assertAlmostEqual(label_loss.eval(), d['rpn_losses/label_loss:0'], places=8)
            self.assertAlmostEqual(box_loss.eval(), d['rpn_losses/box_loss:0'], places=8)
