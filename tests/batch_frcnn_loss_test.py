import os
import numpy as np
import tensorflow as tf

from MaskRCNN.model_frcnn import fastrcnn_losses

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
    
    def assertFastRCNNLoss(self, filename):
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, "frcnn_loss_test_data")
        with self.test_session(use_gpu=True) as sess:
            path = os.path.join(dirname, filename)
            d = np.load(path)
                
            for k, v in d.items():
                print(k, v.shape)

            labels = d["sample_fast_rcnn_targets/sampled_labels:0"]
            label_logits = d["fastrcnn/outputs/class/output:0"]
            fg_boxes = d["mul_1:0"]
            fg_box_logits = d["fg_box_logits:0"]

            label_loss, box_loss = fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits)

            self.assertEqual(label_loss.eval(), d["fastrcnn_losses/label_loss:0"])
            self.assertEqual(box_loss.eval(), d["fastrcnn_losses/box_loss:0"])
            #self.assertAlmostEqual(label_loss.eval(), d['rpn_losses/label_loss:0'], places=8)
            #self.assertAlmostEqual(box_loss.eval(), d['rpn_losses/box_loss:0'], places=8)

