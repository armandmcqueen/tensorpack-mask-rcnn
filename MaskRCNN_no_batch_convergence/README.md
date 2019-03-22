# Convergence Testing

Add pieces of the batch code to the nobatch code 1-by-1. Each piece must have a flag that enables or disables it (hardcoded around line 55 of train.py). 
Before committing to master, we must test that the new pieces leads to convergence at target accuracy. 


## Current Work


### All disabled

Currently have 11 flags in existence. 
Running on Nodes 8 and 9


### RPN loss

No longer certain. 
Running on Node 5. Seems to crash a lot.




### Layer 3, Block 1

BATCH_GENERATE_PROPOSALS
BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX

Running on Node 6.


### Layer 3 Block 2

BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
BATCH_ROI_ALIGN_MASK

Running on Node 7


### Batch crop and resize

This converged before, but retrying given Layer 2, Block 5 issues
Running on Node 4


### RPN Head Batch

Needs to be tested. Can working on it














### Bratin data 20190321

BATCH_GENERATE_PROPOSALS = True

BATCH_ROI_ALIGN_BOX = True

BATCH_FAST_RCNN_OUTPUTS = True
BATCH_FAST_RCNN_LOSSES = True

BATCH_ROI_ALIGN_MASK = True
BATCH_CROP_AND_RESIZE_MASK = True
BATCH_MASK_LOSS = True

Converges TTA on bbox but not segm

Throughput was measured on p3dn, accuracy on p3.16xl

66 img/s





### Layer 2, Block 1

BATCH_DATA_PIPELINE


### Layer 2, Block 2

BATCH_RPN_HEAD
BATCH_RPN_LOSS

Need to run this once BATCH_RPN_LOSS has been independently fixed. Maybe throw in BATCH_RPN_HEAD too

### Layer 2, Block 3

BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX

Converges


### Layer 2, Block 4

BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
Converges, maybe slightly (0.4-0.6%) low on small segmentation


### Layer 2, Block 5

BATCH_ROI_ALIGN_MASK
BATCH_CROP_AND_RESIZE_MASK
BATCH_MASK_LOSS

Did not converge on SEGME. Converged on BBOX





## Individual Pieces




### GenerateProposals 

With new topk code, it converges!
 
### ROI Align box

Converges


### ROI Align mask

Converges. Maybe 0.3% off on small objects. Probably deserves a repeat later


### Sample targets

Fixed issue where randomness was disabled. 
Converges





### Mask crop and resize

Converges.


### Fast RCNN Outputs

Converges


### Fast RCNN Losses (includes FastRCNN.__init__)

Converges

### Mask Loss

Converges




### To Add


* *Eval codepath* (needs to be further broken up, but we don't need it yet)

### Not worried about 

We're not worried about testing these in isolation due to a combination of Can's work and the fact that some are implicitly tested by other pieces. 

* Input pipeline
* FPN backbone
* get_all_anchors
* RPNAnchors + slice_feature_and_anchors
* rpn_head
* BoxProposals