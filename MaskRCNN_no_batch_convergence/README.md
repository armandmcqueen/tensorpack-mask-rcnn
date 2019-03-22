# Convergence Testing

Add pieces of the batch code to the nobatch code 1-by-1. Each piece must have a flag that enables or disables it (hardcoded around line 55 of train.py). 
Before committing to master, we must test that the new pieces leads to convergence at target accuracy. 


## Combined Pieces

### All that have been tested

BATCH_GENERATE_PROPOSALS = True

BATCH_ROI_ALIGN_BOX = True

BATCH_FAST_RCNN_OUTPUTS = True
BATCH_FAST_RCNN_LOSSES = True

BATCH_ROI_ALIGN_MASK = True
BATCH_CROP_AND_RESIZE_MASK = True
BATCH_MASK_LOSS = True

Converges TTA on bbox but not segm
66 img/s





### Set 1

BATCH_GENERATE_PROPOSALS
BATCH_RPN_LOSS

Need to run this once BATCH_RPN_LOSS has been independently fixed. Maybe throw in rpn_head too

### Set 2

BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX

Ran on Node 6 (need to push tfevents)
Converges


### Set 3

BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES

Ran on Node 7 (need to send tfevents)
Converges

### Set 4

BATCH_ROI_ALIGN_MASK
BATCH_CROP_AND_RESIZE_MASK
BATCH_MASK_LOSS

Running on Node 8
Converged on bbox but not segm





## Individual Pieces

### All disabled

Converges
Running on Node 2 to check with 9 flags, all disabled.


### GenerateProposals 

With new topk code, it converges!
 
### ROI Align box

Converges


### ROI Align mask

Converges. Maybe 0.3% off on small objects. Probably deserves a repeat later


### Sample targets

Fixed issue where randomness was disabled. 
Ran on Node 1 (need to copy tfevents)
Converges

### RPN loss

No longer certain. 
Running on Node 5



### Mask crop and resize

Converges.


### Fast RCNN Outputs

Converges


### Fast RCNN Losses (includes FastRCNN.__init__)

Converges

### Mask Loss

Converges


### RPN Head Batch

Needs to be tested. Can working on it


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