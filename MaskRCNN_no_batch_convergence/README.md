# Convergence Testing

Add pieces of the batch code to the nobatch code 1-by-1. Each piece must have a flag that enables or disables it (hardcoded around line 55 of train.py). 
Before committing to master, we must test that the new pieces leads to convergence at target accuracy. 


## Current Work




## Individual blocks that require further investigation 

### RPN loss

Does not converge
4-5% off on bbox
2-3% off on segm


### Roi Align Mask

Converged before. Now not so sure. 
Definitely does not converge TTA. Fine on bbox, 3-8% off on segm













## Completed Work


### All disabled

Currently have 11 flags in existence. 
Converges consistently



### Bratin 20190322

BATCH_GENERATE_PROPOSALS
BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX
BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
BATCH_ROI_ALIGN_MASK

Running for convergence on Node 10, for throughput on Node 2
Does Not converge TTA on Segm, fine on bbox


### Bratin 20190322 FP16

Same as above but with FP16 
Does Not converge TTA on Segm, fine on bbox
FP16 did not seem to have any accuracy impact, but need to compare tfevents side-by-side to be confident about that.
66-67 img/s in steady state






### Layer 4, Block 1

All flags that appear to be non-problematic. A bit speculative since it sits on L2 and L3 blocks that have not been confirmed to work, but good use of parallelization.

BATCH_GENERATE_PROPOSALS
BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX
BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
BATCH_CROP_AND_RESIZE_MASK
BATCH_MASK_LOSS

Converges to target accuracy, 
50 img/s on p3dn



### Layer 4, Block 1 FP16

Same as above, but with FP16

Converges to target accuracy (maybe 0.2% low on small bbox and segm)

64 img/s on p3dn
59 img/s on p3.16xl





### Layer 3, Block 1

BATCH_GENERATE_PROPOSALS
BATCH_SAMPLE_TARGETS
BATCH_ROI_ALIGN_BOX

Converges TTA 


### Layer 3 Block 2

BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
BATCH_ROI_ALIGN_MASK

Converges TTA on bbox but not segm. ROIAlignMask is very likely culprit




### Layer 3, Block 2b

Since ROIAlignMask is problematic, replace L3-B2 (that uses ROIAlignMask) with L3-B2B which sits on top of L2-B5B (more flags, but none that we know are problematic)

BATCH_FAST_RCNN_OUTPUTS
BATCH_FAST_RCNN_LOSSES
BATCH_CROP_AND_RESIZE_MASK
BATCH_MASK_LOSS

This sits on L2-B5B which is not currently confirmed to be solid (running in parallel)

Converges to target accuracy. Might be a touch low (0.2%)





### Bratin data 20190321

BATCH_GENERATE_PROPOSALS = True

BATCH_ROI_ALIGN_BOX = True

BATCH_FAST_RCNN_OUTPUTS = True
BATCH_FAST_RCNN_LOSSES = True

BATCH_ROI_ALIGN_MASK = True
BATCH_CROP_AND_RESIZE_MASK = True
BATCH_MASK_LOSS = True

FP16

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



### Layer 2, Block 5b

Same as Layer 2, Block 5, but without ROIAlignMask that seems problematic

BATCH_CROP_AND_RESIZE_MASK
BATCH_MASK_LOSS

Converges. Small segm might be a touch low (0.2%), but probably not significant




## Individual Pieces




### GenerateProposals 

With new topk code, it converges!
 
### ROI Align box

Converges





### Sample targets

Fixed issue where randomness was disabled. 
Converges





### Mask crop and resize

Converges TTA. 
Retried after Layer 2, Block 5 issue. Still converges TTA individually



### Fast RCNN Outputs

Converges


### Fast RCNN Losses (includes FastRCNN.__init__)

Converges

### Mask Loss

Converges


### RPN Head Batch

Converged TTA





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