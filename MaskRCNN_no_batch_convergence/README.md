# Convergence Testing

Add pieces of the batch code to the nobatch code 1-by-1. Each piece must have a flag that enables or disables it (hardcoded around line 55 of train.py). Before committing to master, we must test that the new pieces leads to convergence at target accuracy. 

## Pieces

### All disabled

Converges


### GenerateProposals 

With new topk code, it converges!
 
### ROI Align box

Converges


### ROI Align mask

Converges. Maybe 0.3% off on small objects. Probably deserves a repeat later


### Sample targets

Fixed issue where randomness was disabled. 
Running on Node 1

### RPN loss

Converges.



### Mask crop and resize

Converges.


### Fast RCNN Outputs

Converges


### Fast RCNN Losses (includes FastRCNN.__init__)

Converges

### Mask Loss

Converges


### To Add

* fastrcnn_head_func (the head_func codepaths were unchanged I believe so maybe we don't need to do them)
* maskrcnn_head_func

* *Eval codepath* (needs to be further broken up, but we don't need it yet)

### Not worried about 

We're not worried about testing these in isolation due to a combination of Can's work and the fact that some are implicitly tested by other pieces. 

* Input pipeline
* FPN backbone
* get_all_anchors
* RPNAnchors + slice_feature_and_anchors
* rpn_head
* BoxProposals