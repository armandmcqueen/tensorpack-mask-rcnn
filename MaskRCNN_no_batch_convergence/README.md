# Convergence Testing

Add pieces of the batch code to the nobatch code 1-by-1. Each piece must have a flag that enables or disables it (hardcoded around line 55 of train.py). Before committing to master, we must test that the new pieces leads to convergence at target accuracy. 

## Pieces

### All disabled

Converges

**Important** - this was run when only 4 flags existed. As more flags are added, we increase the chance that we accidentally break the all_disabled codepath. We will need to repeat this experiment as we add more flags)

Repeating on Node 1.

### GenerateProposals 

Converges, but a touch low in accuracy on small objects/segmentations - 0.5% off (0.005 mAP). Probably fine. Repeating on Node 2.

 
### ROI Align box

Converges


### ROI Align mask

Running on Node 3.


### Sample targets

Running on Node 4.


### Mask crop and resize

Running on Node 5


### RPN loss

Running on Node 6


### To Add

* fastrcnn_head_func
* fastrcnn_outputs
* FastRCNNHead.__init__() + .losses()
* maskrcnn_head_func
* maskrcnn_loss

* *Eval codepath* (needs to be further broken up, but we don't need it yet)

### Not worried about 

We're not worried about testing these in isolation due to a combination of Can's work and the fact that some are implicitly tested by other pieces. 

* Input pipeline
* FPN backbone
* get_all_anchors
* RPNAnchors + slice_feature_and_anchors
* rpn_head
* BoxProposals