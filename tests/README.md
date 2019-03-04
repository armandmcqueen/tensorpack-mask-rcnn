# Running Unit Tests

To run unit tests, download the test data:

```
cd tests
bash download_test_data.sh
```

From the base directory `tensorpack-mask-rcnn`, run unit tests


By class
```
CUDA_VISIBLE_DEVICES=1 python3 -m unittest tests.nobatch_rpn_loss_test
```

By specific test case
```
CUDA_VISIBLE_DEVICES=1 python3 -m unittest tests.nobatch_rpn_loss_test.NoBatchMultilevelRPNLossTest.testMultilevelRpnLoss2
```

ALL test cases
```
CUDA_VISIBLE_DEVICES=1 python3 -m unittest discover -s tests -p '*_test.py'
```
  
