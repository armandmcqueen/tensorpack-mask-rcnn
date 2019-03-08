#!/bin/bash

aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/test_data/rpn_loss_test_data ./rpn_loss_test_data
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/test_data/fastrcnn_loss_test_data ./fastrcnn_loss_test_data
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/test_data/sample_targets ./sample_targets_test_data
