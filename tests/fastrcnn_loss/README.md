# Fast RCNN loss unit tests

Two unit tests. One uses tensors dumped from the nonbatch code, runs it through the batch code and compares output to dumped output tensors. The other uses tensors dumped from the batch code, runs it through the nonbatch code and compares outputs. Tensors from batch code lead aren't great for testing - the correct output for those tensors is `-inf`.

`CUDA_VISIBLE_DEVICES=1 python3 -m unittest tests.fastrcnn_loss.frcnn_loss_test_from_nobatch_tensors`
`CUDA_VISIBLE_DEVICES=1 python3 -m unittest tests.fastrcnn_loss.frcnn_loss_test_from_batch_tensors`

Must be run on GPU - small numeric issue leads to tests failing slightly when run on CPU.

TODO: Change asserts to assert almost equal.