import numpy as np

npz_path = "fastrcnn_loss_test_data/nobatch/DumpTensor-1.npz"


tensor_dict = np.load(npz_path)

for k, v in tensor_dict.items():
    print(k, v.shape)
