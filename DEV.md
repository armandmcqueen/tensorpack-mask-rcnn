# Development Notes

## Building Tensorflow

Requires custom Tensorflow for GPU optimized ops. Build steps were run on the AWS DLAMI 21.2.

```
source activate tensorflow_p36
pip uninstall -y tensorflow horovod

############################################################################################################
# Upgrade Bazel
############################################################################################################ 
rm /home/ubuntu/anaconda3/envs/tensorflow_p36/bin/bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-installer-linux-x86_64.sh
chmod +x bazel-0.19.2-installer-linux-x86_64.sh
./bazel-0.19.2-installer-linux-x86_64.sh --user


############################################################################################################
# Build TF 1.13 with CUDA 10
############################################################################################################

./configure

# XLA JIT: N
# CUDA: Y
# CUDA/CUDNN/NCCL dir: /usr/local/cuda-10.0
# CUDNN: 7.4.1
# NCCL: 2.3.7


############################################################################################################
# Create pip wheel
############################################################################################################

bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg
```


## Upgrading protoc to 3.6.1 for Horovod install

Required on DLAMI 21.2

```
pip uninstall -y protobuf

rm /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/bin/protoc
rm -r /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/include/google/protobuf
rm /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/lib/python3.6/site-packages/protobuf-3.6.0-py3.6-nspkg.pth
rm /home/ubuntu/anaconda3/bin//protoc

wget https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
mkdir -p /home/ubuntu/protoc
mv protoc-3.6.1-linux-x86_64.zip /home/ubuntu/protoc/protoc-3.6.1-linux-x86_64.zip
unzip /home/ubuntu/protoc/protoc-3.6.1-linux-x86_64.zip -d protoc
sudo mv /home/ubuntu/protoc/bin/protoc /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/bin/protoc
sudo mv /home/ubuntu/protoc/include/* /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/include
pip install protobuf==3.6.1
```