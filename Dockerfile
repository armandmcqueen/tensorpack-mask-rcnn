# DockerHub unaltered mirror of AWS Deep Learning Container
FROM armandmcqueen/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04

RUN apt-get install less

# Need to reinstall some libraries the DL container provides due to custom Tensorflow binary
RUN pip uninstall -y tensorflow tensorboard tensorflow-estimator keras h5py horovod numpy

# Download and install custom Tensorflow binary
RUN wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0-WIP/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-estimator==1.13.0 && \
    rm tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl

RUN pip install keras h5py

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1  pip install --no-cache-dir horovod==0.15.2 && \
    ldconfig


# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config


RUN pip install Cython
RUN pip install ujson opencv-python pycocotools matplotlib
RUN pip install --ignore-installed numpy==1.16.2


# TODO: Do I really need this now that we are using the DL container?
ARG CACHEBUST=1
ARG BRANCH_NAME

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow -b $BRANCH_NAME

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install --ignore-installed -e /mask-rcnn-tensorflow/

RUN apt-get update &&\
    apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends \
    gcc-4.9 \
    g++-4.9 \
    gcc-4.9-base


# Pin GCC to 4.9 (priority 200) to compile correctly against TensorFlow, PyTorch, and MXNet.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

RUN git clone --recurse-submodules https://github.com/bytedance/byteps
    cd byteps
    python setup.py install

# Remove GCC pinning
RUN update-alternatives --remove gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove g++ /usr/bin/g++-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9
