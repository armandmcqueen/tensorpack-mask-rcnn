# DockerHub unaltered mirror of AWS Deep Learning Container
FROM armandmcqueen/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04

RUN apt-get install less

# Need to reinstall some libraries the DL container provides due to custom Tensorflow binary
RUN pip uninstall -y tensorflow tensorboard tensorflow-estimator keras h5py horovod numpy

# Download and install custom Tensorflow binary
RUN wget https://github.com/armandmcqueen/tensorpack-mask-rcnn/releases/download/v0.0.0-WIP/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
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

RUN git clone https://github.com/armandmcqueen/tensorpack-mask-rcnn -b $BRANCH_NAME

RUN chmod -R +w /tensorpack-mask-rcnn
RUN pip install --ignore-installed -e /tensorpack-mask-rcnn/

ENV USE_CUDA_PATH /usr/local/cuda:/usr/local/cudnn/lib64
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:$LIBRARY_PATH

ENV BYTEPS_BASE_PATH /usr/local
ENV BYTEPS_PATH $BYTEPS_BASE_PATH/byteps
ENV BYTEPS_GIT_LINK https://github.com/bytedance/byteps

ARG CUDNN_VERSION=7.4.1.5-1+cuda10.0
RUN apt-get update &&\
    apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libopencv-dev \
    libjemalloc-dev \
    graphviz \
    libjpeg-dev \
    libpng-dev \
    iftop \
    lsb-release \
    libcudnn7=${CUDNN_VERSION} \
    libnuma-dev \
    gcc-4.9 \
    g++-4.9 \
    gcc-4.9-base

WORKDIR /root/

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cudnn/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig

RUN ln -sf /usr/local/cudnn/include/cudnn.h /usr/local/cuda/include/ && \
    ln -sf /usr/local/cudnn/lib64/libcudnn* /usr/local/cuda/lib64 &&\
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recurse-submodules $BYTEPS_GIT_LINK

# Pin GCC to 4.9 (priority 200) to compile correctly against TensorFlow, PyTorch, and MXNet.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200


# Install BytePS
ARG BYTEPS_NCCL_LINK=shared
RUN cd $BYTEPS_PATH &&\
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_MXNET=1 python setup.py install &&\
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_MXNET=1 python setup.py bdist_wheel

# Remove GCC pinning
RUN update-alternatives --remove gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove g++ /usr/bin/g++-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9

RUN rm -rf /usr/local/cuda/lib64/libcuda.so && \
    rm -rf /usr/local/cuda/lib64/libcuda.so.1
