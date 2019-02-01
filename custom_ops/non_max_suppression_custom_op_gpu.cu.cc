/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

#define EIGEN_USE_GPU

#include <stdio.h>
#include "non_max_suppression_custom_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline T devIoU(T const * const a, T const * const b) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  T interS = width * height;
  T Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  T Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}


template <typename T>
__global__ void nms_kernel(const int n_boxes, 
                           const float nms_overlap_thresh,
                           const T *dev_boxes, 
                           unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  //printf("Block : {%d, %d, %d} threads. ThreadId = %d\n",
         //blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
  //printf("Block: %d %d %d Thread: %d %d %d\n", 
         //blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

  // if (row_start > col_start) return;
  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    /*
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
    */
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
void nms_cuda(const T* boxes_dev,
              const T* scores, 
              int boxes_num, 
              int max_output_size, 
              float nms_overlap_thresh, 
              float score_threshold, 
              std::vector<int>& selected) {
  using scalar_t = float;
  // AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  // auto scores = boxes.select(1, 4);
  // auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  // auto boxes_sorted = boxes.index_select(0, order_t);

  // int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  // scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  // THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  // THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  // mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));
  cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  cudaMemcpy(&mask_host[0], 
             mask_dev, 
             sizeof(unsigned long long) * boxes_num * col_blocks, 
             cudaMemcpyDeviceToHost);

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  // at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  // int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      // keep_out[num_to_keep++] = i;
      selected[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  cudaFree(mask_dev);

  int keep_size = num_to_keep <= max_output_size ? num_to_keep : max_output_size;
  selected.resize(keep_size);

  // return std::get<0>(order_t.index({keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)}).sort(0, false));
}

namespace functor {

template <typename T>
void NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* context,
    const Eigen::GpuDevice& d,
    const T* boxes,  // typename TTypes<T, 2>::Tensor boxes
    const T* scores, // typename TTypes<T, 1>::Tensor scores,
    int num_boxes,
    int max_output_size,
    float iou_threshold,
    float score_threshold) { 

  std::vector<int> selected(num_boxes);
  
  // std::cout << selected[0] << std::endl;
  // std::cout << selected[1] << std::endl;
  // std::cout << selected[2] << std::endl;
  // std::cout << selected.size() << std::endl;

  nms_cuda(boxes, scores, num_boxes, max_output_size,
           iou_threshold, score_threshold, selected);

  // Allocate output tensor
  Tensor* output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size())});
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_indices));
  cudaMemcpy(output_indices->flat<int>().data(), selected.data(), 
             selected.size() * sizeof(int), cudaMemcpyHostToDevice);
}

} // namespace functor

} // namspace tensorflow


