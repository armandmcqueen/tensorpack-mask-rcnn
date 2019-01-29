
#define EIGEN_USE_GPU

#include "non_max_suppression_custom_op.h"

namespace tensorflow {

// Helper data structure used locally
struct
#ifndef __HIP_PLATFORM_HCC__
    __align__(16)
#endif
Box {
    float x1, y1, x2, y2;
};

#define BOXES_PER_THREAD (8 * sizeof(int))
#define CHUNK_SIZE 2000

#define CAFFE_CUDA_NUM_THREADS         128
#define CAFFE_CUDA_NUM_THREADS_2D_DIMX 16
#define CAFFE_CUDA_NUM_THREADS_2D_DIMY 16

const dim3 CAFFE_CUDA_NUM_THREADS_2D = {
  static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMX),
  static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMY),
  1u
};

__launch_bounds__(
    CAFFE_CUDA_NUM_THREADS_2D_DIMX* CAFFE_CUDA_NUM_THREADS_2D_DIMY,
    4) __global__
    void NMSKernel(
        const Box* d_desc_sorted_boxes,
        const int nboxes,
        const float thresh) {
        //const int mask_ld,
        //int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory
  __shared__ Box shared_i_boxes[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // Same thing with areas
  __shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // The condition of the for loop is common to all threads in the block
  // This is necessary to be able to call __syncthreads() inside of the loop
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i_to_load = i_block_offset + threadIdx.x;
    if (i_to_load < nboxes) {
      // One 1D line load the boxes for x-dimension
      if (threadIdx.y == 0) {
        const Box box = d_desc_sorted_boxes[i_to_load];
        shared_i_areas[threadIdx.x] =
            (box.x2 - box.x1 + 1.0f) * (box.y2 - box.y1 + 1.0f);
        shared_i_boxes[threadIdx.x] = box;
      }
    }
    __syncthreads();
    const int i = i_block_offset + threadIdx.x;
    for (int j_thread_offset =
             BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < nboxes;
         j_thread_offset += BOXES_PER_THREAD * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold
      int above_thresh = 0;
      bool valid = false;
      for (int ib = 0; ib < BOXES_PER_THREAD; ++ib) {
        // This thread will compare Box i and Box j
        const int j = j_thread_offset + ib;
        if (i < j && i < nboxes && j < nboxes) {
          valid = true;
          const Box j_box = d_desc_sorted_boxes[j];
          const Box i_box = shared_i_boxes[threadIdx.x];
          const float j_area =
              (j_box.x2 - j_box.x1 + 1.0f) * (j_box.y2 - j_box.y1 + 1.0f);
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f)
            continue;
          const float xx1 = fmaxf(i_box.x1, j_box.x1);
          const float yy1 = fmaxf(i_box.y1, j_box.y1);
          const float xx2 = fminf(i_box.x2, j_box.x2);
          const float yy2 = fminf(i_box.y2, j_box.y2);

          // fdimf computes the positive difference between xx2+1 and xx1
          const float w = fdimf(xx2 + 1.0f, xx1);
          const float h = fdimf(yy2 + 1.0f, yy1);
          const float intersection = w * h;

          // Testing for a/b > t
          // eq with a > b*t (b is !=0)
          // avoiding divisions
          const float a = intersection;
          const float b = i_area + j_area - intersection;
          const float bt = b * thresh;
          // eq. to if ovr > thresh
          if (a > bt) {
            // we have score[j] <= score[i]
            above_thresh |= (1U << ib);
          }
        }
      }
      /*
      if (valid)
        d_delete_mask[i * mask_ld + j_thread_offset / BOXES_PER_THREAD] =
            above_thresh;
      */
    }
    __syncthreads(); // making sure everyone is done reading smem
  }
}

namespace functor {

template <typename T>
void NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, T>::operator()(
    const Eigen::GpuDevice& d,
    const T* boxes,  // typename TTypes<T, 2>::Tensor boxes
    const T* scores, // typename TTypes<T, 1>::Tensor scores,
    int num_boxes,
    int max_output_size,
    float iou_threshold,
    float score_threshold,
    std::vector<int>& selected) { // typename TTypes<int, 1>::Tensor selected_indices

}

} // namespace functor

} // namspace tensorflow


