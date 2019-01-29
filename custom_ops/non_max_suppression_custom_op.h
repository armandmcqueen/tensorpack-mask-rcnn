#ifndef NON_MAX_SUPPRESSION_CUSTOM_OP_H_
#define NON_MAX_SUPPRESSION_CUSTOM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

using std::vector;

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct NonMaxSuppressionCustomFunctor {
  void operator()(const Device& d,
                  const T* boxes,  // typename TTypes<T, 2>::Tensor boxes
                  const T* scores, // typename TTypes<T, 1>::Tensor scores,
                  int num_boxes,
                  int max_output_size,
                  float iou_threshold,
                  float score_threshold,
                  std::vector<int>& selected_indices); //typename TTypes<int, 1>::Tensor selected_indices);
};

template <typename T>
struct NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const T* boxes,  // typename TTypes<T, 2>::Tensor boxes
                  const T* scores, // typename TTypes<T, 1>::Tensor scores,
                  int num_boxes,
                  int max_output_size,
                  float iou_threshold,
                  float score_threshold,
                  std::vector<int>& selected_indices); // {  //typename TTypes<int, 1>::Tensor selected_indices);
};

template struct NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, float>;
template struct NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, Eigen::half>;

} // functor

} // tensorflow

#endif // NON_MAX_SUPPRESSION_CUSTOM_OP_H_
