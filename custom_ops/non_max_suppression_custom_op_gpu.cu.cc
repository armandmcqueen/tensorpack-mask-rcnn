
#define EIGEN_USE_GPU

#include "non_max_suppression_custom_op.h"

namespace tensorflow {

namespace functor {

template <typename T>
void NonMaxSuppressionCustomFunctor<Eigen::GpuDevice, T>::operator()(
    const Eigen::GpuDevice& d,
    const T* boxes,  // typename TTypes<T, 2>::Tensor boxes
    const T* scores, // typename TTypes<T, 1>::Tensor scores,
    int max_output_size,
    float iou_threshold,
    float score_threshold,
    std::vector<int>& selected) { // typename TTypes<int, 1>::Tensor selected_indices

}

} // namespace functor

} // namspace tensorflow


