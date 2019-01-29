#define EIGEN_USE_THREADS

#include "non_max_suppression_custom_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}

class NonMaxSuppressionV3V4CustomBase : public OpKernel {
 public:
  explicit NonMaxSuppressionV3V4CustomBase(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    boxes_ = context->input(0);
    // scores: [num_boxes]
    scores_ = context->input(1);
    // max_output_size: scalar
    max_output_size_ = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size_.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size_.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    iou_threshold_val_ = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val_ >= 0 && iou_threshold_val_ <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));

    score_threshold_val_ = score_threshold.scalar<float>()();

    num_boxes_ = 0;
    ParseAndCheckBoxSizes(context, boxes_, &num_boxes_);
    CheckScoreSizes(context, num_boxes_, scores_);
    if (!context->status().ok()) {
      return;
    }

    DoComputeAndPostProcess(context);
  }

 protected:
  virtual void DoComputeAndPostProcess(OpKernelContext* context) = 0;

  Tensor boxes_;
  Tensor scores_;
  Tensor max_output_size_;
  int num_boxes_;
  float iou_threshold_val_;
  float score_threshold_val_;
};

template <typename Device, typename T>
class NonMaxSuppressionCustomOp : public NonMaxSuppressionV3V4CustomBase {
 public:
  explicit NonMaxSuppressionCustomOp(OpKernelConstruction* context)
      : NonMaxSuppressionV3V4CustomBase(context) {}

 protected:
  void DoComputeAndPostProcess(OpKernelContext* context) override {
    int max_output_size_val_ = max_output_size_.scalar<int>()();
    std::vector<int> selected(max_output_size_val_);

    // Allocate output tensors
    Tensor* output_indices = nullptr;
    TensorShape output_shape({static_cast<int>(max_output_size_val_)});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_indices));

    functor::NonMaxSuppressionCustomFunctor<Device, T> func;

    func(
      context->eigen_device<Device>(),
      boxes_.flat<T>().data(),
      scores_.flat<T>().data(),
      max_output_size_val_,
      iou_threshold_val_,
      score_threshold_val_,
      selected
    );

  }
};

//extern template struct functor::NonMaxSuppressionCustomFunctor<GPUDevice, float>;

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionCustom")
                        .Device(DEVICE_GPU)
                        .HostMemory("max_output_size")
                        .HostMemory("iou_threshold")
                        .HostMemory("score_threshold")
                        .TypeConstraint<float>("T"),
                        NonMaxSuppressionCustomOp<GPUDevice, float>);

} // namespace tensorflow

