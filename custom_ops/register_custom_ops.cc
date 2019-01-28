#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status NMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks.
  ShapeHandle boxes;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
  ShapeHandle scores;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
  ShapeHandle max_output_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
  ShapeHandle iou_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
  ShapeHandle score_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
  // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
  DimensionHandle unused;
  // The boxes[0] and scores[0] are both num_boxes.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // The boxes[1] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

  c->set_output(0, c->Vector(c->UnknownDim()));
  return Status::OK();
}

} // namespace

REGISTER_OP("NonMaxSuppressionCustom")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: float")
    .Input("score_threshold: float")
    .Output("selected_indices: int32")
    .Attr("T: {half, float} = DT_FLOAT")
    .SetShapeFn(NMSShapeFn);

}  // namespace tensorflow

