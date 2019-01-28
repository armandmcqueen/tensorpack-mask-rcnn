#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "non_max_suppression_custom_op.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


#endif // GOOGLE_CUDA
