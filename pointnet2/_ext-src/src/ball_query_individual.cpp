#include "ball_query_individual.h"
#include "utils.h"

void query_ball_point_kernel_wrapper_individual(int b, int n, int m, const float* radius, int nsample, const float *new_xyz, const float *xyz, int *idx);

at::Tensor ball_query_individual(at::Tensor new_xyz, at::Tensor xyz, at::Tensor radius, const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(radius);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(radius);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_kernel_wrapper_individual(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius.data<float>(), nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return idx;
}
