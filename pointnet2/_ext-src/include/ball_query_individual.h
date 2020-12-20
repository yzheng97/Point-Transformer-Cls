#pragma once
#include <torch/extension.h>

at::Tensor ball_query_individual(at::Tensor new_xyz, at::Tensor xyz, at::Tensor radius, const int nsample);
