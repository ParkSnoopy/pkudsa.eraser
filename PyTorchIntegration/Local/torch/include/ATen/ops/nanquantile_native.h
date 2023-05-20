#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor nanquantile(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim=c10::nullopt, bool keepdim=false, c10::string_view interpolation="linear");
TORCH_API at::Tensor & nanquantile_out(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
TORCH_API at::Tensor nanquantile(const at::Tensor & self, double q, c10::optional<int64_t> dim=c10::nullopt, bool keepdim=false, c10::string_view interpolation="linear");
TORCH_API at::Tensor & nanquantile_out(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
} // namespace native
} // namespace at
