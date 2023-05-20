#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API histogramdd {
  using schema = ::std::tuple<at::Tensor,::std::vector<at::Tensor>> (const at::Tensor &, at::IntArrayRef, c10::optional<at::ArrayRef<double>>, const c10::optional<at::Tensor> &, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::histogramdd")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)")
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> call(const at::Tensor & self, at::IntArrayRef bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
};

struct TORCH_API histogramdd_int_bins {
  using schema = ::std::tuple<at::Tensor,::std::vector<at::Tensor>> (const at::Tensor &, int64_t, c10::optional<at::ArrayRef<double>>, const c10::optional<at::Tensor> &, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::histogramdd")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "int_bins")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "histogramdd.int_bins(Tensor self, int bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)")
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> call(const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
};

struct TORCH_API histogramdd_TensorList_bins {
  using schema = ::std::tuple<at::Tensor,::std::vector<at::Tensor>> (const at::Tensor &, at::TensorList, c10::optional<at::ArrayRef<double>>, const c10::optional<at::Tensor> &, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::histogramdd")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "TensorList_bins")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "histogramdd.TensorList_bins(Tensor self, Tensor[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)")
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> call(const at::Tensor & self, at::TensorList bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
  static ::std::tuple<at::Tensor,::std::vector<at::Tensor>> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::TensorList bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density);
};

}} // namespace at::_ops
