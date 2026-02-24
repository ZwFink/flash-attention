// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh

#pragma once
#if __has_include(<ATen/cuda/detail/UnpackRaw.cuh>)
#include <ATen/cuda/detail/UnpackRaw.cuh>
#elif __has_include("../../../../../pytorch/torch/include/ATen/cuda/detail/UnpackRaw.cuh")
#include "../../../../../pytorch/torch/include/ATen/cuda/detail/UnpackRaw.cuh"
#else
#error "Unable to locate ATen/cuda/detail/UnpackRaw.cuh"
#endif
