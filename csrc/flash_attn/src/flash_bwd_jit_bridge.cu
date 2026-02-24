/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_bwd_jit_bridge.h"

#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#include <proteus/CppJitModule.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"
#include "hardware_info.h"
#include "kernel_traits.h"

#ifndef FLASH_ATTN_SOURCE_DIR_RAW
#define FLASH_ATTN_SOURCE_DIR_RAW .
#endif
#define FLASH_ATTN_STR_IMPL(x) #x
#define FLASH_ATTN_STR(x) FLASH_ATTN_STR_IMPL(x)

namespace FLASH_NAMESPACE {
namespace {

struct BwdKernelConfig {
  const char *KernelTraitsName;
  int HeadDim;
  int BlockM;
  int BlockN;
  int NThreads;
  size_t SmemSize1colblock;  // main kernel smem
  size_t SmemdQSize;         // convert_dq kernel smem
};

template <int HeadDim, int BlockM, int BlockN, int NWarps,
          int AtomLayoutMSdP, int AtomLayoutNdKV, int AtomLayoutMdQ,
          bool Is_V_in_regs, bool No_double_buffer>
BwdKernelConfig make_bwd_fp16_kernel_config(const char *name) {
  using KT = Flash_bwd_kernel_traits<HeadDim, BlockM, BlockN, NWarps,
      AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
      Is_V_in_regs, No_double_buffer, cutlass::half_t>;
  return {name, HeadDim, KT::kBlockM, KT::kBlockN, KT::kNThreads,
          KT::kSmemSize1colblock, KT::kSmemdQSize};
}

template <int HeadDim, int BlockM, int BlockN, int NWarps,
          int AtomLayoutMSdP, int AtomLayoutNdKV, int AtomLayoutMdQ,
          bool Is_V_in_regs, bool No_double_buffer>
BwdKernelConfig make_bwd_bf16_kernel_config(const char *name) {
  using KT = Flash_bwd_kernel_traits<HeadDim, BlockM, BlockN, NWarps,
      AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
      Is_V_in_regs, No_double_buffer, cutlass::bfloat16_t>;
  return {name, HeadDim, KT::kBlockM, KT::kBlockN, KT::kNThreads,
          KT::kSmemSize1colblock, KT::kSmemdQSize};
}

// --- FP16 backward kernel configs ---
// hdim32: !dropout or smem<104KB => V_in_regs=true
const BwdKernelConfig kBwdH32_Vreg_FP16 =
    make_bwd_fp16_kernel_config<32, 128, 128, 8, 4, 4, 4, true, false>(
        "fa2_bwd_h32_b128_n128_vreg_fp16");
// hdim32: dropout & smem>=104KB => V_in_regs=false
const BwdKernelConfig kBwdH32_NoVreg_FP16 =
    make_bwd_fp16_kernel_config<32, 128, 128, 8, 4, 4, 4, false, false>(
        "fa2_bwd_h32_b128_n128_novreg_fp16");
// hdim64: smem>=144KB
const BwdKernelConfig kBwdH64_Big_FP16 =
    make_bwd_fp16_kernel_config<64, 128, 128, 8, 4, 4, 4, false, false>(
        "fa2_bwd_h64_b128_n128_fp16");
// hdim64: smem<144KB
const BwdKernelConfig kBwdH64_Small_FP16 =
    make_bwd_fp16_kernel_config<64, 64, 128, 8, 2, 4, 4, true, false>(
        "fa2_bwd_h64_b64_n128_vreg_fp16");
// hdim96: !dropout or smem<116KB => V_in_regs=true
const BwdKernelConfig kBwdH96_Vreg_FP16 =
    make_bwd_fp16_kernel_config<96, 64, 128, 8, 2, 4, 4, true, false>(
        "fa2_bwd_h96_b64_n128_vreg_fp16");
// hdim96: dropout & smem>=116KB => V_in_regs=false
const BwdKernelConfig kBwdH96_NoVreg_FP16 =
    make_bwd_fp16_kernel_config<96, 64, 128, 8, 2, 4, 4, false, false>(
        "fa2_bwd_h96_b64_n128_novreg_fp16");
// hdim128: smem>=144KB
const BwdKernelConfig kBwdH128_Big_FP16 =
    make_bwd_fp16_kernel_config<128, 64, 128, 8, 2, 4, 2, false, false>(
        "fa2_bwd_h128_b64_n128_fp16");
// hdim128: smem<144KB
const BwdKernelConfig kBwdH128_Small_FP16 =
    make_bwd_fp16_kernel_config<128, 64, 64, 8, 4, 2, 2, true, false>(
        "fa2_bwd_h128_b64_n64_vreg_fp16");
// hdim192: smem>=136KB
const BwdKernelConfig kBwdH192_Big_FP16 =
    make_bwd_fp16_kernel_config<192, 64, 64, 8, 4, 2, 2, false, false>(
        "fa2_bwd_h192_b64_n64_fp16");
// hdim192: smem<136KB
const BwdKernelConfig kBwdH192_Small_FP16 =
    make_bwd_fp16_kernel_config<192, 64, 64, 8, 4, 2, 2, true, true>(
        "fa2_bwd_h192_b64_n64_vreg_nobuf_fp16");
// hdim256: smem>=176KB (H100)
const BwdKernelConfig kBwdH256_Big_FP16 =
    make_bwd_fp16_kernel_config<256, 64, 64, 8, 4, 2, 2, false, false>(
        "fa2_bwd_h256_b64_n64_fp16");
// hdim256: 144KB<=smem<176KB (A100)
const BwdKernelConfig kBwdH256_Mid_FP16 =
    make_bwd_fp16_kernel_config<256, 64, 64, 8, 4, 2, 2, false, true>(
        "fa2_bwd_h256_b64_n64_nobuf_fp16");
// hdim256: smem<144KB, !dropout only
const BwdKernelConfig kBwdH256_Small_FP16 =
    make_bwd_fp16_kernel_config<256, 64, 32, 8, 4, 1, 2, true, true>(
        "fa2_bwd_h256_b64_n32_vreg_nobuf_fp16");

// --- BF16 backward kernel configs ---
const BwdKernelConfig kBwdH32_Vreg_BF16 =
    make_bwd_bf16_kernel_config<32, 128, 128, 8, 4, 4, 4, true, false>(
        "fa2_bwd_h32_b128_n128_vreg_bf16");
const BwdKernelConfig kBwdH32_NoVreg_BF16 =
    make_bwd_bf16_kernel_config<32, 128, 128, 8, 4, 4, 4, false, false>(
        "fa2_bwd_h32_b128_n128_novreg_bf16");
const BwdKernelConfig kBwdH64_Big_BF16 =
    make_bwd_bf16_kernel_config<64, 128, 128, 8, 4, 4, 4, false, false>(
        "fa2_bwd_h64_b128_n128_bf16");
const BwdKernelConfig kBwdH64_Small_BF16 =
    make_bwd_bf16_kernel_config<64, 64, 128, 8, 2, 4, 4, true, false>(
        "fa2_bwd_h64_b64_n128_vreg_bf16");
const BwdKernelConfig kBwdH96_Vreg_BF16 =
    make_bwd_bf16_kernel_config<96, 64, 128, 8, 2, 4, 4, true, false>(
        "fa2_bwd_h96_b64_n128_vreg_bf16");
const BwdKernelConfig kBwdH96_NoVreg_BF16 =
    make_bwd_bf16_kernel_config<96, 64, 128, 8, 2, 4, 4, false, false>(
        "fa2_bwd_h96_b64_n128_novreg_bf16");
const BwdKernelConfig kBwdH128_Big_BF16 =
    make_bwd_bf16_kernel_config<128, 64, 128, 8, 2, 4, 2, false, false>(
        "fa2_bwd_h128_b64_n128_bf16");
const BwdKernelConfig kBwdH128_Small_BF16 =
    make_bwd_bf16_kernel_config<128, 64, 64, 8, 4, 2, 2, true, false>(
        "fa2_bwd_h128_b64_n64_vreg_bf16");
const BwdKernelConfig kBwdH192_Big_BF16 =
    make_bwd_bf16_kernel_config<192, 64, 64, 8, 4, 2, 2, false, false>(
        "fa2_bwd_h192_b64_n64_bf16");
const BwdKernelConfig kBwdH192_Small_BF16 =
    make_bwd_bf16_kernel_config<192, 64, 64, 8, 4, 2, 2, true, true>(
        "fa2_bwd_h192_b64_n64_vreg_nobuf_bf16");
const BwdKernelConfig kBwdH256_Big_BF16 =
    make_bwd_bf16_kernel_config<256, 64, 64, 8, 4, 2, 2, false, false>(
        "fa2_bwd_h256_b64_n64_bf16");
const BwdKernelConfig kBwdH256_Mid_BF16 =
    make_bwd_bf16_kernel_config<256, 64, 64, 8, 4, 2, 2, false, true>(
        "fa2_bwd_h256_b64_n64_nobuf_bf16");
const BwdKernelConfig kBwdH256_Small_BF16 =
    make_bwd_bf16_kernel_config<256, 64, 32, 8, 4, 1, 2, true, true>(
        "fa2_bwd_h256_b64_n32_vreg_nobuf_bf16");

// ---------------------------------------------------------------------------
// Shared helpers (duplicated from forward bridge to keep the two bridges
// independent compilation units).
// ---------------------------------------------------------------------------

std::string normalize_macro_path(std::string Path) {
  Path.erase(
      std::remove_if(Path.begin(), Path.end(),
                     [](unsigned char c) { return std::isspace(c); }),
      Path.end());
  if (Path.size() >= 2 && Path.front() == '"' && Path.back() == '"') {
    Path = Path.substr(1, Path.size() - 2);
  }
  return Path;
}

std::string get_source_root() {
  return normalize_macro_path(FLASH_ATTN_STR(FLASH_ATTN_SOURCE_DIR_RAW));
}

std::string get_torch_include_dir(const std::string &SourceRoot) {
  if (const char *EnvVal = std::getenv("FLASH_ATTN_JIT_TORCH_INCLUDE")) {
    return EnvVal;
  }
  return SourceRoot + "/../../pytorch/torch/include";
}

std::string get_torch_api_include_dir(const std::string &TorchIncludeDir) {
  if (const char *EnvVal = std::getenv("FLASH_ATTN_JIT_TORCH_API_INCLUDE")) {
    return EnvVal;
  }
  return TorchIncludeDir + "/torch/csrc/api/include";
}

std::string get_torch_aten_include_dir(const std::string &SourceRoot) {
  if (const char *EnvVal = std::getenv("FLASH_ATTN_JIT_TORCH_ATEN_INCLUDE")) {
    return EnvVal;
  }
  return SourceRoot + "/../../pytorch/aten/src";
}

std::string get_cuda_include_dir() {
  if (const char *EnvVal = std::getenv("FLASH_ATTN_JIT_CUDA_INCLUDE")) {
    return EnvVal;
  }
  if (const char *CudaHome = std::getenv("CUDA_HOME")) {
    return std::string(CudaHome) + "/include";
  }
  return "/usr/tce/packages/cuda/cuda-12.2.2/include";
}

bool use_proteus_jit() {
  static const bool Enabled = []() {
    const char *EnvVal = std::getenv("FLASH_ATTN_USE_PROTEUS_JIT");
    return EnvVal != nullptr && std::strcmp(EnvVal, "1") == 0;
  }();
  return Enabled;
}

std::vector<std::string> get_extra_args() {
  const std::string SourceRoot = get_source_root();
  const std::string TorchInclude = get_torch_include_dir(SourceRoot);
  const std::string TorchApiInclude = get_torch_api_include_dir(TorchInclude);
  const std::string TorchAtenInclude = get_torch_aten_include_dir(SourceRoot);
  const std::string CudaInclude = get_cuda_include_dir();
  std::vector<std::string> Args = {
      "-I" + SourceRoot + "/csrc/flash_attn/src",
      "-I" + SourceRoot + "/csrc/cutlass/include",
      "-I" + TorchInclude,
      "-I" + TorchApiInclude,
      "-I" + TorchAtenInclude,
      "-I" + CudaInclude,
  };
  if (const char *DebugArgs = std::getenv("FLASH_ATTN_JIT_DEBUG_ARGS");
      DebugArgs != nullptr && std::strcmp(DebugArgs, "1") == 0) {
    for (const auto &Arg : Args) {
      std::fprintf(stderr, "[flash-attn-bwd-jit] %s\n", Arg.c_str());
    }
  }
  return Args;
}

const char *to_template_bool(bool Value) { return Value ? "true" : "false"; }

int get_headdim_bucket(int HeadDim) {
  if (HeadDim <= 32) return 32;
  if (HeadDim <= 64) return 64;
  if (HeadDim <= 96) return 96;
  if (HeadDim <= 128) return 128;
  if (HeadDim <= 192) return 192;
  if (HeadDim <= 256) return 256;
  return 0;
}

// ---------------------------------------------------------------------------
// JIT kernel code string
// ---------------------------------------------------------------------------

std::string get_bwd_jit_kernel_code() {
  return R"code(
#include <tuple>
#include "namespace_config.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

using flash_bwd_params_t = FLASH_NAMESPACE::Flash_bwd_params;

// --- FP16 backward kernel traits ---
using fa2_bwd_h32_b128_n128_vreg_fp16 =
    ::Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4, 4, true, false, cutlass::half_t>;
using fa2_bwd_h32_b128_n128_novreg_fp16 =
    ::Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t>;
using fa2_bwd_h64_b128_n128_fp16 =
    ::Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t>;
using fa2_bwd_h64_b64_n128_vreg_fp16 =
    ::Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4, true, false, cutlass::half_t>;
using fa2_bwd_h96_b64_n128_vreg_fp16 =
    ::Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false, cutlass::half_t>;
using fa2_bwd_h96_b64_n128_novreg_fp16 =
    ::Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, false, false, cutlass::half_t>;
using fa2_bwd_h128_b64_n128_fp16 =
    ::Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t>;
using fa2_bwd_h128_b64_n64_vreg_fp16 =
    ::Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, true, false, cutlass::half_t>;
using fa2_bwd_h192_b64_n64_fp16 =
    ::Flash_bwd_kernel_traits<192, 64, 64, 8, 4, 2, 2, false, false, cutlass::half_t>;
using fa2_bwd_h192_b64_n64_vreg_nobuf_fp16 =
    ::Flash_bwd_kernel_traits<192, 64, 64, 8, 4, 2, 2, true, true, cutlass::half_t>;
using fa2_bwd_h256_b64_n64_fp16 =
    ::Flash_bwd_kernel_traits<256, 64, 64, 8, 4, 2, 2, false, false, cutlass::half_t>;
using fa2_bwd_h256_b64_n64_nobuf_fp16 =
    ::Flash_bwd_kernel_traits<256, 64, 64, 8, 4, 2, 2, false, true, cutlass::half_t>;
using fa2_bwd_h256_b64_n32_vreg_nobuf_fp16 =
    ::Flash_bwd_kernel_traits<256, 64, 32, 8, 4, 1, 2, true, true, cutlass::half_t>;

// --- BF16 backward kernel traits ---
using fa2_bwd_h32_b128_n128_vreg_bf16 =
    ::Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4, 4, true, false, cutlass::bfloat16_t>;
using fa2_bwd_h32_b128_n128_novreg_bf16 =
    ::Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4, 4, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h64_b128_n128_bf16 =
    ::Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h64_b64_n128_vreg_bf16 =
    ::Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4, true, false, cutlass::bfloat16_t>;
using fa2_bwd_h96_b64_n128_vreg_bf16 =
    ::Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false, cutlass::bfloat16_t>;
using fa2_bwd_h96_b64_n128_novreg_bf16 =
    ::Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h128_b64_n128_bf16 =
    ::Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h128_b64_n64_vreg_bf16 =
    ::Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, true, false, cutlass::bfloat16_t>;
using fa2_bwd_h192_b64_n64_bf16 =
    ::Flash_bwd_kernel_traits<192, 64, 64, 8, 4, 2, 2, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h192_b64_n64_vreg_nobuf_bf16 =
    ::Flash_bwd_kernel_traits<192, 64, 64, 8, 4, 2, 2, true, true, cutlass::bfloat16_t>;
using fa2_bwd_h256_b64_n64_bf16 =
    ::Flash_bwd_kernel_traits<256, 64, 64, 8, 4, 2, 2, false, false, cutlass::bfloat16_t>;
using fa2_bwd_h256_b64_n64_nobuf_bf16 =
    ::Flash_bwd_kernel_traits<256, 64, 64, 8, 4, 2, 2, false, true, cutlass::bfloat16_t>;
using fa2_bwd_h256_b64_n32_vreg_nobuf_bf16 =
    ::Flash_bwd_kernel_traits<256, 64, 32, 8, 4, 1, 2, true, true, cutlass::bfloat16_t>;

// --- Kernel 1: dot_do_o (preprocess) ---
template <bool Clear_dQaccum, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_jit(const flash_bwd_params_t params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_NAMESPACE::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
#else
    printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
#endif
}

// --- Kernel 2: main dQ/dK/dV backward kernel ---
template <typename Kernel_traits, bool Is_dropout, bool Is_causal,
          bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Is_softcap>
__global__ void flash_bwd_dq_dk_dv_jit(const flash_bwd_params_t params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(!(Is_causal && Is_local));
    FLASH_NAMESPACE::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout,
        Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
#else
    printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
#endif
}

// --- Kernel 3: convert dQ accumulator to output precision ---
template <typename Kernel_traits>
__global__ void flash_bwd_convert_dq_jit(const flash_bwd_params_t params,
                                         const int nsplits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_NAMESPACE::convert_dQ<Kernel_traits>(params, nsplits);
#else
    printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
#endif
}
)code";
}

// ---------------------------------------------------------------------------
// JIT module accessor
// ---------------------------------------------------------------------------

auto &get_bwd_module() {
  static const std::string JitKernelCode = get_bwd_jit_kernel_code();
  static auto Module = std::make_unique<proteus::CppJitModule>(
      "cuda", JitKernelCode, get_extra_args());
  return *Module;
}

// ---------------------------------------------------------------------------
// Config selection — mirrors run_mha_bwd_hdimXX logic
// ---------------------------------------------------------------------------

const BwdKernelConfig *select_bwd_fp16_kernel_config(
    const Flash_bwd_params &params) {
  const int HeadDimBucket = get_headdim_bucket(params.d);
  if (HeadDimBucket == 0) return nullptr;

  const bool Is_dropout = params.p_dropout < 1.f;

  int Device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&Device));
  int MaxSmemPerBlock = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &MaxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, Device));

  switch (HeadDimBucket) {
    case 32:
      if (MaxSmemPerBlock >= 104 * 1024) {
        return Is_dropout ? &kBwdH32_NoVreg_FP16 : &kBwdH32_Vreg_FP16;
      }
      return &kBwdH32_Vreg_FP16;
    case 64:
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH64_Big_FP16;
      }
      return &kBwdH64_Small_FP16;
    case 96:
      if (MaxSmemPerBlock >= 116 * 1024) {
        return Is_dropout ? &kBwdH96_NoVreg_FP16 : &kBwdH96_Vreg_FP16;
      }
      return &kBwdH96_Vreg_FP16;
    case 128:
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH128_Big_FP16;
      }
      return &kBwdH128_Small_FP16;
    case 192:
      if (MaxSmemPerBlock >= 136 * 1024) {
        return &kBwdH192_Big_FP16;
      }
      return &kBwdH192_Small_FP16;
    case 256:
      if (MaxSmemPerBlock >= 176 * 1024) {
        return &kBwdH256_Big_FP16;
      }
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH256_Mid_FP16;
      }
      // sm86/sm89: only works without dropout
      return Is_dropout ? nullptr : &kBwdH256_Small_FP16;
    default:
      return nullptr;
  }
}

const BwdKernelConfig *select_bwd_bf16_kernel_config(
    const Flash_bwd_params &params) {
  const int HeadDimBucket = get_headdim_bucket(params.d);
  if (HeadDimBucket == 0) return nullptr;

  const bool Is_dropout = params.p_dropout < 1.f;

  int Device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&Device));
  int MaxSmemPerBlock = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &MaxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, Device));

  switch (HeadDimBucket) {
    case 32:
      if (MaxSmemPerBlock >= 104 * 1024) {
        return Is_dropout ? &kBwdH32_NoVreg_BF16 : &kBwdH32_Vreg_BF16;
      }
      return &kBwdH32_Vreg_BF16;
    case 64:
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH64_Big_BF16;
      }
      return &kBwdH64_Small_BF16;
    case 96:
      if (MaxSmemPerBlock >= 116 * 1024) {
        return Is_dropout ? &kBwdH96_NoVreg_BF16 : &kBwdH96_Vreg_BF16;
      }
      return &kBwdH96_Vreg_BF16;
    case 128:
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH128_Big_BF16;
      }
      return &kBwdH128_Small_BF16;
    case 192:
      if (MaxSmemPerBlock >= 136 * 1024) {
        return &kBwdH192_Big_BF16;
      }
      return &kBwdH192_Small_BF16;
    case 256:
      if (MaxSmemPerBlock >= 176 * 1024) {
        return &kBwdH256_Big_BF16;
      }
      if (MaxSmemPerBlock >= 144 * 1024) {
        return &kBwdH256_Mid_BF16;
      }
      return Is_dropout ? nullptr : &kBwdH256_Small_BF16;
    default:
      return nullptr;
  }
}

bool is_eligible_bwd_fastpath(const Flash_bwd_params &params) {
  return get_headdim_bucket(params.d) != 0;
}

// ---------------------------------------------------------------------------
// Common backward JIT launch logic for both fp16 and bf16.
// ---------------------------------------------------------------------------

bool try_run_mha_bwd_jit_impl(Flash_bwd_params &params, cudaStream_t stream,
                               const BwdKernelConfig *Config) {
  if (Config == nullptr) return false;

  const bool Is_dropout_base = params.p_dropout < 1.f;
  const bool Is_causal = params.is_causal;
  const bool Is_local =
      (params.window_size_left >= 0 || params.window_size_right >= 0) &&
      !Is_causal;
  const bool Has_alibi = params.alibi_slopes_ptr != nullptr;
  const bool Is_softcap = params.softcap > 0.0f;

  const bool Is_even_MN_raw =
      params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
      params.seqlen_k % Config->BlockN == 0 &&
      params.seqlen_q % Config->BlockM == 0;
  const bool Is_even_K_raw = params.d == Config->HeadDim;

  // Match the template specialization from run_flash_bwd_seqk_parallel:
  //   Is_dropout && !Is_softcap
  //   Is_causal
  //   Is_local && !Is_causal
  //   Has_alibi
  //   IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && HeadDim<=128
  //   IsEvenKConst && !Has_alibi
  //   Is_softcap
  const bool Is_dropout_t = Is_dropout_base && !Is_softcap;
  const bool Is_local_t = Is_local && !Is_causal;
  const bool Is_even_MN_t = Is_even_MN_raw && Is_even_K_raw &&
                            !Is_local && !Has_alibi &&
                            Config->HeadDim <= 128;
  const bool Is_even_K_t = Is_even_K_raw && !Has_alibi;

  const int num_m_block =
      (params.seqlen_q + Config->BlockM - 1) / Config->BlockM;
  const int num_n_block =
      (params.seqlen_k + Config->BlockN - 1) / Config->BlockN;

  const int num_sm = get_num_sm(get_current_device());
  int gridDimx = num_n_block;
  if (params.deterministic) {
    gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
  }

  const dim3 GridM(num_m_block, params.b, params.h);
  const dim3 GridN(gridDimx, params.b, params.h);
  const dim3 Block(Config->NThreads, 1, 1);

  // -----------------------------------------------------------------------
  // Kernel 1: dot_do_o (preprocess) — smem = 0
  // -----------------------------------------------------------------------
  {
    auto Instance = get_bwd_module().instantiate(
        "flash_bwd_dot_do_o_jit",
        to_template_bool(!params.deterministic),  // Clear_dQaccum
        Config->KernelTraitsName);
    Instance.launch(GridM, Block, /*smem=*/0,
                    reinterpret_cast<void *>(stream), params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // -----------------------------------------------------------------------
  // Kernel 2: main dQ/dK/dV — smem = SmemSize1colblock
  // -----------------------------------------------------------------------
  {
    const size_t SmemSize = Config->SmemSize1colblock;
    auto Instance = get_bwd_module().instantiate(
        "flash_bwd_dq_dk_dv_jit", Config->KernelTraitsName,
        to_template_bool(Is_dropout_t), to_template_bool(Is_causal),
        to_template_bool(Is_local_t), to_template_bool(Has_alibi),
        to_template_bool(Is_even_MN_t), to_template_bool(Is_even_K_t),
        to_template_bool(Is_softcap));
    Instance.launch(GridN, Block, SmemSize,
                    reinterpret_cast<void *>(stream), params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // -----------------------------------------------------------------------
  // Kernel 3: convert_dQ — smem = SmemdQSize
  // -----------------------------------------------------------------------
  {
    const size_t SmemSize = Config->SmemdQSize;
    const int nsplits = !params.deterministic ? 1 : gridDimx;
    auto Instance = get_bwd_module().instantiate(
        "flash_bwd_convert_dq_jit", Config->KernelTraitsName);
    Instance.launch(GridM, Block, SmemSize,
                    reinterpret_cast<void *>(stream), params, nsplits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool try_run_mha_bwd_jit_fp16(Flash_bwd_params &params, cudaStream_t stream) {
  if (!use_proteus_jit() || !is_eligible_bwd_fastpath(params)) {
    return false;
  }
  const BwdKernelConfig *Config = select_bwd_fp16_kernel_config(params);
  auto t0 = std::chrono::steady_clock::now();
  bool result = try_run_mha_bwd_jit_impl(params, stream, Config);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::fprintf(stderr, "[proteus-jit] bwd_fp16 h=%d launch %ld ms\n",
               Config ? Config->HeadDim : params.d, ms);
  return result;
}

bool try_run_mha_bwd_jit_bf16(Flash_bwd_params &params, cudaStream_t stream) {
  if (!use_proteus_jit() || !is_eligible_bwd_fastpath(params)) {
    return false;
  }
  const BwdKernelConfig *Config = select_bwd_bf16_kernel_config(params);
  auto t0 = std::chrono::steady_clock::now();
  bool result = try_run_mha_bwd_jit_impl(params, stream, Config);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::fprintf(stderr, "[proteus-jit] bwd_bf16 h=%d launch %ld ms\n",
               Config ? Config->HeadDim : params.d, ms);
  return result;
}

}  // namespace FLASH_NAMESPACE
