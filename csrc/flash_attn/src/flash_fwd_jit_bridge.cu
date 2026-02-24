/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_fwd_jit_bridge.h"

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

#include "flash_fwd_kernel.h"
#include "hardware_info.h"

#ifndef FLASH_ATTN_SOURCE_DIR_RAW
#define FLASH_ATTN_SOURCE_DIR_RAW .
#endif
#define FLASH_ATTN_STR_IMPL(x) #x
#define FLASH_ATTN_STR(x) FLASH_ATTN_STR_IMPL(x)

namespace FLASH_NAMESPACE {
namespace {

struct KernelConfig {
  const char *KernelTraitsName;
  int HeadDim;
  int BlockM;
  int BlockN;
  int NThreads;
  size_t SmemSize;
};

template <int HeadDim, int BlockM, int BlockN, int NWarps>
KernelConfig make_fp16_kernel_config(const char *KernelTraitsName) {
  using KernelTraits = Flash_fwd_kernel_traits<HeadDim, BlockM, BlockN, NWarps,
                                               false, false, cutlass::half_t>;
  return KernelConfig{KernelTraitsName, HeadDim, KernelTraits::kBlockM,
                      KernelTraits::kBlockN, KernelTraits::kNThreads,
                      KernelTraits::kSmemSize};
}

template <int HeadDim, int BlockM, int BlockN, int NWarps>
KernelConfig make_bf16_kernel_config(const char *KernelTraitsName) {
  using KernelTraits = Flash_fwd_kernel_traits<HeadDim, BlockM, BlockN, NWarps,
                                               false, false, cutlass::bfloat16_t>;
  return KernelConfig{KernelTraitsName, HeadDim, KernelTraits::kBlockM,
                      KernelTraits::kBlockN, KernelTraits::kNThreads,
                      KernelTraits::kSmemSize};
}

const KernelConfig kCfgH32B128N128W4 =
    make_fp16_kernel_config<32, 128, 128, 4>("fa2_traits_h32_b128_n128_w4_fp16");
const KernelConfig kCfgH64B128N128W4 =
    make_fp16_kernel_config<64, 128, 128, 4>("fa2_traits_h64_b128_n128_w4_fp16");
const KernelConfig kCfgH64B128N64W4 =
    make_fp16_kernel_config<64, 128, 64, 4>("fa2_traits_h64_b128_n64_w4_fp16");
const KernelConfig kCfgH96B128N64W4 =
    make_fp16_kernel_config<96, 128, 64, 4>("fa2_traits_h96_b128_n64_w4_fp16");
const KernelConfig kCfgH96B64N64W4 =
    make_fp16_kernel_config<96, 64, 64, 4>("fa2_traits_h96_b64_n64_w4_fp16");
const KernelConfig kCfgH128B128N32W4 =
    make_fp16_kernel_config<128, 128, 32, 4>("fa2_traits_h128_b128_n32_w4_fp16");
const KernelConfig kCfgH128B128N64W4 =
    make_fp16_kernel_config<128, 128, 64, 4>("fa2_traits_h128_b128_n64_w4_fp16");
const KernelConfig kCfgH128B64N64W4 =
    make_fp16_kernel_config<128, 64, 64, 4>("fa2_traits_h128_b64_n64_w4_fp16");
const KernelConfig kCfgH192B128N64W8 =
    make_fp16_kernel_config<192, 128, 64, 8>("fa2_traits_h192_b128_n64_w8_fp16");
const KernelConfig kCfgH192B64N64W4 =
    make_fp16_kernel_config<192, 64, 64, 4>("fa2_traits_h192_b64_n64_w4_fp16");
const KernelConfig kCfgH256B128N64W8 =
    make_fp16_kernel_config<256, 128, 64, 8>("fa2_traits_h256_b128_n64_w8_fp16");
const KernelConfig kCfgH256B64N64W4 =
    make_fp16_kernel_config<256, 64, 64, 4>("fa2_traits_h256_b64_n64_w4_fp16");

const KernelConfig kCfgH32B128N128W4_BF16 =
    make_bf16_kernel_config<32, 128, 128, 4>("fa2_traits_h32_b128_n128_w4_bf16");
const KernelConfig kCfgH64B128N128W4_BF16 =
    make_bf16_kernel_config<64, 128, 128, 4>("fa2_traits_h64_b128_n128_w4_bf16");
const KernelConfig kCfgH64B128N64W4_BF16 =
    make_bf16_kernel_config<64, 128, 64, 4>("fa2_traits_h64_b128_n64_w4_bf16");
const KernelConfig kCfgH96B128N64W4_BF16 =
    make_bf16_kernel_config<96, 128, 64, 4>("fa2_traits_h96_b128_n64_w4_bf16");
const KernelConfig kCfgH96B64N64W4_BF16 =
    make_bf16_kernel_config<96, 64, 64, 4>("fa2_traits_h96_b64_n64_w4_bf16");
const KernelConfig kCfgH128B128N32W4_BF16 =
    make_bf16_kernel_config<128, 128, 32, 4>("fa2_traits_h128_b128_n32_w4_bf16");
const KernelConfig kCfgH128B128N64W4_BF16 =
    make_bf16_kernel_config<128, 128, 64, 4>("fa2_traits_h128_b128_n64_w4_bf16");
const KernelConfig kCfgH128B64N64W4_BF16 =
    make_bf16_kernel_config<128, 64, 64, 4>("fa2_traits_h128_b64_n64_w4_bf16");
const KernelConfig kCfgH192B128N64W8_BF16 =
    make_bf16_kernel_config<192, 128, 64, 8>("fa2_traits_h192_b128_n64_w8_bf16");
const KernelConfig kCfgH192B64N64W4_BF16 =
    make_bf16_kernel_config<192, 64, 64, 4>("fa2_traits_h192_b64_n64_w4_bf16");
const KernelConfig kCfgH256B128N64W8_BF16 =
    make_bf16_kernel_config<256, 128, 64, 8>("fa2_traits_h256_b128_n64_w8_bf16");
const KernelConfig kCfgH256B64N64W4_BF16 =
    make_bf16_kernel_config<256, 64, 64, 4>("fa2_traits_h256_b64_n64_w4_bf16");

std::string normalize_macro_path(std::string Path) {
  // Macro stringification can insert spaces for unquoted -D path tokens.
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

std::string get_jit_kernel_code() {
  return R"code(
#include <tuple>
#include "namespace_config.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

using flash_fwd_params_t = FLASH_NAMESPACE::Flash_fwd_params;
using fa2_traits_h32_b128_n128_w4_fp16 =
    ::Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, cutlass::half_t>;
using fa2_traits_h64_b128_n128_w4_fp16 =
    ::Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t>;
using fa2_traits_h64_b128_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<64, 128, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h96_b128_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h96_b64_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<96, 64, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h128_b128_n32_w4_fp16 =
    ::Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, cutlass::half_t>;
using fa2_traits_h128_b128_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h128_b64_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h192_b128_n64_w8_fp16 =
    ::Flash_fwd_kernel_traits<192, 128, 64, 8, false, false, cutlass::half_t>;
using fa2_traits_h192_b64_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<192, 64, 64, 4, false, false, cutlass::half_t>;
using fa2_traits_h256_b128_n64_w8_fp16 =
    ::Flash_fwd_kernel_traits<256, 128, 64, 8, false, false, cutlass::half_t>;
using fa2_traits_h256_b64_n64_w4_fp16 =
    ::Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::half_t>;

using fa2_traits_h32_b128_n128_w4_bf16 =
    ::Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h64_b128_n128_w4_bf16 =
    ::Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h64_b128_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<64, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h96_b128_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h96_b64_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<96, 64, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h128_b128_n32_w4_bf16 =
    ::Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h128_b128_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h128_b64_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h192_b128_n64_w8_bf16 =
    ::Flash_fwd_kernel_traits<192, 128, 64, 8, false, false, cutlass::bfloat16_t>;
using fa2_traits_h192_b64_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<192, 64, 64, 4, false, false, cutlass::bfloat16_t>;
using fa2_traits_h256_b128_n64_w8_bf16 =
    ::Flash_fwd_kernel_traits<256, 128, 64, 8, false, false, cutlass::bfloat16_t>;
using fa2_traits_h256_b64_n64_w4_bf16 =
    ::Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template <typename KernelTraits, bool Is_dropout, bool Is_causal, bool Is_local,
          bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
          bool Return_softmax>
__global__ void flash_fwd_kernel_jit(const flash_fwd_params_t params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static_assert(!(Is_causal && Is_local));
  FLASH_NAMESPACE::compute_attn<KernelTraits, Is_dropout, Is_causal, Is_local,
                                Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
                                Return_softmax>(params);
#else
  printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
#endif
}
)code";
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
      std::fprintf(stderr, "[flash-attn-jit] %s\n", Arg.c_str());
    }
  }
  return Args;
}

auto &get_module() {
  static const std::string JitKernelCode = get_jit_kernel_code();
  static auto Module = std::make_unique<proteus::CppJitModule>(
      "cuda", JitKernelCode, get_extra_args());
  return *Module;
}

const char *to_template_bool(bool Value) { return Value ? "true" : "false"; }

int get_headdim_bucket(int HeadDim) {
  if (HeadDim <= 32) {
    return 32;
  }
  if (HeadDim <= 64) {
    return 64;
  }
  if (HeadDim <= 96) {
    return 96;
  }
  if (HeadDim <= 128) {
    return 128;
  }
  if (HeadDim <= 192) {
    return 192;
  }
  if (HeadDim <= 256) {
    return 256;
  }
  return 0;
}

const KernelConfig *select_fp16_kernel_config(const Flash_fwd_params &params) {
  const int HeadDimBucket = get_headdim_bucket(params.d);
  if (HeadDimBucket == 0) {
    return nullptr;
  }

  const bool Is_dropout = params.p_dropout < 1.f;
  const bool Is_causal = params.is_causal;

  switch (HeadDimBucket) {
    case 32:
      return &kCfgH32B128N128W4;
    case 64:
      return Is_dropout ? &kCfgH64B128N64W4 : &kCfgH64B128N128W4;
    case 96: {
      auto [CcMajor, CcMinor] =
          get_compute_capability(get_current_device());
      const bool IsSm8x = CcMajor == 8 && CcMinor > 0;
      if (IsSm8x && Is_causal) {
        return &kCfgH96B64N64W4;
      }
      return &kCfgH96B128N64W4;
    }
    case 128: {
      auto [CcMajor, CcMinor] =
          get_compute_capability(get_current_device());
      const bool IsSm8x = CcMajor == 8 && CcMinor > 0;
      if (!Is_dropout) {
        if (IsSm8x) {
          return Is_causal ? &kCfgH128B64N64W4 : &kCfgH128B128N32W4;
        }
        return &kCfgH128B128N64W4;
      }
      return &kCfgH128B128N32W4;
    }
    case 192:
      return Is_dropout ? &kCfgH192B64N64W4 : &kCfgH192B128N64W8;
    case 256: {
      int Device = 0;
      C10_CUDA_CHECK(cudaGetDevice(&Device));

      int MaxSmemPerSm = 0;
      int MaxSmemPerBlock = 0;
      C10_CUDA_CHECK(cudaDeviceGetAttribute(
          &MaxSmemPerSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, Device));
      C10_CUDA_CHECK(cudaDeviceGetAttribute(
          &MaxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, Device));

      // Matches run_mha_fwd_hdim256() selection heuristic.
      if (MaxSmemPerBlock >= 2 * HeadDimBucket * (128 + 2 * 64) &&
          MaxSmemPerSm < 4 * HeadDimBucket * (64 + 2 * 64)) {
        return &kCfgH256B128N64W8;
      }
      return &kCfgH256B64N64W4;
    }
    default:
      return nullptr;
  }
}

const KernelConfig *select_bf16_kernel_config(const Flash_fwd_params &params) {
  const int HeadDimBucket = get_headdim_bucket(params.d);
  if (HeadDimBucket == 0) {
    return nullptr;
  }

  const bool Is_dropout = params.p_dropout < 1.f;
  const bool Is_causal = params.is_causal;

  switch (HeadDimBucket) {
    case 32:
      return &kCfgH32B128N128W4_BF16;
    case 64:
      return Is_dropout ? &kCfgH64B128N64W4_BF16 : &kCfgH64B128N128W4_BF16;
    case 96: {
      auto [CcMajor, CcMinor] =
          get_compute_capability(get_current_device());
      const bool IsSm8x = CcMajor == 8 && CcMinor > 0;
      if (IsSm8x && Is_causal) {
        return &kCfgH96B64N64W4_BF16;
      }
      return &kCfgH96B128N64W4_BF16;
    }
    case 128: {
      auto [CcMajor, CcMinor] =
          get_compute_capability(get_current_device());
      const bool IsSm8x = CcMajor == 8 && CcMinor > 0;
      if (!Is_dropout) {
        if (IsSm8x) {
          return Is_causal ? &kCfgH128B64N64W4_BF16 : &kCfgH128B128N32W4_BF16;
        }
        return &kCfgH128B128N64W4_BF16;
      }
      return &kCfgH128B128N32W4_BF16;
    }
    case 192:
      return Is_dropout ? &kCfgH192B64N64W4_BF16 : &kCfgH192B128N64W8_BF16;
    case 256: {
      int Device = 0;
      C10_CUDA_CHECK(cudaGetDevice(&Device));

      int MaxSmemPerSm = 0;
      int MaxSmemPerBlock = 0;
      C10_CUDA_CHECK(cudaDeviceGetAttribute(
          &MaxSmemPerSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, Device));
      C10_CUDA_CHECK(cudaDeviceGetAttribute(
          &MaxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, Device));

      if (MaxSmemPerBlock >= 2 * HeadDimBucket * (128 + 2 * 64) &&
          MaxSmemPerSm < 4 * HeadDimBucket * (64 + 2 * 64)) {
        return &kCfgH256B128N64W8_BF16;
      }
      return &kCfgH256B64N64W4_BF16;
    }
    default:
      return nullptr;
  }
}

bool is_eligible_fastpath(const Flash_fwd_params &params) {
  return get_headdim_bucket(params.d) != 0;
}

}  // namespace

bool try_run_mha_fwd_jit_fp16(Flash_fwd_params &params, cudaStream_t stream) {
  if (!use_proteus_jit() || !is_eligible_fastpath(params)) {
    return false;
  }

  const KernelConfig *Config = select_fp16_kernel_config(params);
  if (Config == nullptr) {
    return false;
  }

  const bool Is_dropout_base = params.p_dropout < 1.f;
  const bool Is_causal = params.is_causal;
  const bool Is_local =
      (params.window_size_left >= 0 || params.window_size_right >= 0) &&
      !Is_causal;
  const bool Has_alibi = params.alibi_slopes_ptr != nullptr;
  const bool Return_softmax = params.p_ptr != nullptr;
  const bool Is_softcap = params.softcap > 0.0f;

  const bool Is_even_MN_raw =
      params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
      params.seqlen_k % Config->BlockN == 0 &&
      params.seqlen_q % Config->BlockM == 0;
  const bool Is_even_K_raw = params.d == Config->HeadDim;

  // Keep JIT template selection aligned with run_flash_fwd<> template logic.
  const bool Is_dropout_t = Is_dropout_base && !Is_softcap;
  const bool Is_even_MN_t = Is_even_MN_raw && Is_even_K_raw && !Is_local &&
                            !Has_alibi && !Return_softmax &&
                            Config->HeadDim <= 128;
  const bool Is_even_K_t = Is_even_K_raw && !Return_softmax && !Has_alibi;
  const bool Return_softmax_t = Return_softmax && Is_dropout_base && !Is_softcap;

  const int num_m_block =
      (params.seqlen_q + Config->BlockM - 1) / Config->BlockM;
  const dim3 Grid(num_m_block, params.b, params.h);
  const dim3 Block(Config->NThreads, 1, 1);
  const size_t SmemSize = Config->SmemSize;

  auto Instance = get_module().instantiate(
      "flash_fwd_kernel_jit", Config->KernelTraitsName,
      to_template_bool(Is_dropout_t), to_template_bool(Is_causal),
      to_template_bool(Is_local), to_template_bool(Has_alibi),
      to_template_bool(Is_even_MN_t), to_template_bool(Is_even_K_t),
      to_template_bool(Is_softcap), to_template_bool(Return_softmax_t));

  auto t0 = std::chrono::steady_clock::now();
  Instance.launch(Grid, Block, SmemSize, reinterpret_cast<void *>(stream),
                  params);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::fprintf(stderr, "[proteus-jit] fwd_fp16 h=%d launch %ld ms\n",
               Config->HeadDim, ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool try_run_mha_fwd_jit_bf16(Flash_fwd_params &params, cudaStream_t stream) {
  if (!use_proteus_jit() || !is_eligible_fastpath(params)) {
    return false;
  }

  const KernelConfig *Config = select_bf16_kernel_config(params);
  if (Config == nullptr) {
    return false;
  }

  const bool Is_dropout_base = params.p_dropout < 1.f;
  const bool Is_causal = params.is_causal;
  const bool Is_local =
      (params.window_size_left >= 0 || params.window_size_right >= 0) &&
      !Is_causal;
  const bool Has_alibi = params.alibi_slopes_ptr != nullptr;
  const bool Return_softmax = params.p_ptr != nullptr;
  const bool Is_softcap = params.softcap > 0.0f;

  const bool Is_even_MN_raw =
      params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
      params.seqlen_k % Config->BlockN == 0 &&
      params.seqlen_q % Config->BlockM == 0;
  const bool Is_even_K_raw = params.d == Config->HeadDim;

  const bool Is_dropout_t = Is_dropout_base && !Is_softcap;
  const bool Is_even_MN_t = Is_even_MN_raw && Is_even_K_raw && !Is_local &&
                            !Has_alibi && !Return_softmax &&
                            Config->HeadDim <= 128;
  const bool Is_even_K_t = Is_even_K_raw && !Return_softmax && !Has_alibi;
  const bool Return_softmax_t = Return_softmax && Is_dropout_base && !Is_softcap;

  const int num_m_block =
      (params.seqlen_q + Config->BlockM - 1) / Config->BlockM;
  const dim3 Grid(num_m_block, params.b, params.h);
  const dim3 Block(Config->NThreads, 1, 1);
  const size_t SmemSize = Config->SmemSize;

  auto Instance = get_module().instantiate(
      "flash_fwd_kernel_jit", Config->KernelTraitsName,
      to_template_bool(Is_dropout_t), to_template_bool(Is_causal),
      to_template_bool(Is_local), to_template_bool(Has_alibi),
      to_template_bool(Is_even_MN_t), to_template_bool(Is_even_K_t),
      to_template_bool(Is_softcap), to_template_bool(Return_softmax_t));

  auto t0 = std::chrono::steady_clock::now();
  Instance.launch(Grid, Block, SmemSize, reinterpret_cast<void *>(stream),
                  params);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::fprintf(stderr, "[proteus-jit] fwd_bf16 h=%d launch %ld ms\n",
               Config->HeadDim, ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace FLASH_NAMESPACE
