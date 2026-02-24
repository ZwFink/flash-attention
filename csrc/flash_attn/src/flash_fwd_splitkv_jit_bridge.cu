/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_fwd_splitkv_jit_bridge.h"

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

// ---------------------------------------------------------------------------
// Kernel config: one per HeadDim per dtype.
// ---------------------------------------------------------------------------
struct SplitKVKernelConfig {
  const char *KernelTraitsName;
  int HeadDim;
  int BlockM;
  int BlockN;
  int NThreads;
  size_t SmemSize;
  int CombineBlockM;  // kBlockM used by the combine kernel
};

// Split-KV uses fixed kBlockM=64, kBlockN depends on HeadDim, NWarps=4,
// Is_Q_in_regs=false, Share_Q_K_smem=false for all HeadDims.
template <int HeadDim, int BlockN>
SplitKVKernelConfig make_splitkv_fp16_config(const char *name) {
  using KT = Flash_fwd_kernel_traits<HeadDim, 64, BlockN, 4, false, false,
                                     cutlass::half_t>;
  constexpr int CombBlockM = HeadDim % 128 == 0 ? 4 : (HeadDim % 64 == 0 ? 8 : 16);
  return {name, HeadDim, KT::kBlockM, KT::kBlockN, KT::kNThreads,
          KT::kSmemSize, CombBlockM};
}

template <int HeadDim, int BlockN>
SplitKVKernelConfig make_splitkv_bf16_config(const char *name) {
  using KT = Flash_fwd_kernel_traits<HeadDim, 64, BlockN, 4, false, false,
                                     cutlass::bfloat16_t>;
  constexpr int CombBlockM = HeadDim % 128 == 0 ? 4 : (HeadDim % 64 == 0 ? 8 : 16);
  return {name, HeadDim, KT::kBlockM, KT::kBlockN, KT::kNThreads,
          KT::kSmemSize, CombBlockM};
}

// kBlockN = HeadDim<=64 ? 256 : (HeadDim<=128 ? 128 : 64)
// fp16 configs
const SplitKVKernelConfig kSplitH32  = make_splitkv_fp16_config<32,  256>("fa2_splitkv_h32_fp16");
const SplitKVKernelConfig kSplitH64  = make_splitkv_fp16_config<64,  256>("fa2_splitkv_h64_fp16");
const SplitKVKernelConfig kSplitH96  = make_splitkv_fp16_config<96,  128>("fa2_splitkv_h96_fp16");
const SplitKVKernelConfig kSplitH128 = make_splitkv_fp16_config<128, 128>("fa2_splitkv_h128_fp16");
const SplitKVKernelConfig kSplitH192 = make_splitkv_fp16_config<192,  64>("fa2_splitkv_h192_fp16");
const SplitKVKernelConfig kSplitH256 = make_splitkv_fp16_config<256,  64>("fa2_splitkv_h256_fp16");

// bf16 configs
const SplitKVKernelConfig kSplitH32_BF16  = make_splitkv_bf16_config<32,  256>("fa2_splitkv_h32_bf16");
const SplitKVKernelConfig kSplitH64_BF16  = make_splitkv_bf16_config<64,  256>("fa2_splitkv_h64_bf16");
const SplitKVKernelConfig kSplitH96_BF16  = make_splitkv_bf16_config<96,  128>("fa2_splitkv_h96_bf16");
const SplitKVKernelConfig kSplitH128_BF16 = make_splitkv_bf16_config<128, 128>("fa2_splitkv_h128_bf16");
const SplitKVKernelConfig kSplitH192_BF16 = make_splitkv_bf16_config<192,  64>("fa2_splitkv_h192_bf16");
const SplitKVKernelConfig kSplitH256_BF16 = make_splitkv_bf16_config<256,  64>("fa2_splitkv_h256_bf16");

// ---------------------------------------------------------------------------
// Shared helpers (duplicated from forward bridge for link independence).
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
      std::fprintf(stderr, "[flash-attn-splitkv-jit] %s\n", Arg.c_str());
    }
  }
  return Args;
}

const char *to_template_bool(bool Value) { return Value ? "true" : "false"; }

int get_headdim_bucket(int HeadDim) {
  if (HeadDim <= 32)  return 32;
  if (HeadDim <= 64)  return 64;
  if (HeadDim <= 96)  return 96;
  if (HeadDim <= 128) return 128;
  if (HeadDim <= 192) return 192;
  if (HeadDim <= 256) return 256;
  return 0;
}

// ---------------------------------------------------------------------------
// JIT code string — defines the kernel wrappers that Proteus will specialize.
// ---------------------------------------------------------------------------
std::string get_splitkv_jit_kernel_code() {
  return R"code(
#include <tuple>
#include "namespace_config.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

using flash_fwd_params_t = FLASH_NAMESPACE::Flash_fwd_params;

// fp16 kernel traits — one per HeadDim
using fa2_splitkv_h32_fp16  = ::Flash_fwd_kernel_traits<32,  64, 256, 4, false, false, cutlass::half_t>;
using fa2_splitkv_h64_fp16  = ::Flash_fwd_kernel_traits<64,  64, 256, 4, false, false, cutlass::half_t>;
using fa2_splitkv_h96_fp16  = ::Flash_fwd_kernel_traits<96,  64, 128, 4, false, false, cutlass::half_t>;
using fa2_splitkv_h128_fp16 = ::Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t>;
using fa2_splitkv_h192_fp16 = ::Flash_fwd_kernel_traits<192, 64,  64, 4, false, false, cutlass::half_t>;
using fa2_splitkv_h256_fp16 = ::Flash_fwd_kernel_traits<256, 64,  64, 4, false, false, cutlass::half_t>;

// bf16 kernel traits
using fa2_splitkv_h32_bf16  = ::Flash_fwd_kernel_traits<32,  64, 256, 4, false, false, cutlass::bfloat16_t>;
using fa2_splitkv_h64_bf16  = ::Flash_fwd_kernel_traits<64,  64, 256, 4, false, false, cutlass::bfloat16_t>;
using fa2_splitkv_h96_bf16  = ::Flash_fwd_kernel_traits<96,  64, 128, 4, false, false, cutlass::bfloat16_t>;
using fa2_splitkv_h128_bf16 = ::Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;
using fa2_splitkv_h192_bf16 = ::Flash_fwd_kernel_traits<192, 64,  64, 4, false, false, cutlass::bfloat16_t>;
using fa2_splitkv_h256_bf16 = ::Flash_fwd_kernel_traits<256, 64,  64, 4, false, false, cutlass::bfloat16_t>;

// Main split-KV kernel wrapper
template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV>
__global__ void flash_fwd_splitkv_kernel_jit(const flash_fwd_params_t params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi,
                                        Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
#else
  printf("FATAL: FlashAttention requires sm >= 80!\n");
#endif
}

// Combine kernel wrapper
template <typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K>
__global__ void flash_fwd_splitkv_combine_kernel_jit(const flash_fwd_params_t params) {
  static_assert(Log_max_splits >= 1);
  FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}
)code";
}

// ---------------------------------------------------------------------------
// Proteus JIT module (lazily initialized singleton).
// ---------------------------------------------------------------------------
auto &get_splitkv_module() {
  static const std::string Code = get_splitkv_jit_kernel_code();
  static auto Module = std::make_unique<proteus::CppJitModule>(
      "cuda", Code, get_extra_args());
  return *Module;
}

// ---------------------------------------------------------------------------
// Config selection — simple HeadDim bucket lookup (no dropout/smem branching).
// ---------------------------------------------------------------------------
const SplitKVKernelConfig *select_splitkv_fp16_config(const Flash_fwd_params &params) {
  switch (get_headdim_bucket(params.d)) {
    case  32: return &kSplitH32;
    case  64: return &kSplitH64;
    case  96: return &kSplitH96;
    case 128: return &kSplitH128;
    case 192: return &kSplitH192;
    case 256: return &kSplitH256;
    default:  return nullptr;
  }
}

const SplitKVKernelConfig *select_splitkv_bf16_config(const Flash_fwd_params &params) {
  switch (get_headdim_bucket(params.d)) {
    case  32: return &kSplitH32_BF16;
    case  64: return &kSplitH64_BF16;
    case  96: return &kSplitH96_BF16;
    case 128: return &kSplitH128_BF16;
    case 192: return &kSplitH192_BF16;
    case 256: return &kSplitH256_BF16;
    default:  return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Launch implementation (shared by fp16 and bf16).
// ---------------------------------------------------------------------------
bool try_run_mha_fwd_splitkv_jit_impl(
    Flash_fwd_params &params, cudaStream_t stream,
    const SplitKVKernelConfig *Config, const char *DtypeLabel) {
  if (Config == nullptr) return false;

  // --- Boolean flags ---
  const bool Is_causal = params.is_causal;
  const bool Is_local =
      (params.window_size_left >= 0 || params.window_size_right >= 0) &&
      !Is_causal;
  const bool Has_alibi = params.alibi_slopes_ptr != nullptr;
  const bool Is_softcap = params.softcap > 0.0f;
  const bool Split = params.num_splits > 1;
  const bool Append_KV = params.knew_ptr != nullptr;

  const bool Is_even_MN_raw =
      params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
      params.seqlen_k % Config->BlockN == 0 &&
      params.seqlen_q % Config->BlockM == 0;
  const bool Is_even_K_raw = params.d == Config->HeadDim;

  // Template arg refinement — matches run_flash_splitkv_fwd<> logic.
  const bool Is_even_MN_t = Is_even_MN_raw && !Append_KV && Is_even_K_raw &&
                            !Is_local && !Has_alibi &&
                            Config->HeadDim <= 128;
  const bool Is_even_K_t = Is_even_K_raw && !Has_alibi;

  // --- Launch main split-KV kernel ---
  const int num_m_block =
      (params.seqlen_q + Config->BlockM - 1) / Config->BlockM;
  const dim3 Grid(num_m_block,
                  Split ? params.num_splits : params.b,
                  Split ? params.b * params.h : params.h);
  const dim3 Block(Config->NThreads, 1, 1);
  const size_t SmemSize = Config->SmemSize;

  auto MainInstance = get_splitkv_module().instantiate(
      "flash_fwd_splitkv_kernel_jit", Config->KernelTraitsName,
      to_template_bool(Is_causal),
      to_template_bool(Is_local && !Is_causal),
      to_template_bool(Has_alibi),
      to_template_bool(Is_even_MN_t),
      to_template_bool(Is_even_K_t),
      to_template_bool(Is_softcap),
      to_template_bool(Split),
      to_template_bool(Append_KV));

  auto t0_main = std::chrono::steady_clock::now();
  MainInstance.launch(Grid, Block, SmemSize,
                      reinterpret_cast<void *>(stream), params);
  auto t1_main = std::chrono::steady_clock::now();
  auto ms_main = std::chrono::duration_cast<std::chrono::milliseconds>(t1_main - t0_main).count();
  std::fprintf(stderr, "[proteus-jit] %s h=%d main_launch %ld ms\n",
               DtypeLabel, Config->HeadDim, ms_main);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // --- Launch combine kernel (only if num_splits > 1) ---
  if (params.num_splits > 1) {
    const int CombBlockM = Config->CombineBlockM;

    int LogMaxSplits;
    if (params.num_splits <= 2)        LogMaxSplits = 1;
    else if (params.num_splits <= 4)   LogMaxSplits = 2;
    else if (params.num_splits <= 8)   LogMaxSplits = 3;
    else if (params.num_splits <= 16)  LogMaxSplits = 4;
    else if (params.num_splits <= 32)  LogMaxSplits = 5;
    else if (params.num_splits <= 64)  LogMaxSplits = 6;
    else                               LogMaxSplits = 7;

    const dim3 GridCombine(
        (params.b * params.h * params.seqlen_q + CombBlockM - 1) / CombBlockM);
    const dim3 BlockCombine(Config->NThreads, 1, 1);

    auto CombineInstance = get_splitkv_module().instantiate(
        "flash_fwd_splitkv_combine_kernel_jit", Config->KernelTraitsName,
        std::to_string(CombBlockM).c_str(),
        std::to_string(LogMaxSplits).c_str(),
        to_template_bool(Is_even_K_t));

    auto t0_comb = std::chrono::steady_clock::now();
    CombineInstance.launch(GridCombine, BlockCombine, /*smem=*/0,
                           reinterpret_cast<void *>(stream), params);
    auto t1_comb = std::chrono::steady_clock::now();
    auto ms_comb = std::chrono::duration_cast<std::chrono::milliseconds>(t1_comb - t0_comb).count();
    std::fprintf(stderr, "[proteus-jit] %s h=%d combine_launch %ld ms\n",
                 DtypeLabel, Config->HeadDim, ms_comb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
bool try_run_mha_fwd_splitkv_jit_fp16(Flash_fwd_params &params,
                                       cudaStream_t stream) {
  if (!use_proteus_jit()) return false;
  return try_run_mha_fwd_splitkv_jit_impl(
      params, stream, select_splitkv_fp16_config(params), "splitkv_fp16");
}

bool try_run_mha_fwd_splitkv_jit_bf16(Flash_fwd_params &params,
                                       cudaStream_t stream) {
  if (!use_proteus_jit()) return false;
  return try_run_mha_fwd_splitkv_jit_impl(
      params, stream, select_splitkv_bf16_config(params), "splitkv_bf16");
}

}  // namespace FLASH_NAMESPACE
