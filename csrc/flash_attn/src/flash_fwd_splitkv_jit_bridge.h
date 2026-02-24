/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "flash.h"

namespace FLASH_NAMESPACE {

// Returns true if the kernel was launched via Proteus JIT, false to request
// fallback to the existing AOT dispatch path.
bool try_run_mha_fwd_splitkv_jit_fp16(Flash_fwd_params &params, cudaStream_t stream);
bool try_run_mha_fwd_splitkv_jit_bf16(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
