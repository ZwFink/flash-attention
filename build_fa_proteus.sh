#!/bin/bash
# build_fa_proteus.sh — Build Flash-Attention with Proteus JIT support
#
# This script:
#   1. Clones and builds Proteus (JIT framework) against the conda env's LLVM
#   2. Clones and builds Flash-Attention with Proteus JIT enabled
#   3. Runs a smoke test to verify the build
#
# Prerequisites:
#   - CUDA 12.2+ available at CUDA_ROOT (or /usr/tce/packages/cuda/cuda-12.2.2)
#   - A conda environment with: LLVM/Clang 19.x, Python 3.12, PyTorch 2.x
#   - GCC >= 9 (the system default may be too old; override with GCC_ROOT)
#   - An active SLURM/Flux allocation with GPU access (for the smoke test)
#
# Usage:
#   # Minimal (uses defaults for LLNL matrix cluster):
#   ./build_fa_proteus.sh
#
#   # Custom paths:
#   INSTALL_DIR=/path/to/install CUDA_ROOT=/usr/local/cuda ./build_fa_proteus.sh
#
#   # Skip Proteus build (reuse existing install):
#   PROTEUS_INSTALL=/existing/proteus/install ./build_fa_proteus.sh
#
set -euo pipefail

###############################################################################
# Configuration — override any of these via environment variables
###############################################################################

# Where to put everything (clones + builds + installs)
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/fa-proteus-build}"

# CUDA toolkit
CUDA_ROOT="${CUDA_ROOT:-/usr/tce/packages/cuda/cuda-12.2.2}"

# GCC >= 9 for host compilation (PyTorch headers require it)
GCC_ROOT="${GCC_ROOT:-/usr/tce/packages/gcc/gcc-12.1.1}"

# Conda environment providing LLVM 19.x, Python, PyTorch
# The script auto-detects from CONDA_PREFIX if set.
CONDA_ENV="${CONDA_ENV:-${CONDA_PREFIX:-/usr/workspace/fink12/miniconda3_tioga/envs/proteus_matrix}}"

# Flash-Attention repo
FA_REPO="${FA_REPO:-git@github.com:ZwFink/flash-attention.git}"
FA_BRANCH="${FA_BRANCH:-main}"

# Proteus repo
PROTEUS_REPO="${PROTEUS_REPO:-git@github.com:Olympus-HPC/proteus.git}"
PROTEUS_BRANCH="${PROTEUS_BRANCH:-main}"

# If set, skip building Proteus and use this existing installation
PROTEUS_INSTALL="${PROTEUS_INSTALL:-}"

# CUDA architectures to compile for (semicolon-separated)
CUDA_ARCHS="${CUDA_ARCHS:-80;90}"

# Number of parallel build jobs
JOBS="${JOBS:-$(nproc)}"

###############################################################################
# Derived paths
###############################################################################

LLVM_DIR="${CONDA_ENV}/lib"        # libLLVM.so, libclang-cpp.so
CLANG_BIN="${CONDA_ENV}/bin"       # clang, clang++ (for Proteus cmake)
PYTHON_BIN="${CONDA_ENV}/bin/python"

export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"

###############################################################################
# Sanity checks
###############################################################################

echo "=== Flash-Attention + Proteus JIT Build Script ==="
echo ""

check_file() { [ -e "$1" ] || { echo "ERROR: $1 not found"; exit 1; }; }
check_cmd()  { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found in PATH"; exit 1; }; }

check_file "${CUDA_ROOT}/bin/nvcc"
check_file "${CC}"
check_file "${CXX}"
check_file "${LLVM_DIR}/libLLVM.so"
check_file "${LLVM_DIR}/libclang-cpp.so"
check_file "${PYTHON_BIN}"
check_cmd  cmake
check_cmd  ninja || check_cmd make

echo "CUDA_ROOT    = ${CUDA_ROOT}"
echo "GCC          = ${CC}"
echo "CONDA_ENV    = ${CONDA_ENV}"
echo "LLVM_DIR     = ${LLVM_DIR}"
echo "INSTALL_DIR  = ${INSTALL_DIR}"
echo "CUDA_ARCHS   = ${CUDA_ARCHS}"
echo ""

mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

###############################################################################
# Step 1: Build Proteus
###############################################################################

if [ -n "${PROTEUS_INSTALL}" ] && [ -f "${PROTEUS_INSTALL}/lib64/libproteus.a" ]; then
    echo "=== Using existing Proteus install: ${PROTEUS_INSTALL} ==="
else
    echo "=== Step 1: Building Proteus ==="

    PROTEUS_SRC="${INSTALL_DIR}/proteus"
    PROTEUS_BUILD="${INSTALL_DIR}/proteus-build"
    PROTEUS_INSTALL="${INSTALL_DIR}/proteus-install"

    if [ ! -d "${PROTEUS_SRC}" ]; then
        echo "Cloning Proteus..."
        git clone "${PROTEUS_REPO}" "${PROTEUS_SRC}"
    fi
    cd "${PROTEUS_SRC}"
    git checkout "${PROTEUS_BRANCH}"
    cd "${INSTALL_DIR}"

    mkdir -p "${PROTEUS_BUILD}"
    cd "${PROTEUS_BUILD}"

    echo "Configuring Proteus..."
    cmake "${PROTEUS_SRC}" \
        -DLLVM_INSTALL_DIR="${CONDA_ENV}" \
        -DPROTEUS_ENABLE_CUDA=ON \
        -DBUILD_SHARED=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_COMPILER="${CLANG_BIN}/clang" \
        -DCMAKE_CXX_COMPILER="${CLANG_BIN}/clang++" \
        -DCMAKE_INSTALL_PREFIX="${PROTEUS_INSTALL}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    echo "Building Proteus (${JOBS} jobs)..."
    make -j"${JOBS}"

    echo "Installing Proteus..."
    make install

    echo "Proteus installed to: ${PROTEUS_INSTALL}"
    cd "${INSTALL_DIR}"
fi

# Verify Proteus has the right LLVM symbols (19.x, not 20.x)
if nm "${PROTEUS_INSTALL}/lib64/libproteus.a" 2>/dev/null | grep -q 'LLVMTargetMachine'; then
    echo "  Proteus links against LLVM 19.x symbols (LLVMTargetMachine) — OK"
elif nm "${PROTEUS_INSTALL}/lib64/libproteus.a" 2>/dev/null | grep -q 'TargetMachineE$'; then
    echo "  WARNING: Proteus may be built against LLVM 20+, which will cause"
    echo "  symbol conflicts with PyTorch's LLVM 19. Consider rebuilding Proteus"
    echo "  with LLVM_INSTALL_DIR pointing to the conda env's LLVM 19."
fi

###############################################################################
# Step 2: Build Flash-Attention with Proteus JIT
###############################################################################

echo ""
echo "=== Step 2: Building Flash-Attention with Proteus JIT ==="

FA_SRC="${INSTALL_DIR}/flash-attention"

if [ ! -d "${FA_SRC}" ]; then
    echo "Cloning Flash-Attention (with submodules)..."
    git clone --recurse-submodules -b "${FA_BRANCH}" "${FA_REPO}" "${FA_SRC}"
fi

cd "${FA_SRC}"

echo "Building flash_attn_2_cuda extension in-place..."
FLASH_ATTN_WITH_PROTEUS_JIT=1 \
FLASH_ATTN_CUDA_ARCHS="${CUDA_ARCHS}" \
PROTEUS_PREFIX="${PROTEUS_INSTALL}" \
PROTEUS_LLVM_LIBDIR="${LLVM_DIR}" \
PROTEUS_CLANG_LIBDIR="${LLVM_DIR}" \
PROTEUS_CUDA_ROOT="${CUDA_ROOT}" \
"${PYTHON_BIN}" setup.py build_ext --inplace

# Verify the .so was built
FA_SO=$(ls "${FA_SRC}"/flash_attn_2_cuda*.so 2>/dev/null | head -1)
if [ -z "${FA_SO}" ]; then
    echo "ERROR: flash_attn_2_cuda .so not found after build"
    exit 1
fi
echo "Built: ${FA_SO}"

# Verify no dual-LLVM linkage
LLVM_VERSIONS=$(ldd "${FA_SO}" 2>/dev/null | grep -c 'libLLVM\.so\.' || true)
if [ "${LLVM_VERSIONS}" -gt 1 ]; then
    echo "WARNING: Multiple LLVM versions detected in .so linkage."
    echo "This will likely cause segfaults. Check PROTEUS_LLVM_LIBDIR."
    ldd "${FA_SO}" | grep LLVM
fi

###############################################################################
# Auto-detect PyTorch include paths for JIT compilation
###############################################################################

echo ""
echo "=== Detecting PyTorch include paths ==="

# Run from /tmp to avoid CWD interference from the FA source tree.
# Use head -1 to discard any library warnings printed to stdout (e.g. proteus).
TORCH_DIR=$(cd /tmp && "${PYTHON_BIN}" -c "import torch; print(torch.__path__[0])" 2>/dev/null | head -1) || true
if [ -z "${TORCH_DIR}" ] || [ ! -d "${TORCH_DIR}" ]; then
    echo "ERROR: Could not detect PyTorch installation via ${PYTHON_BIN}"
    echo "       Make sure PyTorch is installed in the conda environment."
    exit 1
fi

TORCH_INCLUDE="${TORCH_DIR}/include"
TORCH_API_INCLUDE="${TORCH_INCLUDE}/torch/csrc/api/include"
TORCH_ATEN_INCLUDE="${TORCH_INCLUDE}"
CUDA_INCLUDE="${CUDA_ROOT}/include"

echo "TORCH_DIR          = ${TORCH_DIR}"
echo "TORCH_INCLUDE      = ${TORCH_INCLUDE}"
echo "TORCH_API_INCLUDE  = ${TORCH_API_INCLUDE}"
echo "TORCH_ATEN_INCLUDE = ${TORCH_ATEN_INCLUDE}"
echo "CUDA_INCLUDE       = ${CUDA_INCLUDE}"

# Verify key headers exist
check_file "${TORCH_API_INCLUDE}/torch/torch.h"
check_file "${TORCH_ATEN_INCLUDE}/ATen/cuda/CUDAGeneratorImpl.h"
check_file "${CUDA_INCLUDE}/cuda.h"

###############################################################################
# Step 3: Smoke test
###############################################################################

echo ""
echo "=== Step 3: Smoke Test ==="

# Run from /tmp to avoid sys.path CWD interference
cd /tmp

if PYTHONPATH="${FA_SRC}:${PYTHONPATH:-}" \
   FLASH_ATTN_USE_PROTEUS_JIT=1 \
   FLASH_ATTN_JIT_TORCH_INCLUDE="${TORCH_INCLUDE}" \
   FLASH_ATTN_JIT_TORCH_API_INCLUDE="${TORCH_API_INCLUDE}" \
   FLASH_ATTN_JIT_TORCH_ATEN_INCLUDE="${TORCH_ATEN_INCLUDE}" \
   FLASH_ATTN_JIT_CUDA_INCLUDE="${CUDA_INCLUDE}" \
   "${PYTHON_BIN}" -u -c "
import torch
from flash_attn.flash_attn_interface import flash_attn_func
torch.set_grad_enabled(False)
q = torch.randn(2, 256, 8, 128, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, torch.randn_like(q), torch.randn_like(q), causal=False)
print(f'Smoke test PASSED: output shape {out.shape}, mean {out.float().mean().item():.6f}')
" 2>&1; then
    echo ""
    echo "=== BUILD AND TEST SUCCESSFUL ==="
else
    SMOKE_EXIT=$?
    echo ""
    echo "=== Smoke test failed (exit code ${SMOKE_EXIT}) ==="
    echo "The extension built, but the runtime test failed."
    echo "Make sure you have a GPU available (run inside a SLURM/Flux allocation)."
    exit "${SMOKE_EXIT}"
fi

###############################################################################
# Print usage instructions
###############################################################################

echo ""
echo "==========================================================================="
echo " To use this Flash-Attention build with Proteus JIT:"
echo "==========================================================================="
echo ""
echo "  # Add to PYTHONPATH (run from any directory EXCEPT the FA source tree):"
echo "  export PYTHONPATH=${FA_SRC}:\${PYTHONPATH:-}"
echo ""
echo "  # Enable JIT at runtime:"
echo "  export FLASH_ATTN_USE_PROTEUS_JIT=1"
echo ""
echo "  # Tell the JIT compiler where to find PyTorch and CUDA headers:"
echo "  export FLASH_ATTN_JIT_TORCH_INCLUDE=${TORCH_INCLUDE}"
echo "  export FLASH_ATTN_JIT_TORCH_API_INCLUDE=${TORCH_API_INCLUDE}"
echo "  export FLASH_ATTN_JIT_TORCH_ATEN_INCLUDE=${TORCH_ATEN_INCLUDE}"
echo "  export FLASH_ATTN_JIT_CUDA_INCLUDE=${CUDA_INCLUDE}"
echo ""
echo "  # Then use flash_attn as usual:"
echo "  python -c 'from flash_attn import flash_attn_func; print(\"OK\")'"
echo ""
echo "  # To disable JIT and fall back to AOT kernels (requires non-JIT build):"
echo "  unset FLASH_ATTN_USE_PROTEUS_JIT"
echo "==========================================================================="
