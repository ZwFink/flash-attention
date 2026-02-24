#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>

// Lightweight typedef so the CppJitModule-generated extern "C" wrapper
// (which uses c10::Half parameter types via typeName<T>()) can compile
// without pulling in the full PyTorch c10 header tree.
// ABI-compatible: both are alignas(2), 2-byte unsigned short layout.
namespace c10 { using Half = __half; }

#define C10_WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl_xor_sync(mask, value, laneMask, width);
}

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false)
{
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    if (is_transformer_mask) {
        mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
    } else {
        mask += idx_offset;
    }

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*element_count;
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += ::exp(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = ::exp(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*element_count;
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += ::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = ::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = ::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*element_count+it*WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}
