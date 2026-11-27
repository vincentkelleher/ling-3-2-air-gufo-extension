// Ling 3.2 Air HIP kernels.
//
// Layouts are GGUF/llama.cpp's: Q8_0 in 32-weight blocks (fp16 scale + int8),
// Q4_K and Q6_K in 256-weight super-blocks. The unpacking matches Gufo's host
// reference (src/core/quant/ggml_dequant.cpp), which tests/dot_test.cpp checks
// against `Dot()`; tests/operator_gpu_test.cpp checks these kernels against a
// host dot on the same bytes.
#include "src/hip/kernels.hpp"

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::ling3_air::hip {
namespace {

#pragma pack(push, 1)
struct BlockQ8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
struct BlockQ5K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block layout");
static_assert(sizeof(BlockQ4K) == 144, "Q4_K block layout");
static_assert(sizeof(BlockQ6K) == 210, "Q6_K block layout");
static_assert(sizeof(BlockQ5K) == 176, "Q5_K block layout");
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block layout");

__device__ __forceinline__ float HalfToFloat(std::uint16_t bits) {
  const __half value = *reinterpret_cast<const __half*>(&bits);
  return __half2float(value);
}

/// Q4_K sub-block scale and minimum (Gufo host: GetQ4ScaleMin).
__device__ __forceinline__ void Q4ScaleMin(int index, const std::uint8_t* packed,
                                           float& scale, float& minimum) {
  if (index < 4) {
    scale = static_cast<float>(packed[index] & 0x3FU);
    minimum = static_cast<float>(packed[index + 4] & 0x3FU);
    return;
  }
  scale = static_cast<float>((packed[index + 4] & 0x0FU) |
                             ((packed[index - 4] >> 6U) << 4U));
  minimum = static_cast<float>((packed[index + 4] >> 4U) |
                               ((packed[index] >> 6U) << 4U));
}

__device__ __forceinline__ float Q4Value(const BlockQ4K& block, int index) {
  const int group = index / 32;
  const int pair = group / 2;
  const bool high = (group & 1) != 0;
  const int lane = index % 32;
  const std::uint8_t packed = block.qs[pair * 32 + lane];
  const float quant = static_cast<float>(high ? (packed >> 4U) : (packed & 0x0FU));
  float scale = 0.0f;
  float minimum = 0.0f;
  Q4ScaleMin(group, block.scales, scale, minimum);
  return HalfToFloat(block.d) * scale * quant - HalfToFloat(block.dmin) * minimum;
}

__device__ __forceinline__ int Q4Code(const BlockQ4K& block, int index) {
  const int group = index / 32;
  const int lane = index % 32;
  const std::uint8_t packed = block.qs[(group / 2) * 32 + lane];
  return (group & 1) ? (packed >> 4) : (packed & 0x0F);
}

__device__ __forceinline__ int Q6Code(const BlockQ6K& block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int segment = within / 32;
  const int lane = within % 32;
  const int ql_base = half * 64;
  const std::uint8_t qh = block.qh[half * 32 + lane];
  std::uint8_t low = 0, high = 0;
  switch (segment) {
    case 0: low = block.ql[ql_base + lane] & 0x0F; high = qh & 0x03; break;
    case 1: low = block.ql[ql_base + 32 + lane] & 0x0F; high = (qh >> 2) & 0x03; break;
    case 2: low = block.ql[ql_base + lane] >> 4; high = (qh >> 4) & 0x03; break;
    default: low = block.ql[ql_base + 32 + lane] >> 4; high = (qh >> 6) & 0x03; break;
  }
  return static_cast<int>((high << 4) | low) - 32;
}

/// Q4_0: one fp16 scale per 32 values (which may be negative); a byte holds the
/// low nibble of value `j` and the high nibble of value `j + 16`.
__device__ __forceinline__ int Q4_0Code(const BlockQ4_0& block, int index) {
  const int byte = index & 15;
  const int nibble = (index < 16) ? (block.qs[byte] & 0x0F) : (block.qs[byte] >> 4);
  return nibble - 8;
}

__device__ __forceinline__ float Q5Value(const BlockQ5K& block, int index) {
  const int group = index / 64;
  const int within = index % 64;
  const int lane = within % 32;
  const bool low_half = within < 32;
  const std::uint8_t packed = block.qs[group * 32 + lane];
  const std::uint8_t quant4 = low_half ? (packed & 0x0FU) : (packed >> 4U);
  const std::uint8_t high = block.qh[lane];
  const int bit = 2 * group + (low_half ? 0 : 1);
  const float quant =
      static_cast<float>(quant4) + (((high >> bit) & 1U) ? 16.0f : 0.0f);
  const int scale_index = 2 * group + (low_half ? 0 : 1);
  float scale = 0.0f;
  float minimum = 0.0f;
  Q4ScaleMin(scale_index, block.scales, scale, minimum);
  return HalfToFloat(block.d) * scale * quant - HalfToFloat(block.dmin) * minimum;
}

__device__ __forceinline__ float Q6Value(const BlockQ6K& block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int segment = within / 32;
  const int lane = within % 32;
  const int ql_base = half * 64;
  const std::uint8_t qh = block.qh[half * 32 + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }
  const int scale_index = half * 8 + lane / 16 + segment * 2;
  const float quant = static_cast<float>(static_cast<int>((high << 4U) | low) - 32);
  return HalfToFloat(block.d) * static_cast<float>(block.scales[scale_index]) * quant;
}


/// llama.cpp quantizes activations to Q8_K (one fp16 scale per 256 weights,
/// int8 values) before the K-quant dot products. Matching that rounding is what
/// closes the last logit error; see notes/engine-plan.md.
__device__ __forceinline__ void ActivationScale(const float* x, int count,
                                               float& scale) {
  float amax = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float value = fabsf(x[i]);
    amax = value > amax ? value : amax;
  }
  scale = amax > 0.0f ? amax / 127.0f : 1.0f;
}

__device__ __forceinline__ float QuantizedActivation(float value, float scale) {
  const float quantized = rintf(value / scale);
  const float clipped = fminf(127.0f, fmaxf(-127.0f, quantized));
  return clipped * scale;
}

__device__ __forceinline__ int WarpSumInt(int value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float WarpMax(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float WarpSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

template <int TYPE>
__device__ __forceinline__ float DotRow(const std::uint8_t* row, const float* x,
                                        int cols) {
  if constexpr (TYPE == kF32) {
    const float* weights = reinterpret_cast<const float*>(row);
    float sum = 0.0f;
    for (int i = threadIdx.x % 32; i < cols; i += 32) {
      sum += weights[i] * x[i];
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ8_0) {
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(row);
    const int count = cols / 32;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float scale = 1.0f;
      ActivationScale(x + b * 32, 32, scale);
      const float inverse = 1.0f / scale;
      int total = 0;
      for (int j = 0; j < 32; ++j) {
        int a = __float2int_rn(x[b * 32 + j] * inverse);
        a = a > 127 ? 127 : (a < -127 ? -127 : a);
        total += static_cast<int>(blocks[b].qs[j]) * a;
      }
      sum += HalfToFloat(blocks[b].d) * scale * static_cast<float>(total);
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ4_0) {
    const auto* blocks = reinterpret_cast<const BlockQ4_0*>(row);
    const int count = cols / 32;
    float sum = 0.0f;
    // Warp-per-row: stride by LANE, not by threadIdx, or every warp but the
    // first starts at a shifted block (that was the 0.386/0.000 bug).
    for (int b = threadIdx.x & 31; b < count; b += 32) {
      float scale = 1.0f;
      ActivationScale(x + b * 32, 32, scale);
      const float inverse = 1.0f / scale;
      int total = 0;
      for (int j = 0; j < 32; ++j) {
        int a = __float2int_rn(x[b * 32 + j] * inverse);
        a = a > 127 ? 127 : (a < -127 ? -127 : a);
        total += Q4_0Code(blocks[b], j) * a;
      }
      sum += HalfToFloat(blocks[b].d) * scale * static_cast<float>(total);
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ4K) {
    const auto* blocks = reinterpret_cast<const BlockQ4K*>(row);
    const int count = cols / 256;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float scale = 1.0f;
      ActivationScale(x + b * 256, 256, scale);
      const float d = HalfToFloat(blocks[b].d);
      const float dmin = HalfToFloat(blocks[b].dmin);
      float inner = 0.0f;
      for (int g = 0; g < 8; ++g) {
        int sum_qa = 0;
        int sum_a = 0;
        const float inverse = 1.0f / scale;
        for (int i = 0; i < 32; ++i) {
          const int j = g * 32 + i;
          int a = __float2int_rn(x[b * 256 + j] * inverse);
          a = a > 127 ? 127 : (a < -127 ? -127 : a);
          sum_qa += Q4Code(blocks[b], j) * a;
          sum_a += a;
        }
        float sc = 0.0f;
        float m = 0.0f;
        Q4ScaleMin(g, blocks[b].scales, sc, m);
        inner += (d * sc * static_cast<float>(sum_qa) - dmin * m * static_cast<float>(sum_a)) * scale;
      }
      sum += inner;
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ5K) {
    const auto* blocks = reinterpret_cast<const BlockQ5K*>(row);
    const int count = cols / 256;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float scale = 1.0f;
      ActivationScale(x + b * 256, 256, scale);
      float inner = 0.0f;
      for (int j = 0; j < 256; ++j) {
        inner += Q5Value(blocks[b], j) *
                 QuantizedActivation(x[b * 256 + j], scale);
      }
      sum += inner;
    }
    return WarpSum(sum);
  } else {
    const auto* blocks = reinterpret_cast<const BlockQ6K*>(row);
    const int count = cols / 256;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float scale = 1.0f;
      ActivationScale(x + b * 256, 256, scale);
      const float inverse = 1.0f / scale;
      const float d = HalfToFloat(blocks[b].d);
      float inner = 0.0f;
      // Segment-major, matching tests/dot_vs_gufo_test.cpp's verified form: the
      // 16-element groups never straddle a segment, so the nibble and 2-bit
      // selection hoist out of the inner loop.
      for (int g = 0; g < 16; ++g) {
        const int half = g / 8;
        const int segment = (g % 8) / 2;
        const int lane_base = (g * 16) % 32;
        const std::uint8_t* ql = blocks[b].ql + half * 64 + (segment % 2) * 32;
        const int ql_shift = (segment / 2) * 4;
        const int qh_shift = segment * 2;
        int total = 0;
        for (int i = 0; i < 16; ++i) {
          const int j = g * 16 + i;
          const int lane = lane_base + i;
          const int low = (ql[lane] >> ql_shift) & 0x0F;
          const int high = (blocks[b].qh[half * 32 + lane] >> qh_shift) & 0x03;
          const int code = ((high << 4) | low) - 32;
          int a = __float2int_rn(x[b * 256 + j] * inverse);
          a = a > 127 ? 127 : (a < -127 ? -127 : a);
          total += code * a;
        }
        inner += d * static_cast<float>(blocks[b].scales[g]) * scale *
                 static_cast<float>(total);
      }
      sum += inner;
    }
    return WarpSum(sum);
  }
}

/// Quantises one activation vector once per token for the Q4_K dot: `scales` holds a
/// scale per 256-element block and `sums` the eight per-group integer sums that the
/// Q4_K formula needs. The values are identical to what DotRow<kQ4K> computes per row,
/// so results stay bit-identical while the work stops being repeated per row.
__global__ void QuantizeActivationKernel(const float* x, int cols, std::int8_t* q,
                                         float* scales, int* sums) {
  const int blocks = cols / 256;
  const int b = blockIdx.x;
  if (b >= blocks) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const float* source = x + static_cast<std::size_t>(b) * 256;
  float amax = 0.0f;
  for (int i = lane; i < 256; i += 32) {
    const float value = fabsf(source[i]);
    amax = value > amax ? value : amax;
  }
  amax = WarpMax(amax);
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inverse = 1.0f / scale;
  for (int i = lane; i < 256; i += 32) {
    int a = __float2int_rn(source[i] * inverse);
    a = a > 127 ? 127 : (a < -127 ? -127 : a);
    q[static_cast<std::size_t>(b) * 256 + i] = static_cast<std::int8_t>(a);
  }
  if (lane == 0) {
    scales[b] = scale;
  }
  // One lane per element of the group, reduced across the warp: the sums must cover
  // all 32 values, which is what the per-row dot accumulated.
  for (int g = 0; g < 8; ++g) {
    const int value =
        static_cast<int>(q[static_cast<std::size_t>(b) * 256 + g * 32 + lane]);
    const int total = WarpSumInt(value);
    if (lane == 0) {
      sums[b * 8 + g] = total;
    }
  }
}

/// Per-slot quantisation: every selected expert has its own activation, so the MoE's
/// down matmul needs one quantised vector per slot.
__global__ void QuantizeActivationBatchKernel(const float* x, int cols, std::int8_t* q,
                                              float* scales, int* sums) {
  const int blocks = cols / 256;
  const int b = blockIdx.x;
  const int slot = blockIdx.y;
  if (b >= blocks) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const float* source = x + static_cast<std::size_t>(slot) * cols + b * 256;
  std::int8_t* target = q + static_cast<std::size_t>(slot) * cols + b * 256;
  float amax = 0.0f;
  for (int i = lane; i < 256; i += 32) {
    const float value = fabsf(source[i]);
    amax = value > amax ? value : amax;
  }
  amax = WarpMax(amax);
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inverse = 1.0f / scale;
  for (int i = lane; i < 256; i += 32) {
    int a = __float2int_rn(source[i] * inverse);
    a = a > 127 ? 127 : (a < -127 ? -127 : a);
    target[i] = static_cast<std::int8_t>(a);
  }
  const std::size_t scale_index = static_cast<std::size_t>(slot) * blocks + b;
  if (lane == 0) {
    scales[scale_index] = scale;
  }
  for (int g = 0; g < 8; ++g) {
    const int value = static_cast<int>(target[g * 32 + lane]);
    const int total = WarpSumInt(value);
    if (lane == 0) {
      sums[scale_index * 8 + g] = total;
    }
  }
}

__device__ __forceinline__ float DotRowQ4KPre(const std::uint8_t* row,
                                              const std::int8_t* q,
                                              const float* scales, const int* sums,
                                              int cols, int blocks_per_slot);

/// Q4_K row dot with the (block, group) work items spread across the warp. A 2560-wide
/// row is 10 blocks, so the block-per-lane loop left 22 of 32 lanes idle and each active
/// lane ran 8 groups serially; here every lane takes two or three of the 80 items.
__device__ __forceinline__ float DotRowQ4KPreWide(const std::uint8_t* row,
                                                  const std::int8_t* q,
                                                  const float* scales, const int* sums,
                                                  int cols) {
  const auto* blocks = reinterpret_cast<const BlockQ4K*>(row);
  const int count = cols / 256;
  const int items = count * 8;
  const int lane = threadIdx.x & 31;
  float local = 0.0f;
  for (int item = lane; item < items; item += 32) {
    const int b = item >> 3;
    const int g = item & 7;
    const float d = HalfToFloat(blocks[b].d);
    const float dmin = HalfToFloat(blocks[b].dmin);
    int sum_qa = 0;
    for (int i = 0; i < 32; ++i) {
      const int j = g * 32 + i;
      sum_qa += Q4Code(blocks[b], j) *
                static_cast<int>(q[static_cast<std::size_t>(b) * 256 + j]);
    }
    float sc = 0.0f;
    float m = 0.0f;
    Q4ScaleMin(g, blocks[b].scales, sc, m);
    local += (d * sc * static_cast<float>(sum_qa) -
              dmin * m * static_cast<float>(sums[b * 8 + g])) *
             scales[b];
  }
  return WarpSum(local);
}

/// Slot-aware wide dot: each selected expert's own quantised activation, items spread
/// across the warp like DotRowQ4KPreWide.
__global__ void GemvExpertsQ4KPreSlotWideKernel(const std::uint8_t* weights,
                                                const std::int32_t* ids,
                                                const std::int8_t* q, const float* scales,
                                                const int* sums, float* y,
                                                int rows_per_expert, int cols,
                                                int row_bytes, int experts,
                                                int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= experts * rows_per_expert) {
    return;
  }
  const int expert_slot = warp / rows_per_expert;
  const int row_in_expert = warp - expert_slot * rows_per_expert;
  const int expert = ids[expert_slot];
  const std::uint8_t* row = weights + (static_cast<std::size_t>(expert) * rows_per_expert +
                                      row_in_expert) * row_bytes;
  const int blocks_per_slot = cols / 256;
  const auto* slot_q = q + static_cast<std::size_t>(expert_slot) * cols;
  const auto* slot_scales =
      scales + static_cast<std::size_t>(expert_slot) * blocks_per_slot;
  const auto* slot_sums = sums + static_cast<std::size_t>(expert_slot) * blocks_per_slot * 8;
  const float sum = DotRowQ4KPreWide(row, slot_q, slot_scales, slot_sums, cols);
  if ((threadIdx.x & 31) == 0) {
    y[static_cast<std::size_t>(expert_slot) * out_stride + row_in_expert] = sum;
  }
}

/// One warp per (expert, row) with the wide dot (single activation shared by slots).
__global__ void GemvExpertsQ4KPreWideKernel(const std::uint8_t* weights,
                                            const std::int32_t* ids, const std::int8_t* q,
                                            const float* scales, const int* sums, float* y,
                                            int rows_per_expert, int cols, int row_bytes,
                                            int experts, int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= experts * rows_per_expert) {
    return;
  }
  const int expert_slot = warp / rows_per_expert;
  const int row_in_expert = warp - expert_slot * rows_per_expert;
  const int expert = ids[expert_slot];
  const std::uint8_t* row = weights + (static_cast<std::size_t>(expert) * rows_per_expert +
                                      row_in_expert) * row_bytes;
  const float sum = DotRowQ4KPreWide(row, q, scales, sums, cols);
  if ((threadIdx.x & 31) == 0) {
    y[static_cast<std::size_t>(expert_slot) * out_stride + row_in_expert] = sum;
  }
}

/// One warp per (expert, row), each slot reading its own quantised activation.
__global__ void GemvExpertsQ4KPreSlotKernel(const std::uint8_t* weights,
                                            const std::int32_t* ids, const std::int8_t* q,
                                            const float* scales, const int* sums, float* y,
                                            int rows_per_expert, int cols, int row_bytes,
                                            int experts, int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= experts * rows_per_expert) {
    return;
  }
  const int expert_slot = warp / rows_per_expert;
  const int row_in_expert = warp - expert_slot * rows_per_expert;
  const int expert = ids[expert_slot];
  const std::uint8_t* row = weights + (static_cast<std::size_t>(expert) * rows_per_expert +
                                      row_in_expert) * row_bytes;
  const int blocks_per_slot = cols / 256;
  const auto* slot_q = q + static_cast<std::size_t>(expert_slot) * cols;
  const auto* slot_scales = scales + static_cast<std::size_t>(expert_slot) * blocks_per_slot;
  const auto* slot_sums = sums + static_cast<std::size_t>(expert_slot) * blocks_per_slot * 8;
  const float sum = DotRowQ4KPre(row, slot_q, slot_scales, slot_sums, cols,
                                 blocks_per_slot);
  if ((threadIdx.x & 31) == 0) {
    y[static_cast<std::size_t>(expert_slot) * out_stride + row_in_expert] = sum;
  }
}

/// The Q4_K row dot against a pre-quantised activation (see the kernel above).
__device__ __forceinline__ float DotRowQ4KPre(const std::uint8_t* row,
                                              const std::int8_t* q,
                                              const float* scales, const int* sums,
                                              int cols, int blocks_per_slot) {
  const auto* blocks = reinterpret_cast<const BlockQ4K*>(row);
  const int count = cols / 256;
  float sum = 0.0f;
  for (int b = threadIdx.x & 31; b < count; b += 32) {
    const float d = HalfToFloat(blocks[b].d);
    const float dmin = HalfToFloat(blocks[b].dmin);
    const float scale = scales[b];
    const int sums_base = (blocks_per_slot >= 0 ? b : b) * 8;
    float inner = 0.0f;
    for (int g = 0; g < 8; ++g) {
      int sum_qa = 0;
      for (int i = 0; i < 32; ++i) {
        const int j = g * 32 + i;
        sum_qa += Q4Code(blocks[b], j) *
                  static_cast<int>(q[static_cast<std::size_t>(b) * 256 + j]);
      }
      float sc = 0.0f;
      float m = 0.0f;
      Q4ScaleMin(g, blocks[b].scales, sc, m);
      inner += (d * sc * static_cast<float>(sum_qa) -
                dmin * m * static_cast<float>(sums[sums_base + g])) *
               scale;
    }
    sum += inner;
  }
  return WarpSum(sum);
}

/// One warp per (expert, row) against a pre-quantised activation.
__global__ void GemvExpertsQ4KPreKernel(const std::uint8_t* weights,
                                        const std::int32_t* ids, const std::int8_t* q,
                                        const float* scales, const int* sums, float* y,
                                        int rows_per_expert, int cols, int row_bytes,
                                        int experts, int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= experts * rows_per_expert) {
    return;
  }
  const int expert_slot = warp / rows_per_expert;
  const int row_in_expert = warp - expert_slot * rows_per_expert;
  const int expert = ids[expert_slot];
  const std::uint8_t* row = weights + (static_cast<std::size_t>(expert) * rows_per_expert +
                                      row_in_expert) * row_bytes;
  const float sum = DotRowQ4KPre(row, q, scales, sums, cols, 0);
  if ((threadIdx.x & 31) == 0) {
    y[static_cast<std::size_t>(expert_slot) * out_stride + row_in_expert] = sum;
  }
}

template <int TYPE>
__global__ void GemvKernel(const std::uint8_t* weights, const float* x,
                           float* y, int rows, int cols, int row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= rows) {
    return;
  }
  const std::uint8_t* row = weights + static_cast<std::size_t>(warp) * row_bytes;
  const float sum = DotRow<TYPE>(row, x, cols);
  if (lane == 0) {
    y[warp] = sum;
  }
}

template <int TYPE>
__global__ void GemvAddKernel(const std::uint8_t* weights, const float* x,
                              const float* bias, float* y, int rows, int cols,
                              int row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= rows) {
    return;
  }
  const std::uint8_t* row = weights + static_cast<std::size_t>(warp) * row_bytes;
  const float sum = DotRow<TYPE>(row, x, cols);
  if (lane == 0) {
    y[warp] = sum + (bias == nullptr ? 0.0f : bias[warp]);
  }
}

/// One warp per (expert, row). The row pointer comes from the device-side id
/// list, so every selected expert is covered by a single launch. Each warp owns
/// its own row, so lane 0 writes DotRow's warp-reduced result directly -- a
/// block-wide sum would mix eight different rows.
template <int TYPE>
__global__ void GemvExpertsKernel(const std::uint8_t* weights,
                                  const std::int32_t* ids, const float* x,
                                  float* y, int rows_per_expert, int cols,
                                  int row_bytes, int experts, int batch_stride,
                                  int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= experts * rows_per_expert) {
    return;
  }
  const int expert_slot = warp / rows_per_expert;
  const int row_in_expert = warp - expert_slot * rows_per_expert;
  const int expert = ids[expert_slot];
  const std::uint8_t* row =
      weights + (static_cast<std::size_t>(expert) * rows_per_expert +
                 row_in_expert) * row_bytes;
  const float* source = x + static_cast<std::size_t>(expert_slot) * batch_stride;
  const float sum = DotRow<TYPE>(row, source, cols);
  if ((threadIdx.x & 31) == 0) {
    y[static_cast<std::size_t>(expert_slot) * out_stride + row_in_expert] = sum;
  }
}

/// Batched form of the quantized dot: `out[m][row] = dot(W[row], X[m])` for up
/// to `kMaxBatch` tokens. One block per row, so each weight block is read once
/// and reused across the batch -- that is what makes prefill pay, because the
/// weights stop being read once per token.
constexpr int kMaxGemmBatch = 8;

// Prepared activations for the batched K-quant GEMM: one int8 value and one
// 256-block scale per token. Quantizing inside the weight loop cost 5x.
__device__ signed char g_gemm_q[8 * 8192];
__device__ float g_gemm_s[8 * 32];

__global__ void QuantizeGemmActivationsKernel(const float* x, int batch, int cols,
                                              int x_stride) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;  // 256-block index
  const int m = blockIdx.y;
  if (m >= batch || b >= cols / 256) {
    return;
  }
  const float* row = x + static_cast<std::size_t>(m) * x_stride + b * 256;
  float amax = 0.0f;
  for (int j = 0; j < 256; ++j) {
    const float v = fabsf(row[j]);
    amax = v > amax ? v : amax;
  }
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inverse = 1.0f / scale;
  g_gemm_s[m * 32 + b] = scale;
  for (int j = 0; j < 256; ++j) {
    int a = __float2int_rn(row[j] * inverse);
    a = a > 127 ? 127 : (a < -127 ? -127 : a);
    g_gemm_q[m * 8192 + b * 256 + j] = static_cast<signed char>(a);
  }
}

template <int TYPE>
__global__ void GemmKernel(const std::uint8_t* weights, const float* x, float* y,
                           int rows, int cols, int row_bytes, int batch,
                           int x_stride, int y_stride) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const std::uint8_t* weight_row =
      weights + static_cast<std::size_t>(row) * row_bytes;
  float accumulator[kMaxGemmBatch];
  for (int m = 0; m < batch; ++m) {
    accumulator[m] = 0.0f;
  }
  if constexpr (TYPE == kF32) {
    const float* values = reinterpret_cast<const float*>(weight_row);
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
      const float weight = values[k];
      for (int m = 0; m < batch; ++m) {
        accumulator[m] += weight * x[m * x_stride + k];
      }
    }
  } else if constexpr (TYPE == kQ8_0) {
    // Spread *elements* across the block, not 32-weight blocks: with one thread
    // per block only 80 of 256 threads worked, so per-thread work grew with the
    // batch and the weight read never amortised (measured 8.8x for batch 8).
    // Quantise the activation per 32-value block exactly as DotRow<kQ8_0> does (and as
    // llama.cpp does): dequantising the weight and multiplying by an f32 activation made
    // this branch disagree with the per-token path by ~0.5 % per matvec.
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(weight_row);
    for (int b = threadIdx.x; b < cols / 32; b += blockDim.x) {
      const float weight_d = HalfToFloat(blocks[b].d);
      for (int m = 0; m < batch; ++m) {
        const float* row = x + static_cast<std::size_t>(m) * x_stride + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
          amax = fmaxf(amax, fabsf(row[i]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inverse = 1.0f / scale;
        int total = 0;
        for (int i = 0; i < 32; ++i) {
          int a = __float2int_rn(row[i] * inverse);
          a = a > 127 ? 127 : (a < -127 ? -127 : a);
          total += static_cast<int>(blocks[b].qs[i]) * a;
        }
        accumulator[m] += weight_d * scale * static_cast<float>(total);
      }
    }
  } else if constexpr (TYPE == kQ4K) {
    // Same arithmetic as the per-row Q4_K dot: per 32-value group, the weight is
    // d*sc*code - dmin*m, and the activations are quantized per 256-block.
    const auto* blocks = reinterpret_cast<const BlockQ4K*>(weight_row);
    for (int b = threadIdx.x; b < cols / 256; b += blockDim.x) {
      const float d = HalfToFloat(blocks[b].d);
      const float dmin = HalfToFloat(blocks[b].dmin);
      for (int g = 0; g < 8; ++g) {
        float sc = 0.0f;
        float m_min = 0.0f;
        Q4ScaleMin(g, blocks[b].scales, sc, m_min);
        const float weight_scale = d * sc;
        const float weight_min = dmin * m_min;
        for (int i = 0; i < 32; ++i) {
          const int j = b * 256 + g * 32 + i;
          const int code = Q4Code(blocks[b], g * 32 + i);
          const float weight = weight_scale * static_cast<float>(code) - weight_min;
          for (int m = 0; m < batch; ++m) {
            accumulator[m] += weight *
                static_cast<float>(static_cast<int>(g_gemm_q[m * 8192 + j]));
          }
        }
      }
      for (int m = 0; m < batch; ++m) {
        accumulator[m] *= g_gemm_s[m * 32 + b];
      }
    }
  } else if constexpr (TYPE == kQ6K) {
    const auto* blocks = reinterpret_cast<const BlockQ6K*>(weight_row);
    for (int b = threadIdx.x; b < cols / 256; b += blockDim.x) {
      const float d = HalfToFloat(blocks[b].d);
      for (int g = 0; g < 16; ++g) {
        const int half = g / 8;
        const int segment = (g % 8) / 2;
        const int lane_base = (g * 16) % 32;
        const std::uint8_t* ql = blocks[b].ql + half * 64 + (segment % 2) * 32;
        const int ql_shift = (segment / 2) * 4;
        const int qh_shift = segment * 2;
        const float group_scale = d * static_cast<float>(blocks[b].scales[g]);
        for (int i = 0; i < 16; ++i) {
          const int j = b * 256 + g * 16 + i;
          const int lane = lane_base + i;
          const int low = (ql[lane] >> ql_shift) & 0x0F;
          const int high = (blocks[b].qh[half * 32 + lane] >> qh_shift) & 0x03;
          const int code = ((high << 4) | low) - 32;
          const float weight = group_scale * static_cast<float>(code);
          for (int m = 0; m < batch; ++m) {
            accumulator[m] += weight *
                static_cast<float>(static_cast<int>(g_gemm_q[m * 8192 + j]));
          }
        }
      }
      for (int m = 0; m < batch; ++m) {
        accumulator[m] *= g_gemm_s[m * 32 + b];
      }
    }
  } else {
    // K-quants go through the dequantising value functions, which are already
    // validated against Gufo's reference.
    for (int b = threadIdx.x; b < cols / 256; b += blockDim.x) {
      const auto* block = reinterpret_cast<const BlockQ4K*>(weight_row) + b;
      for (int j = 0; j < 256; ++j) {
        const float weight = TYPE == kQ4K
                                 ? Q4Value(*reinterpret_cast<const BlockQ4K*>(
                                       reinterpret_cast<const BlockQ4K*>(weight_row) + b),
                                           j)
                                 : 0.0f;
        const int k = b * 256 + j;
        for (int m = 0; m < batch; ++m) {
          accumulator[m] += weight * x[m * x_stride + k];
        }
      }
      (void)block;
    }
  }
  // Deterministic reduction: per-warp partials, then a fixed-order sum by lane 0
  // of each warp, which writes its own token's result.
  __shared__ float warp_sums[kMaxGemmBatch][8];
  for (int m = 0; m < batch; ++m) {
    float value = accumulator[m];
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down(value, offset);
    }
    if ((threadIdx.x & 31) == 0) {
      warp_sums[m][threadIdx.x >> 5] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int m = 0; m < batch; ++m) {
      float total = 0.0f;
      for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) {
        total += warp_sums[m][w];
      }
      y[m * y_stride + row] = total;
    }
  }
}

/// Quantizes activations in 256-wide blocks (the Q8_K shape llama.cpp uses for
/// K-quants) into caller-owned buffers, for the batched/prepared-activation
/// paths. Each thread owns one block, so it also owns that block's scale.
__global__ void QuantizeActivationsKernel(const float* x, int blocks,
                                          signed char* values, float* scales) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) {
    return;
  }
  float amax = 0.0f;
  for (int j = 0; j < 256; ++j) {
    const float v = fabsf(x[b * 256 + j]);
    amax = v > amax ? v : amax;
  }
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inverse = 1.0f / scale;
  scales[b] = scale;
  for (int j = 0; j < 256; ++j) {
    int a = __float2int_rn(x[b * 256 + j] * inverse);
    a = a > 127 ? 127 : (a < -127 ? -127 : a);
    values[b * 256 + j] = static_cast<signed char>(a);
  }
}

/// K-quant GEMV against activations that were quantized once by
/// LaunchQuantizeActivations. Must reproduce the per-row path exactly (both use
/// the same activation rounding); the operator suite asserts that.
template <int TYPE>
__global__ void GemvPreparedKernel(const std::uint8_t* weights,
                                   const signed char* values, const float* scales,
                                   float* y, int rows, int cols, int row_bytes) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= rows) {
    return;
  }
  const std::uint8_t* row = weights + static_cast<std::size_t>(warp) * row_bytes;
  float sum = 0.0f;
  if constexpr (TYPE == kQ4K) {
    const auto* blocks = reinterpret_cast<const BlockQ4K*>(row);
    for (int b = threadIdx.x & 31; b < cols / 256; b += 32) {
      const float d = HalfToFloat(blocks[b].d);
      const float dmin = HalfToFloat(blocks[b].dmin);
      float inner = 0.0f;
      for (int g = 0; g < 8; ++g) {
        int sum_qa = 0;
        int sum_a = 0;
        for (int i = 0; i < 32; ++i) {
          const int j = g * 32 + i;
          const int a = static_cast<int>(values[b * 256 + j]);
          sum_qa += Q4Code(blocks[b], j) * a;
          sum_a += a;
        }
        float sc = 0.0f;
        float m = 0.0f;
        Q4ScaleMin(g, blocks[b].scales, sc, m);
        inner += d * sc * static_cast<float>(sum_qa) -
                 dmin * m * static_cast<float>(sum_a);
      }
      sum += inner * scales[b];
    }
  } else if constexpr (TYPE == kQ6K) {
    const auto* blocks = reinterpret_cast<const BlockQ6K*>(row);
    for (int b = threadIdx.x & 31; b < cols / 256; b += 32) {
      const float d = HalfToFloat(blocks[b].d);
      float inner = 0.0f;
      for (int g = 0; g < 16; ++g) {
        const int half = g / 8;
        const int segment = (g % 8) / 2;
        const int lane_base = (g * 16) % 32;
        const std::uint8_t* ql = blocks[b].ql + half * 64 + (segment % 2) * 32;
        const int ql_shift = (segment / 2) * 4;
        const int qh_shift = segment * 2;
        int total = 0;
        for (int i = 0; i < 16; ++i) {
          const int j = g * 16 + i;
          const int lane = lane_base + i;
          const int low = (ql[lane] >> ql_shift) & 0x0F;
          const int high = (blocks[b].qh[half * 32 + lane] >> qh_shift) & 0x03;
          total += (((high << 4) | low) - 32) * static_cast<int>(values[b * 256 + j]);
        }
        inner += d * static_cast<float>(blocks[b].scales[g]) *
                 static_cast<float>(total);
      }
      sum += inner * scales[b];
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    y[warp] = sum;
  }
}

__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              int rows, int cols, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* source = x + static_cast<std::size_t>(row) * cols;
  float* target = y + static_cast<std::size_t>(row) * cols;
  // Deterministic reduction: per-warp sums in shared memory, then a fixed-order
  // sum by thread 0. atomicAdd made the result depend on warp arrival order,
  // which flipped tokens at 0.02 logit margins (see notes/engine-plan.md).
  __shared__ float warp_sums[32];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += source[i] * source[i];
  }
  sum = WarpSum(sum);
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>((blockDim.x + 31) / 32); ++w) {
    total += warp_sums[w];
  }
  const float scale = rsqrtf(total / static_cast<float>(cols) + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    target[i] = source[i] * scale * weight[i];
  }
}

__global__ void SiluKernel(const float* x, float* y, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void SiluMulKernel(const float* x, const float* gate, float* y,
                              std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    y[i] = (v / (1.0f + expf(-v))) * (1.0f / (1.0f + expf(-gate[i])));
  }
}

constexpr int kThreads = 256;

}  // namespace



std::size_t RowBytes(StorageType type, std::size_t cols) {
  switch (type) {
    case kF32:
      return cols * sizeof(float);
    case kQ8_0:
      return cols / 32 * sizeof(BlockQ8_0);
    case kQ4K:
      return cols / 256 * sizeof(BlockQ4K);
    case kQ4_0:
      return cols / 32 * sizeof(BlockQ4_0);
    case kQ5K:
      return cols / 256 * sizeof(BlockQ5K);
    case kQ6K:
      return cols / 256 * sizeof(BlockQ6K);
  }
  return 0;
}

void LaunchGemv(const void* weights, StorageType type, std::size_t rows,
                std::size_t cols, const float* x, float* y,
                hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int warps = static_cast<int>(rows);
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const auto* data = static_cast<const std::uint8_t*>(weights);
  switch (type) {
    case kF32:
      GemvKernel<kF32><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ8_0:
      GemvKernel<kQ8_0><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ4K:
      GemvKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ4_0:
      GemvKernel<kQ4_0><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ5K:
      GemvKernel<kQ5K><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ6K:
      GemvKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
  }
}

void LaunchGemvAdd(const void* weights, StorageType type, std::size_t rows,
                   std::size_t cols, const float* x, const float* bias, float* y,
                   hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int warps = static_cast<int>(rows);
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const auto* data = static_cast<const std::uint8_t*>(weights);
  switch (type) {
    case kF32:
      GemvAddKernel<kF32><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ8_0:
      GemvAddKernel<kQ8_0><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ4K:
      GemvAddKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ5K:
      GemvAddKernel<kQ5K><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ6K:
      GemvAddKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
  }
}

void LaunchQuantizeActivation(const float* x, std::size_t cols, std::int8_t* q,
                              float* scales, int* sums, hipStream_t stream) {
  const int blocks = static_cast<int>(cols / 256);
  if (blocks == 0) {
    return;
  }
  QuantizeActivationKernel<<<blocks, 32, 0, stream>>>(x, static_cast<int>(cols), q,
                                                      scales, sums);
}

void LaunchGemvExpertsPreSlotsWide(const void* weights, StorageType type,
                                   const std::int32_t* ids, std::size_t experts,
                                   std::size_t rows_per_expert, std::size_t cols,
                                   const float* x, const std::int8_t* q,
                                   const float* scales, const int* sums, float* y,
                                   std::size_t out_stride, hipStream_t stream) {
  if (type != kQ4K) {
    LaunchGemvExperts(weights, type, ids, experts, rows_per_expert, cols, x,
                      /*batch_stride=*/cols, y, out_stride, stream);
    return;
  }
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int total_rows = static_cast<int>(experts * rows_per_expert);
  const int blocks = (total_rows + kThreads / 32 - 1) / (kThreads / 32);
  GemvExpertsQ4KPreSlotWideKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(weights), ids, q, scales, sums, y,
      static_cast<int>(rows_per_expert), static_cast<int>(cols), row_bytes,
      static_cast<int>(experts), static_cast<int>(out_stride));
}

void LaunchQuantizeActivationBatch(const float* x, std::size_t cols, std::size_t slots,
                                   std::int8_t* q, float* scales, int* sums,
                                   hipStream_t stream) {
  const int blocks = static_cast<int>(cols / 256);
  if (blocks == 0 || slots == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(slots));
  QuantizeActivationBatchKernel<<<grid, 32, 0, stream>>>(x, static_cast<int>(cols), q,
                                                         scales, sums);
}

void LaunchGemvExpertsPreSlots(const void* weights, StorageType type,
                               const std::int32_t* ids, std::size_t experts,
                               std::size_t rows_per_expert, std::size_t cols,
                               const float* x, const std::int8_t* q, const float* scales,
                               const int* sums, float* y, std::size_t out_stride,
                               hipStream_t stream) {
  if (type != kQ4K) {
    LaunchGemvExperts(weights, type, ids, experts, rows_per_expert, cols, x,
                      /*batch_stride=*/cols, y, out_stride, stream);
    return;
  }
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int total_rows = static_cast<int>(experts * rows_per_expert);
  const int blocks = (total_rows + kThreads / 32 - 1) / (kThreads / 32);
  GemvExpertsQ4KPreSlotKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(weights), ids, q, scales, sums, y,
      static_cast<int>(rows_per_expert), static_cast<int>(cols), row_bytes,
      static_cast<int>(experts), static_cast<int>(out_stride));
}

void LaunchGemvExpertsPreWide(const void* weights, StorageType type,
                              const std::int32_t* ids, std::size_t experts,
                              std::size_t rows_per_expert, std::size_t cols,
                              const float* x, const std::int8_t* q, const float* scales,
                              const int* sums, float* y, std::size_t out_stride,
                              hipStream_t stream) {
  if (type != kQ4K) {
    LaunchGemvExperts(weights, type, ids, experts, rows_per_expert, cols, x,
                      /*batch_stride=*/0, y, out_stride, stream);
    return;
  }
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int total_rows = static_cast<int>(experts * rows_per_expert);
  const int blocks = (total_rows + kThreads / 32 - 1) / (kThreads / 32);
  GemvExpertsQ4KPreWideKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(weights), ids, q, scales, sums, y,
      static_cast<int>(rows_per_expert), static_cast<int>(cols), row_bytes,
      static_cast<int>(experts), static_cast<int>(out_stride));
}

void LaunchGemvExpertsPre(const void* weights, StorageType type,
                          const std::int32_t* ids, std::size_t experts,
                          std::size_t rows_per_expert, std::size_t cols, const float* x,
                          const std::int8_t* q, const float* scales, const int* sums,
                          float* y, std::size_t out_stride, hipStream_t stream) {
  if (type != kQ4K) {
    // Only Q4_K has the pre-quantised path; other storages keep the float activation.
    LaunchGemvExperts(weights, type, ids, experts, rows_per_expert, cols, x,
                      /*batch_stride=*/0, y, out_stride, stream);
    return;
  }
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int total_rows = static_cast<int>(experts * rows_per_expert);
  const int blocks = (total_rows + kThreads / 32 - 1) / (kThreads / 32);
  GemvExpertsQ4KPreKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(weights), ids, q, scales, sums, y,
      static_cast<int>(rows_per_expert), static_cast<int>(cols), row_bytes,
      static_cast<int>(experts), static_cast<int>(out_stride));
}

void LaunchGemvExperts(const void* weights, StorageType type,
                       const std::int32_t* ids, std::size_t experts,
                       std::size_t rows_per_expert, std::size_t cols,
                       const float* x, std::size_t batch_stride, float* y,
                       std::size_t out_stride, hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int total_rows = static_cast<int>(experts * rows_per_expert);
  const int blocks = (total_rows + kThreads / 32 - 1) / (kThreads / 32);
  const auto* data = static_cast<const std::uint8_t*>(weights);
  const int rows_int = static_cast<int>(rows_per_expert);
  const int cols_int = static_cast<int>(cols);
  const int experts_int = static_cast<int>(experts);
  const int batch_int = static_cast<int>(batch_stride);
  const int out_int = static_cast<int>(out_stride);
  switch (type) {
    case kF32:
      GemvExpertsKernel<kF32><<<blocks, kThreads, 0, stream>>>(
          data, ids, x, y, rows_int, cols_int, row_bytes, experts_int, batch_int, out_int);
      break;
    case kQ8_0:
      GemvExpertsKernel<kQ8_0><<<blocks, kThreads, 0, stream>>>(
          data, ids, x, y, rows_int, cols_int, row_bytes, experts_int, batch_int, out_int);
      break;
    case kQ4K:
      GemvExpertsKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
          data, ids, x, y, rows_int, cols_int, row_bytes, experts_int, batch_int, out_int);
      break;
    case kQ5K:
      GemvExpertsKernel<kQ5K><<<blocks, kThreads, 0, stream>>>(
          data, ids, x, y, rows_int, cols_int, row_bytes, experts_int, batch_int, out_int);
      break;
    case kQ6K:
      GemvExpertsKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
          data, ids, x, y, rows_int, cols_int, row_bytes, experts_int, batch_int, out_int);
      break;
  }
}

void LaunchGemm(const void* weights, StorageType type, std::size_t rows,
                std::size_t cols, const float* x_block, std::size_t batch,
                std::size_t x_stride, float* y_block, std::size_t y_stride,
                hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const auto* data = static_cast<const std::uint8_t*>(weights);
  const int rows_int = static_cast<int>(rows);
  const int cols_int = static_cast<int>(cols);
  const int x_int = static_cast<int>(x_stride);
  const int y_int = static_cast<int>(y_stride);
  // The kernel keeps one accumulator per token in registers, so a large chunk is
  // issued as several slices of at most kMaxGemmBatch tokens.
  const float* x = x_block;
  float* y = y_block;
  for (std::size_t offset = 0; offset < batch; offset += kMaxGemmBatch) {
    const std::size_t slice = batch - offset < kMaxGemmBatch ? batch - offset : kMaxGemmBatch;
    const int batch_int = static_cast<int>(slice);
    x = x_block + offset * x_stride;
    y = y_block + offset * y_stride;
    if (type == kQ4K || type == kQ6K) {
      // Quantize this tile immediately before its GEMM: the prepared buffers hold
      // kMaxGemmBatch tokens, and same-stream ordering makes the dependency hold
      // without a synchronisation.
      const dim3 grid(static_cast<unsigned>((cols / 256 + 31) / 32),
                      static_cast<unsigned>(slice));
      QuantizeGemmActivationsKernel<<<grid, 32, 0, stream>>>(
          x, batch_int, static_cast<int>(cols), static_cast<int>(x_stride));
    }
  switch (type) {
    case kF32:
      GemmKernel<kF32><<<rows_int, kThreads, 0, stream>>>(
          data, x, y, rows_int, cols_int, row_bytes, batch_int, x_int, y_int);
      break;
    case kQ8_0:
      GemmKernel<kQ8_0><<<rows_int, kThreads, 0, stream>>>(
          data, x, y, rows_int, cols_int, row_bytes, batch_int, x_int, y_int);
      break;
    case kQ4K:
      GemmKernel<kQ4K><<<rows_int, kThreads, 0, stream>>>(
          data, x, y, rows_int, cols_int, row_bytes, batch_int, x_int, y_int);
      break;
    case kQ6K:
      GemmKernel<kQ6K><<<rows_int, kThreads, 0, stream>>>(
          data, x, y, rows_int, cols_int, row_bytes, batch_int, x_int, y_int);
      break;
    default:
      // No batched kernel for this storage: run the batch as per-token GEMVs, which every type
      // supports. Correct for any type, slower than a GEMM, and better than silently producing
      // nothing for a tensor whose type the batched path does not know.
      for (int m = 0; m < batch_int; ++m) {
        LaunchGemv(data, type, rows, cols, x + static_cast<std::size_t>(m) * x_int,
                   y + static_cast<std::size_t>(m) * y_int, stream);
      }
      break;
  }
  }
}

void LaunchQuantizeActivations(const float* x, std::size_t cols,
                               signed char* values, float* scales,
                               hipStream_t stream) {
  const int blocks = static_cast<int>(cols / 256);
  QuantizeActivationsKernel<<<(blocks + 31) / 32, 32, 0, stream>>>(
      x, blocks, values, scales);
}

void LaunchGemvPrepared(const void* weights, StorageType type, std::size_t rows,
                        std::size_t cols, const signed char* values,
                        const float* scales, float* y, hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int warps = static_cast<int>(rows);
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const auto* data = static_cast<const std::uint8_t*>(weights);
  if (type == kQ4K) {
    GemvPreparedKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
        data, values, scales, y, static_cast<int>(rows), static_cast<int>(cols),
        row_bytes);
  } else if (type == kQ6K) {
    GemvPreparedKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
        data, values, scales, y, static_cast<int>(rows), static_cast<int>(cols),
        row_bytes);
  }
}

void LaunchRmsNorm(const float* x, const float* weight, float* y,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream) {
  RmsNormKernel<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      x, weight, y, static_cast<int>(rows), static_cast<int>(cols), eps);
}

void LaunchSilu(const float* x, float* y, std::size_t count, hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  SiluKernel<<<blocks, kThreads, 0, stream>>>(x, y, count);
}

void LaunchSiluMul(const float* x, const float* gate, float* y, std::size_t count,
                   hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  SiluMulKernel<<<blocks, kThreads, 0, stream>>>(x, gate, y, count);
}

}  // namespace gufo::ext::ling3_air::hip
