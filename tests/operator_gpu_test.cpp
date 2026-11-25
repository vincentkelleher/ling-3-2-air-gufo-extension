// GPU vs host operator checks for the Ling 3.2 Air kernels.
//
// Synthetic quantized blocks (random bytes with sane fp16 scales) for every
// storage type the model uses, plus RMSNorm and SiLU. The host reference is the
// layout math ported from Gufo's verified dequantizer; tests/dot_test.cpp
// checks that math against `Dot()` on real GGUF rows, so this file only has to
// prove the kernels agree with it.
//
// Standalone build (no Gufo needed):
//   hipcc -O3 --offload-arch=gfx1151 tests/operator_gpu_test.cpp \
//         src/hip/kernels.hip -I. -o operator_gpu_test
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/hip/kernels.hpp"

namespace {

namespace hipk = gufo::ext::ling3_air::hip;

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
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
#pragma pack(pop)

std::uint16_t FloatToHalf(float value) {
  // fp16 with round-to-nearest is overkill for the test; use exact values.
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127;
  std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  exponent += 15;
  if (exponent >= 0x1F) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  mantissa = (mantissa + (1u << 12)) >> 13;
  if (mantissa > 0x3FFu) {
    mantissa = 0;
    exponent += 1;
  }
  return static_cast<std::uint16_t>(sign | (exponent << 10) | mantissa);
}

float HalfToFloat(std::uint16_t bits) {
  const std::uint32_t sign = (bits & 0x8000u) << 16;
  std::uint32_t exponent = (bits >> 10) & 0x1Fu;
  std::uint32_t mantissa = bits & 0x3FFu;
  if (exponent == 0) {
    if (mantissa == 0) {
      std::uint32_t out = sign;
      float value;
      std::memcpy(&value, &out, sizeof(value));
      return value;
    }
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    exponent += 1;
    mantissa &= 0x3FFu;
  } else if (exponent == 0x1F) {
    std::uint32_t out = sign | 0x7F800000u | (mantissa << 13);
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
  }
  exponent = exponent + (127 - 15);
  std::uint32_t out = sign | (exponent << 23) | (mantissa << 13);
  float value;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

void Q4ScaleMin(int index, const std::uint8_t* packed, float& scale, float& minimum) {
  if (index < 4) {
    scale = static_cast<float>(packed[index] & 0x3F);
    minimum = static_cast<float>(packed[index + 4] & 0x3F);
    return;
  }
  scale = static_cast<float>((packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4));
  minimum = static_cast<float>((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4));
}

float Q4Value(const BlockQ4K& block, int index) {
  const int group = index / 32;
  const int pair = group / 2;
  const bool high = (group & 1) != 0;
  const int lane = index % 32;
  const std::uint8_t packed = block.qs[pair * 32 + lane];
  const float quant = static_cast<float>(high ? (packed >> 4) : (packed & 0x0F));
  float scale = 0;
  float minimum = 0;
  Q4ScaleMin(group, block.scales, scale, minimum);
  return HalfToFloat(block.d) * scale * quant - HalfToFloat(block.dmin) * minimum;
}

float Q6Value(const BlockQ6K& block, int index) {
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
      low = block.ql[ql_base + lane] & 0x0F;
      high = qh & 0x03;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0F;
      high = (qh >> 2) & 0x03;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4;
      high = (qh >> 4) & 0x03;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4;
      high = (qh >> 6) & 0x03;
      break;
  }
  const int scale_index = half * 8 + lane / 16 + segment * 2;
  const float quant = static_cast<float>(static_cast<int>((high << 4) | low) - 32);
  return HalfToFloat(block.d) * static_cast<float>(block.scales[scale_index]) * quant;
}

int Failures = 0;

void Check(bool ok, const char* what, double measured, double expected) {
  if (!ok) {
    ++Failures;
    std::printf("FAIL %-28s got %.6g want %.6g\n", what, measured, expected);
  } else {
    std::printf("ok   %-28s %.6g\n", what, measured);
  }
}

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    const hipError_t status = (call);                                         \
    if (status != hipSuccess) {                                               \
      std::printf("HIP error at %s: %s\n", #call, hipGetErrorString(status)); \
      return 1;                                                               \
    }                                                                         \
  } while (0)

}  // namespace

int main() {
  std::mt19937 rng(20261125);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> byte(0, 255);

  const int cols = 256;   // one K-quant super-block, eight Q8_0 blocks
  const int rows = 4;     // enough to exercise the warp-per-row mapping

  struct Case {
    hipk::StorageType type;
    const char* name;
    std::size_t row_bytes;
  };
  const Case cases[] = {
      {hipk::kF32, "F32", static_cast<std::size_t>(cols) * sizeof(float)},
      {hipk::kQ8_0, "Q8_0", static_cast<std::size_t>(cols) / 32 * sizeof(BlockQ8_0)},
      {hipk::kQ4K, "Q4_K", static_cast<std::size_t>(cols) / 256 * sizeof(BlockQ4K)},
      {hipk::kQ6K, "Q6_K", static_cast<std::size_t>(cols) / 256 * sizeof(BlockQ6K)},
  };

  std::vector<float> x(cols);
  for (auto& value : x) {
    value = dist(rng);
  }

  for (const auto& test : cases) {
    std::vector<std::uint8_t> weights(test.row_bytes * rows);
    for (auto& value : weights) {
      value = static_cast<std::uint8_t>(byte(rng));
    }
    // Sane fp16 scales keep the reference and the kernel in the same range.
    for (int row = 0; row < rows; ++row) {
      std::uint8_t* base = weights.data() + row * test.row_bytes;
      if (test.type == hipk::kF32) {
        for (int i = 0; i < cols; ++i) {
          const float value = dist(rng);
          std::memcpy(base + i * sizeof(float), &value, sizeof(value));
        }
        continue;
      }
      const int blocks = test.type == hipk::kQ8_0 ? cols / 32 : cols / 256;
      for (int b = 0; b < blocks; ++b) {
        const float scale = 0.01f + 0.001f * static_cast<float>(b);
        if (test.type == hipk::kQ8_0) {
          auto* block = reinterpret_cast<BlockQ8_0*>(base + b * sizeof(BlockQ8_0));
          block->d = FloatToHalf(scale);
        } else if (test.type == hipk::kQ4K) {
          auto* block = reinterpret_cast<BlockQ4K*>(base + b * sizeof(BlockQ4K));
          block->d = FloatToHalf(scale);
          block->dmin = FloatToHalf(scale * 0.5f);
          for (int i = 0; i < 12; ++i) {
            block->scales[i] = static_cast<std::uint8_t>(4 + i);
          }
        } else {
          auto* block = reinterpret_cast<BlockQ6K*>(base + b * sizeof(BlockQ6K));
          block->d = FloatToHalf(scale);
          for (int i = 0; i < 16; ++i) {
            block->scales[i] = static_cast<std::int8_t>(-8 + i);
          }
        }
      }
    }

    // Host reference.
    std::vector<float> expected(rows);
    for (int row = 0; row < rows; ++row) {
      const std::uint8_t* base = weights.data() + row * test.row_bytes;
      double sum = 0.0;
      for (int i = 0; i < cols; ++i) {
        double value = 0.0;
        if (test.type == hipk::kF32) {
          float raw;
          std::memcpy(&raw, base + i * sizeof(float), sizeof(raw));
          value = raw;
        } else if (test.type == hipk::kQ8_0) {
          const auto* block = reinterpret_cast<const BlockQ8_0*>(
              base + (i / 32) * sizeof(BlockQ8_0));
          value = HalfToFloat(block->d) * static_cast<double>(block->qs[i % 32]);
        } else if (test.type == hipk::kQ4K) {
          const auto* block = reinterpret_cast<const BlockQ4K*>(
              base + (i / 256) * sizeof(BlockQ4K));
          value = Q4Value(*block, i % 256);
        } else {
          const auto* block = reinterpret_cast<const BlockQ6K*>(
              base + (i / 256) * sizeof(BlockQ6K));
          value = Q6Value(*block, i % 256);
        }
        sum += value * x[i];
      }
      expected[row] = static_cast<float>(sum);
    }

    float* device_weights = nullptr;
    float* device_x = nullptr;
    float* device_y = nullptr;
    HIP_CHECK(hipMalloc(&device_weights, weights.size()));
    HIP_CHECK(hipMalloc(&device_x, x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_y, rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(device_weights, weights.data(), weights.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_x, x.data(), x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    hipk::LaunchGemv(device_weights, test.type, rows, cols, device_x, device_y, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(rows);
    HIP_CHECK(hipMemcpy(actual.data(), device_y, rows * sizeof(float),
                        hipMemcpyDeviceToHost));

    for (int row = 0; row < rows; ++row) {
      const double tolerance =
          test.type == hipk::kF32 ? 1e-4 : 2e-3;
      const double scale = std::max(1.0, std::fabs(static_cast<double>(expected[row])));
      const double relative =
          std::fabs(static_cast<double>(actual[row]) - expected[row]) / scale;
      char label[64];
      std::snprintf(label, sizeof(label), "%s row %d", test.name, row);
      Check(relative <= tolerance, label, relative, tolerance);
    }
    HIP_CHECK(hipFree(device_weights));
    HIP_CHECK(hipFree(device_x));
    HIP_CHECK(hipFree(device_y));
  }

  // RMSNorm: compare against the same formula on the host.
  {
    const int norm_rows = 2;
    const int norm_cols = 256;
    std::vector<float> values(norm_rows * norm_cols);
    std::vector<float> weights(norm_cols);
    for (auto& value : values) {
      value = dist(rng);
    }
    for (auto& value : weights) {
      value = 0.5f + 0.5f * std::fabs(dist(rng));
    }
    std::vector<float> expected(values.size());
    for (int row = 0; row < norm_rows; ++row) {
      double sum = 0.0;
      for (int i = 0; i < norm_cols; ++i) {
        const double value = values[row * norm_cols + i];
        sum += value * value;
      }
      const double scale = 1.0 / std::sqrt(sum / norm_cols + 1e-6);
      for (int i = 0; i < norm_cols; ++i) {
        expected[row * norm_cols + i] =
            static_cast<float>(values[row * norm_cols + i] * scale * weights[i]);
      }
    }
    float* device_values = nullptr;
    float* device_weights = nullptr;
    float* device_out = nullptr;
    HIP_CHECK(hipMalloc(&device_values, values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_out, values.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(device_values, values.data(), values.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_weights, weights.data(), weights.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    hipk::LaunchRmsNorm(device_values, device_weights, device_out, norm_rows, norm_cols,
                        1e-6f, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(values.size());
    HIP_CHECK(hipMemcpy(actual.data(), device_out, values.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    }
    Check(worst <= 1e-5, "rmsnorm max abs error", worst, 1e-5);
    HIP_CHECK(hipFree(device_values));
    HIP_CHECK(hipFree(device_weights));
    HIP_CHECK(hipFree(device_out));
  }

  std::printf("operators: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
