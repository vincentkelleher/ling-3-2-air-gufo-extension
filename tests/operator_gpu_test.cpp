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
#include <algorithm>
#include <random>
#include <vector>

#include "src/hip/kernels.hpp"
#include "src/hip/attention.hpp"
#include "src/hip/layers.hpp"
#include "src/hip/moe.hpp"

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
      // Quantized types quantize the *activations* to int8 per block (as
      // llama.cpp does), so they differ from an exact float reference by up to
      // ~0.5% of the magnitude; that is expected, not an error.
      const double tolerance = test.type == hipk::kF32 ? 1e-4 : 5e-2;
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

  // Causal conv1d step: output, then the shifted state.
  {
    const int channels = 16;
    std::vector<float> x(channels), weight(4 * channels), state(3 * channels);
    for (auto& v : x) v = dist(rng);
    for (auto& v : weight) v = dist(rng);
    for (auto& v : state) v = dist(rng);
    std::vector<float> expected_out(channels);
    std::vector<float> expected_state(state);
    for (int c = 0; c < channels; ++c) {
      const float window[4] = {state[c], state[channels + c], state[2 * channels + c], x[c]};
      float sum = 0.0f;
      for (int k = 0; k < 4; ++k) sum += weight[c * 4 + k] * window[k];
      expected_out[c] = sum;
      expected_state[c] = window[1];
      expected_state[channels + c] = window[2];
      expected_state[2 * channels + c] = window[3];
    }
    float *dx, *dw, *ds, *dy;
    HIP_CHECK(hipMalloc(&dx, channels * sizeof(float)));
    HIP_CHECK(hipMalloc(&dw, weight.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&ds, state.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, channels * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, x.data(), channels * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds, state.data(), state.size() * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchConv1dStep(dx, dw, ds, channels, 0, dy, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> out(channels), got_state(state.size());
    HIP_CHECK(hipMemcpy(out.data(), dy, channels * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_state.data(), ds, state.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0, worst_state = 0.0;
    for (int c = 0; c < channels; ++c) {
      worst = std::max(worst, std::fabs(static_cast<double>(out[c]) - expected_out[c]));
    }
    for (std::size_t i = 0; i < state.size(); ++i) {
      worst_state = std::max(worst_state, std::fabs(static_cast<double>(got_state[i]) - expected_state[i]));
    }
    Check(worst <= 1e-5, "conv1d step max abs error", worst, 1e-5);
    Check(worst_state <= 1e-5, "conv1d state max abs error", worst_state, 1e-5);
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(ds)); HIP_CHECK(hipFree(dy));
  }

  // L2 normalisation per head.
  {
    const int heads = 4, dim = 32;
    std::vector<float> values(heads * dim);
    for (auto& v : values) v = dist(rng);
    std::vector<float> expected(values.size());
    for (int h = 0; h < heads; ++h) {
      double sum = 0.0;
      for (int i = 0; i < dim; ++i) sum += values[h * dim + i] * values[h * dim + i];
      const double norm = std::sqrt(sum);
      const double scale = 1.0 / (norm < 1e-6 ? 1e-6 : norm);
      for (int i = 0; i < dim; ++i) expected[h * dim + i] = static_cast<float>(values[h * dim + i] * scale);
    }
    float *dv, *dy;
    HIP_CHECK(hipMalloc(&dv, values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, values.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dv, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchL2Norm(dv, dy, heads, dim, 1e-6f, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(values.size());
    HIP_CHECK(hipMemcpy(actual.data(), dy, values.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    Check(worst <= 1e-5, "l2norm max abs error", worst, 1e-5);
    HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(dy));
  }

  // RoPE (neox pairing) on the first 64 dims of a 192-wide head.
  {
    const int heads = 1, dim = 192, n_rot = 64;
    const int position = 7;
    const float base = 6000000.0f;
    std::vector<float> values(dim);
    for (auto& v : values) v = dist(rng);
    std::vector<float> expected(values);
    for (int pair = 0; pair < n_rot / 2; ++pair) {
      const float inv = std::pow(base, -2.0f * static_cast<float>(pair) / static_cast<float>(n_rot));
      const float angle = static_cast<float>(position) * inv;
      const float c = std::cos(angle), s = std::sin(angle);
      const float a = expected[pair], b = expected[pair + n_rot / 2];
      expected[pair] = a * c - b * s;
      expected[pair + n_rot / 2] = a * s + b * c;
    }
    float* dv;
    HIP_CHECK(hipMalloc(&dv, dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(dv, values.data(), dim * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchRope(dv, heads, dim, n_rot, position, base, true, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(dim);
    HIP_CHECK(hipMemcpy(actual.data(), dv, dim * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int i = 0; i < dim; ++i) worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    Check(worst <= 2e-5, "rope max abs error", worst, 2e-5);
    HIP_CHECK(hipFree(dv));
  }

  // Argmax.
  {
    const int count = 1000;
    std::vector<float> values(count);
    for (auto& v : values) v = dist(rng);
    const int expected = static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
    float* dv;
    std::int32_t* dr;
    HIP_CHECK(hipMalloc(&dv, count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dr, sizeof(std::int32_t)));
    HIP_CHECK(hipMemcpy(dv, values.data(), count * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchArgmax(dv, count, dr, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::int32_t actual = -1;
    HIP_CHECK(hipMemcpy(&actual, dr, sizeof(std::int32_t), hipMemcpyDeviceToHost));
    Check(actual == expected, "argmax index", actual, expected);
    HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(dr));
  }

  // KDA delta-rule step against the ggml algorithm on the host.
  {
    const int heads = 2, dim = 8;
    std::vector<float> q(heads * dim), k(heads * dim), v(heads * dim),
        g(heads * dim), beta(heads), state(heads * dim * dim);
    for (auto& value : q) value = dist(rng);
    for (auto& value : k) value = dist(rng);
    for (auto& value : v) value = dist(rng);
    for (auto& value : g) value = -0.5f + 0.5f * dist(rng);
    for (auto& value : beta) value = 0.5f + 0.5f * std::fabs(dist(rng));
    for (auto& value : state) value = 0.1f * dist(rng);

    std::vector<float> expected_state(state), expected_out(heads * dim);
    for (int h = 0; h < heads; ++h) {
      float* m = expected_state.data() + h * dim * dim;
      const float* qh = q.data() + h * dim;
      const float* kh = k.data() + h * dim;
      const float* vh = v.data() + h * dim;
      const float* gh = g.data() + h * dim;
      for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < dim; ++i) m[j * dim + i] *= std::exp(gh[i]);
      }
      std::vector<float> delta(dim);
      for (int j = 0; j < dim; ++j) {
        double dot = 0.0;
        for (int i = 0; i < dim; ++i) dot += m[j * dim + i] * kh[i];
        delta[j] = static_cast<float>((vh[j] - dot) * beta[h]);
      }
      for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < dim; ++i) m[j * dim + i] += kh[i] * delta[j];
      }
      for (int j = 0; j < dim; ++j) {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) sum += m[j * dim + i] * qh[i];
        expected_out[h * dim + j] = static_cast<float>(sum / std::sqrt(static_cast<double>(dim)));
      }
    }

    float *dq, *dk, *dv, *dg, *db, *ds, *dy;
    HIP_CHECK(hipMalloc(&dq, q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dk, k.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dv, v.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dg, g.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, beta.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&ds, state.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, expected_out.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dq, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dk, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dg, g.data(), g.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, beta.data(), beta.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds, state.data(), state.size() * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchDeltaRuleStep(dq, dk, dv, dg, db, ds, dy, heads, dim, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got_out(heads * dim), got_state(state.size());
    HIP_CHECK(hipMemcpy(got_out.data(), dy, got_out.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_state.data(), ds, state.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst_out = 0.0, worst_state = 0.0;
    for (std::size_t i = 0; i < got_out.size(); ++i) {
      worst_out = std::max(worst_out, std::fabs(static_cast<double>(got_out[i]) - expected_out[i]));
    }
    for (std::size_t i = 0; i < got_state.size(); ++i) {
      worst_state = std::max(worst_state, std::fabs(static_cast<double>(got_state[i]) - expected_state[i]));
    }
    Check(worst_out <= 1e-5, "delta rule out max abs error", worst_out, 1e-5);
    Check(worst_state <= 1e-5, "delta rule state max abs error", worst_state, 1e-5);
    HIP_CHECK(hipFree(dq)); HIP_CHECK(hipFree(dk)); HIP_CHECK(hipFree(dv));
    HIP_CHECK(hipFree(dg)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(ds));
    HIP_CHECK(hipFree(dy));
  }

  // MLA: absorb, scores, softmax, latent sum, value absorb and gate.
  {
    const int heads = 2;
    const int len = 3;
    const std::size_t latent = hipk::kMlaLatent, rope = hipk::kMlaRope,
                      nope = hipk::kMlaNope, vdim = hipk::kMlaV;
    std::vector<float> q_nope(heads * nope), q_pe(heads * rope),
        wk_b(heads * latent * nope), wv_b(heads * vdim * latent),
        gate(heads), cache(len * hipk::kMlaCacheStride);
    for (auto& x : q_nope) x = dist(rng);
    for (auto& x : q_pe) x = dist(rng);
    for (auto& x : wk_b) x = 0.1f * dist(rng);
    for (auto& x : wv_b) x = 0.1f * dist(rng);
    for (auto& x : gate) x = dist(rng);
    for (auto& x : cache) x = dist(rng);

    std::vector<float> q_abs(heads * latent), scores(heads * len),
        latent_out(heads * latent), expected(heads * vdim);
    for (int h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < latent; ++i) {
        double sum = 0.0;
        for (std::size_t d = 0; d < nope; ++d) {
          sum += q_nope[h * nope + d] * wk_b[(h * latent + i) * nope + d];
        }
        q_abs[h * latent + i] = static_cast<float>(sum);
      }
      for (int p = 0; p < len; ++p) {
        double sum = 0.0;
        for (std::size_t i = 0; i < latent; ++i) {
          sum += q_abs[h * latent + i] * cache[p * hipk::kMlaCacheStride + i];
        }
        for (std::size_t j = 0; j < rope; ++j) {
          sum += q_pe[h * rope + j] * cache[p * hipk::kMlaCacheStride + latent + j];
        }
        scores[h * len + p] = static_cast<float>(sum * hipk::kMlaScoreScale);
      }
      double max_score = scores[h * len];
      for (int p = 1; p < len; ++p) max_score = std::max(max_score, static_cast<double>(scores[h * len + p]));
      double total = 0.0;
      for (int p = 0; p < len; ++p) {
        scores[h * len + p] = static_cast<float>(std::exp(scores[h * len + p] - max_score));
        total += scores[h * len + p];
      }
      for (int p = 0; p < len; ++p) scores[h * len + p] = static_cast<float>(scores[h * len + p] / total);
      for (std::size_t i = 0; i < latent; ++i) {
        double sum = 0.0;
        for (int p = 0; p < len; ++p) sum += scores[h * len + p] * cache[p * hipk::kMlaCacheStride + i];
        latent_out[h * latent + i] = static_cast<float>(sum);
      }
      const double g = 1.0 / (1.0 + std::exp(-static_cast<double>(gate[h])));
      for (std::size_t o = 0; o < vdim; ++o) {
        double sum = 0.0;
        for (std::size_t i = 0; i < latent; ++i) {
          sum += latent_out[h * latent + i] * wv_b[(h * vdim + o) * latent + i];
        }
        expected[h * vdim + o] = static_cast<float>(sum * g);
      }
    }

    float *dq, *dqr, *dwk, *dwv, *dg, *dcache, *dqa, *dsc, *dlat, *dout;
    HIP_CHECK(hipMalloc(&dq, q_nope.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dqr, q_pe.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dwk, wk_b.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dwv, wv_b.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dg, gate.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dcache, cache.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dqa, q_abs.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dsc, scores.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dlat, latent_out.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, expected.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dq, q_nope.data(), q_nope.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dqr, q_pe.data(), q_pe.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dwk, wk_b.data(), wk_b.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dwv, wv_b.data(), wv_b.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dg, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcache, cache.data(), cache.size() * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchMlaAbsorbQ(dq, dwk, dqa, heads, 0);
    hipk::LaunchMlaScores(dqa, dqr, dcache, heads, len, dsc, 0);
    hipk::LaunchMlaSoftmax(dsc, heads, len, 0);
    hipk::LaunchMlaLatentSum(dsc, dcache, heads, len, dlat, 0);
    hipk::LaunchMlaValueAbsorb(dlat, dwv, dg, dout, heads, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(expected.size());
    HIP_CHECK(hipMemcpy(actual.data(), dout, expected.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    }
    Check(worst <= 1e-4, "mla step max abs error", worst, 1e-4);
    HIP_CHECK(hipFree(dq)); HIP_CHECK(hipFree(dqr)); HIP_CHECK(hipFree(dwk));
    HIP_CHECK(hipFree(dwv)); HIP_CHECK(hipFree(dg)); HIP_CHECK(hipFree(dcache));
    HIP_CHECK(hipFree(dqa)); HIP_CHECK(hipFree(dsc)); HIP_CHECK(hipFree(dlat));
    HIP_CHECK(hipFree(dout));
  }

  // Router: group-limited top-k on the model's real shape.
  {
    const int n_expert = 512, n_group = 8, group_used = 4, top_k = 8, per_group = 64;
    const float scale = 2.5f;
    std::vector<float> logits(n_expert), bias(n_expert);
    for (auto& v : logits) v = 2.0f * dist(rng);
    for (auto& v : bias) v = 0.5f * dist(rng);
    std::vector<float> scores(n_expert);
    for (int e = 0; e < n_expert; ++e) scores[e] = 1.0f / (1.0f + std::exp(-(logits[e] + bias[e])));
    std::vector<float> group_score(n_group);
    for (int g = 0; g < n_group; ++g) {
      float first = -1.0f, second = -1.0f;
      for (int k = 0; k < per_group; ++k) {
        const float value = scores[g * per_group + k];
        if (value > first) { second = first; first = value; }
        else if (value > second) { second = value; }
      }
      group_score[g] = first + second;
    }
    std::vector<int> kept, expected_ids;
    for (int picked = 0; picked < group_used; ++picked) {
      int best = -1; float best_value = -1.0f;
      for (int g = 0; g < n_group; ++g) {
        if (std::find(kept.begin(), kept.end(), g) != kept.end()) continue;
        if (group_score[g] > best_value) { best_value = group_score[g]; best = g; }
      }
      kept.push_back(best);
    }
    for (int picked = 0; picked < top_k; ++picked) {
      int best = -1; float best_value = -1.0f;
      for (int e = 0; e < n_expert; ++e) {
        if (std::find(expected_ids.begin(), expected_ids.end(), e) != expected_ids.end()) continue;
        if (std::find(kept.begin(), kept.end(), e / per_group) == kept.end()) continue;
        if (scores[e] > best_value) { best_value = scores[e]; best = e; }
      }
      expected_ids.push_back(best);
    }
    float sum = 0.0f;
    for (int id : expected_ids) sum += scores[id];
    std::vector<float> expected_weights(top_k);
    for (int i = 0; i < top_k; ++i) expected_weights[i] = scores[expected_ids[i]] / sum * scale;

    float *dl, *db;
    std::int32_t* di;
    float* dw;
    HIP_CHECK(hipMalloc(&dl, n_expert * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, n_expert * sizeof(float)));
    HIP_CHECK(hipMalloc(&di, top_k * sizeof(std::int32_t)));
    HIP_CHECK(hipMalloc(&dw, top_k * sizeof(float)));
    HIP_CHECK(hipMemcpy(dl, logits.data(), n_expert * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, bias.data(), n_expert * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchRouterSelect(dl, db, n_expert, n_group, group_used, top_k, scale, di, dw, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::int32_t> got_ids(top_k);
    std::vector<float> got_weights(top_k);
    HIP_CHECK(hipMemcpy(got_ids.data(), di, top_k * sizeof(std::int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_weights.data(), dw, top_k * sizeof(float), hipMemcpyDeviceToHost));
    bool same = true;
    for (int i = 0; i < top_k; ++i) same = same && got_ids[i] == expected_ids[i];
    double worst = 0.0;
    for (int i = 0; i < top_k; ++i) worst = std::max(worst, std::fabs(static_cast<double>(got_weights[i]) - expected_weights[i]));
    Check(same, "router expert ids", same ? 0.0 : 1.0, 0.0);
    Check(worst <= 1e-5, "router weight max abs error", worst, 1e-5);
    HIP_CHECK(hipFree(dl)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(di)); HIP_CHECK(hipFree(dw));
  }

  // Expert gather: two F32 experts, n_embd = 8, n_ff = 4.
  {
    const int n_embd = 8, n_ff = 4, experts = 2;
    std::vector<float> gate(experts * n_ff * n_embd), up(experts * n_ff * n_embd),
        down(experts * n_embd * n_ff), x(n_embd);
    for (auto& v : gate) v = 0.5f * dist(rng);
    for (auto& v : up) v = 0.5f * dist(rng);
    for (auto& v : down) v = 0.5f * dist(rng);
    for (auto& v : x) v = dist(rng);
    std::vector<std::int32_t> ids{1, 0};
    std::vector<float> weights{0.6f, 0.4f};
    std::vector<float> expected(n_embd, 0.0f);
    for (int i = 0; i < 2; ++i) {
      const int e = ids[i];
      for (int o = 0; o < n_embd; ++o) {
        float acc = 0.0f;
        for (int j = 0; j < n_ff; ++j) {
          double g = 0.0, u = 0.0;
          for (int d = 0; d < n_embd; ++d) {
            g += gate[(e * n_ff + j) * n_embd + d] * x[d];
            u += up[(e * n_ff + j) * n_embd + d] * x[d];
          }
          acc += static_cast<float>((g / (1.0 + std::exp(-g))) * u) * down[(e * n_embd + o) * n_ff + j];
        }
        expected[o] += weights[i] * acc;
      }
    }
    float *dg, *du, *dd, *dx, *dout, *scratch;
    std::int32_t* di;
    float* dw;
    HIP_CHECK(hipMalloc(&dg, gate.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&du, up.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dd, down.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dx, n_embd * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, n_embd * sizeof(float)));
    HIP_CHECK(hipMalloc(&scratch, (2 * n_ff + n_embd) * sizeof(float)));
    HIP_CHECK(hipMalloc(&di, 2 * sizeof(std::int32_t)));
    HIP_CHECK(hipMalloc(&dw, 2 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dg, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(du, up.data(), up.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, down.data(), down.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), n_embd * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(di, ids.data(), 2 * sizeof(std::int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw, weights.data(), 2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dout, 0, n_embd * sizeof(float)));
    hipk::LaunchMoeGather(dg, hipk::kF32, du, hipk::kF32, dd, hipk::kF32, n_embd, n_ff,
                          dx, di, dw, 2, dout, scratch, scratch + n_ff,
                          scratch + 2 * n_ff, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(n_embd);
    HIP_CHECK(hipMemcpy(actual.data(), dout, n_embd * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int o = 0; o < n_embd; ++o) worst = std::max(worst, std::fabs(static_cast<double>(actual[o]) - expected[o]));
    Check(worst <= 1e-4, "moe gather max abs error", worst, 1e-4);
    HIP_CHECK(hipFree(dg)); HIP_CHECK(hipFree(du)); HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dout)); HIP_CHECK(hipFree(scratch));
    HIP_CHECK(hipFree(di)); HIP_CHECK(hipFree(dw));
  }

  // Fused expert GEMV: one launch covering every selected expert.
  {
    const int experts = 3, rows = 2, cols = 8;
    std::vector<float> weights(experts * rows * cols), x(cols);
    for (auto& v : weights) v = 0.5f * dist(rng);
    for (auto& v : x) v = dist(rng);
    std::vector<std::int32_t> ids{2, 0, 1};
    std::vector<float> expected(experts * rows, 0.0f);
    for (int e = 0; e < experts; ++e) {
      for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
          sum += weights[(ids[e] * rows + r) * cols + c] * x[c];
        }
        expected[e * rows + r] = static_cast<float>(sum);
      }
    }
    float* dw; float* dx; float* dy; std::int32_t* di;
    HIP_CHECK(hipMalloc(&dw, weights.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, expected.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&di, experts * sizeof(std::int32_t)));
    HIP_CHECK(hipMemcpy(dw, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(di, ids.data(), experts * sizeof(std::int32_t), hipMemcpyHostToDevice));
    hipk::LaunchGemvExperts(dw, hipk::kF32, di, experts, rows, cols, dx, 0, dy, rows, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(expected.size());
    HIP_CHECK(hipMemcpy(actual.data(), dy, expected.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    }
    Check(worst <= 1e-5, "expert gemv max abs error", worst, 1e-5);
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(di));
  }

  // Batched dot (prefill shape): M tokens against N rows, weight read once.
  for (const auto type : {hipk::kF32, hipk::kQ8_0}) {
    const int rows = 6, cols = 64, batch = 4;
    const std::size_t row_bytes = type == hipk::kF32
                                      ? static_cast<std::size_t>(cols) * sizeof(float)
                                      : static_cast<std::size_t>(cols) / 32 * 34;
    std::vector<std::uint8_t> weights(row_bytes * rows);
    for (auto& value : weights) value = static_cast<std::uint8_t>(byte(rng));
    if (type == hipk::kQ8_0) {
      for (int row = 0; row < rows; ++row) {
        auto* block = reinterpret_cast<BlockQ8_0*>(weights.data() + row * row_bytes);
        for (int b = 0; b < cols / 32; ++b) {
          block[b].d = FloatToHalf(0.01f + 0.001f * static_cast<float>(b));
        }
      }
    } else {
      for (auto& value : weights) value = 0;
      for (int row = 0; row < rows; ++row) {
        auto* values = reinterpret_cast<float*>(weights.data() + row * row_bytes);
        for (int c = 0; c < cols; ++c) values[c] = dist(rng);
      }
    }
    std::vector<float> x(batch * cols), expected(batch * rows, 0.0f);
    for (auto& value : x) value = dist(rng);
    for (int m = 0; m < batch; ++m) {
      for (int row = 0; row < rows; ++row) {
        const auto* base = weights.data() + row * row_bytes;
        double sum = 0.0;
        if (type == hipk::kF32) {
          const auto* values = reinterpret_cast<const float*>(base);
          for (int c = 0; c < cols; ++c) sum += values[c] * x[m * cols + c];
        } else {
          const auto* blocks = reinterpret_cast<const BlockQ8_0*>(base);
          for (int b = 0; b < cols / 32; ++b) {
            const float d = HalfToFloat(blocks[b].d);
            for (int j = 0; j < 32; ++j) {
              sum += d * static_cast<double>(blocks[b].qs[j]) * x[m * cols + b * 32 + j];
            }
          }
        }
        expected[m * rows + row] = static_cast<float>(sum);
      }
    }
    void* dw = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    HIP_CHECK(hipMalloc(&dw, weights.size()));
    HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, expected.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, weights.data(), weights.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchGemm(dw, type, rows, cols, dx, batch, cols, dy, rows, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual(expected.size());
    HIP_CHECK(hipMemcpy(actual.data(), dy, expected.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    }
    char label[64];
    std::snprintf(label, sizeof(label), "gemm %s max abs error",
                  type == hipk::kF32 ? "F32" : "Q8_0");
    Check(worst <= 1e-4, label, worst, 1e-4);
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
  }


  // Does batching actually pay? Time the same 4096x2560 Q8_0 matrix (11 MB, the
  // size of one attention projection) at batch 1 and batch 8. The prefill plan
  // assumes the weight read amortises; this measures it.
  {
    const int rows = 4096, cols = 2560, batch = 8;
    const std::size_t row_bytes = static_cast<std::size_t>(cols) / 32 * 34;
    void* dw = nullptr;
    std::vector<float> x(batch * cols, 0.5f), y(static_cast<std::size_t>(batch) * rows, 0.0f);
    HIP_CHECK(hipMalloc(&dw, row_bytes * rows));
    HIP_CHECK(hipMemset(dw, 1, row_bytes * rows));
    float* dx = nullptr;
    float* dy = nullptr;
    HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, y.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    for (const int m : {1, 8}) {
      hipEvent_t a, b;
      hipEventCreate(&a);
      hipEventCreate(&b);
      hipk::LaunchGemm(dw, hipk::kQ8_0, rows, cols, dx, m, cols, dy, rows, 0);
      HIP_CHECK(hipDeviceSynchronize());
      hipEventRecord(a);
      for (int r = 0; r < 5; ++r) {
        hipk::LaunchGemm(dw, hipk::kQ8_0, rows, cols, dx, m, cols, dy, rows, 0);
      }
      hipEventRecord(b);
      HIP_CHECK(hipEventSynchronize(b));
      float ms = 0.0f;
      hipEventElapsedTime(&ms, a, b);
      const double per_call = ms / 5.0;
      const double gbs = (row_bytes * rows) / (per_call / 1000.0) / 1e9;
      std::printf("gemm bench: batch %d -> %.3f ms/call, %.1f GB/s of weights (%s)\n",
                  m, per_call, gbs,
                  m == 1 ? "one token" : "amortised over 8 tokens");
      hipEventDestroy(a);
      hipEventDestroy(b);
    }
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
  }

  // Activation quantizer: the device buffers must equal a host quantization.
  {
    const int cols = 512;  // two 256-blocks
    std::vector<float> x(cols);
    for (auto& v : x) v = dist(rng);
    signed char* dv = nullptr;
    float* ds = nullptr;
    float* dx = nullptr;
    HIP_CHECK(hipMalloc(&dv, cols));
    HIP_CHECK(hipMalloc(&ds, (cols / 256) * sizeof(float)));
    HIP_CHECK(hipMalloc(&dx, cols * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, x.data(), cols * sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchQuantizeActivations(dx, cols, dv, ds, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<signed char> values(cols);
    std::vector<float> scales(cols / 256);
    HIP_CHECK(hipMemcpy(values.data(), dv, cols, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(scales.data(), ds, scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    int mismatches = 0;
    for (int b = 0; b < cols / 256; ++b) {
      float amax = 0.0f;
      for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[b * 256 + j]));
      const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
      if (std::fabs(scale - scales[b]) > 1e-6f * std::max(1.0f, scale)) ++mismatches;
      for (int j = 0; j < 256; ++j) {
        int expected = static_cast<int>(std::lrint(x[b * 256 + j] / scale));
        expected = expected > 127 ? 127 : (expected < -127 ? -127 : expected);
        if (values[b * 256 + j] != expected) ++mismatches;
      }
    }
    Check(mismatches == 0, "activation quantizer mismatches", mismatches, 0);
    HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(ds)); HIP_CHECK(hipFree(dx));
  }

  // Prepared-activation K-quant GEMV must equal the per-row path exactly.
  for (const auto type : {hipk::kQ4K, hipk::kQ6K}) {
    const int cols = 256, rows = 4;
    const std::size_t row_bytes = type == hipk::kQ4K ? 144 : 210;
    std::vector<std::uint8_t> weights(row_bytes * rows);
    for (auto& value : weights) value = static_cast<std::uint8_t>(byte(rng));
    // Sane fp16 scales: raw random bytes give NaN/inf scales and both paths then
    // differ in garbage, which is not a parity measurement.
    for (int row = 0; row < rows; ++row) {
      std::uint8_t* base = weights.data() + row * row_bytes;
      const float d = 0.01f + 0.001f * static_cast<float>(row);
      if (type == hipk::kQ4K) {
        auto* block = reinterpret_cast<BlockQ4K*>(base);
        block->d = FloatToHalf(d);
        block->dmin = FloatToHalf(d * 0.5f);
        for (int i = 0; i < 12; ++i) block->scales[i] = static_cast<std::uint8_t>(3 + i);
      } else {
        auto* block = reinterpret_cast<BlockQ6K*>(base);
        block->d = FloatToHalf(d);
        for (int i = 0; i < 16; ++i) block->scales[i] = static_cast<std::int8_t>(-6 + i);
      }
    }
    std::vector<float> x(cols);
    for (auto& value : x) value = dist(rng);
    // Host quantization, the same rounding the kernels use.
    std::vector<signed char> values(cols);
    std::vector<float> scales(1);
    float amax = 0.0f;
    for (const float value : x) amax = std::max(amax, std::fabs(value));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    scales[0] = scale;
    for (int i = 0; i < cols; ++i) {
      int a = static_cast<int>(std::lrint(x[i] / scale));
      a = a > 127 ? 127 : (a < -127 ? -127 : a);
      values[i] = static_cast<signed char>(a);
    }
    void* dw = nullptr;
    float* dx = nullptr;
    float* dy_row = nullptr;
    float* dy_prep = nullptr;
    signed char* dv = nullptr;
    float* ds = nullptr;
    HIP_CHECK(hipMalloc(&dw, weights.size()));
    HIP_CHECK(hipMalloc(&dx, cols * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy_row, rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy_prep, rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&dv, cols));
    HIP_CHECK(hipMalloc(&ds, sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, weights.data(), weights.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv, values.data(), cols, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds, scales.data(), sizeof(float), hipMemcpyHostToDevice));
    hipk::LaunchGemv(dw, type, rows, cols, dx, dy_row, 0);
    hipk::LaunchGemvPrepared(dw, type, rows, cols, dv, ds, dy_prep, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> row_out(rows), prep_out(rows);
    HIP_CHECK(hipMemcpy(row_out.data(), dy_row, rows * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(prep_out.data(), dy_prep, rows * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int i = 0; i < rows; ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(row_out[i]) - prep_out[i]));
    }
    char label[64];
    std::snprintf(label, sizeof(label), "prepared vs per-row %s",
                  type == hipk::kQ4K ? "Q4_K" : "Q6_K");
    Check(worst <= 1e-5, label, worst, 1e-5);
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy_row));
    HIP_CHECK(hipFree(dy_prep)); HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(ds));
  }

  std::printf("operators: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
