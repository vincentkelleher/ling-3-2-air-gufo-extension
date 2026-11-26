// Does this extension's dequantization agree with Gufo's reference on real
// weight rows of every storage type the model uses?
//
// The kernels dequantize with the same math (device port); if these disagree
// with core::Dot, every GEMV in the engine is wrong.
//
// Usage: dot_vs_gufo_test [model.gguf]   (else $LING3_GGUF; without one: skip)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/experimental/ggml_gemm.hpp>

#include <gufo/ext/gguf_reader.hpp>

namespace {

namespace core = gufo::core;
namespace quant = gufo::quant;

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
struct BlockQ5K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
#pragma pack(pop)

float Half(std::uint16_t bits) {
  const std::uint32_t sign = (bits & 0x8000u) << 16;
  std::uint32_t exponent = (bits >> 10) & 0x1Fu;
  std::uint32_t mantissa = bits & 0x3FFu;
  std::uint32_t out = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      out = sign;
    } else {
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      ++exponent;
      mantissa &= 0x3FFu;
      out = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    out = sign | 0x7F800000u | (mantissa << 13);
  } else {
    out = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

void ScaleMin(int index, const std::uint8_t* packed, float& scale, float& minimum) {
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
  ScaleMin(group, block.scales, scale, minimum);
  return Half(block.d) * scale * quant - Half(block.dmin) * minimum;
}

float Q5Value(const BlockQ5K& block, int index) {
  const int group = index / 64;
  const int within = index % 64;
  const int lane = within % 32;
  const bool low_half = within < 32;
  const std::uint8_t packed = block.qs[group * 32 + lane];
  const std::uint8_t quant4 = low_half ? (packed & 0x0F) : (packed >> 4);
  const std::uint8_t high = block.qh[lane];
  const int bit = 2 * group + (low_half ? 0 : 1);
  const float quant = static_cast<float>(quant4) + (((high >> bit) & 1u) ? 16.0f : 0.0f);
  const int scale_index = 2 * group + (low_half ? 0 : 1);
  float scale = 0;
  float minimum = 0;
  ScaleMin(scale_index, block.scales, scale, minimum);
  return Half(block.d) * scale * quant - Half(block.dmin) * minimum;
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
    case 0: low = block.ql[ql_base + lane] & 0x0F; high = qh & 0x03; break;
    case 1: low = block.ql[ql_base + 32 + lane] & 0x0F; high = (qh >> 2) & 0x03; break;
    case 2: low = block.ql[ql_base + lane] >> 4; high = (qh >> 4) & 0x03; break;
    default: low = block.ql[ql_base + 32 + lane] >> 4; high = (qh >> 6) & 0x03; break;
  }
  const int scale_index = half * 8 + lane / 16 + segment * 2;
  const float quant = static_cast<float>(static_cast<int>((high << 4) | low) - 32);
  return Half(block.d) * static_cast<float>(block.scales[scale_index]) * quant;
}

/// Same layout as the kernels: row of `cols` weights, dot with `x`.
double HostDot(core::GgmlType type, const void* row, const std::vector<float>& x,
               std::size_t cols) {
  double sum = 0.0;
  switch (type) {
    case core::GgmlType::kF32: {
      const auto* values = static_cast<const float*>(row);
      for (std::size_t i = 0; i < cols; ++i) sum += values[i] * x[i];
      return sum;
    }
    case core::GgmlType::kQ8_0: {
      const auto* blocks = static_cast<const BlockQ8_0*>(row);
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const float d = Half(blocks[b].d);
        for (int j = 0; j < 32; ++j) sum += d * blocks[b].qs[j] * x[b * 32 + j];
      }
      return sum;
    }
    case core::GgmlType::kQ4_K: {
      const auto* blocks = static_cast<const BlockQ4K*>(row);
      for (std::size_t b = 0; b < cols / 256; ++b) {
        for (int j = 0; j < 256; ++j) sum += Q4Value(blocks[b], j) * x[b * 256 + j];
      }
      return sum;
    }
    case core::GgmlType::kQ5_K: {
      const auto* blocks = static_cast<const BlockQ5K*>(row);
      for (std::size_t b = 0; b < cols / 256; ++b) {
        for (int j = 0; j < 256; ++j) sum += Q5Value(blocks[b], j) * x[b * 256 + j];
      }
      return sum;
    }
    case core::GgmlType::kQ6_K: {
      const auto* blocks = static_cast<const BlockQ6K*>(row);
      for (std::size_t b = 0; b < cols / 256; ++b) {
        for (int j = 0; j < 256; ++j) sum += Q6Value(blocks[b], j) * x[b * 256 + j];
      }
      return sum;
    }
    case core::GgmlType::kQ4_0: {
      const auto* blocks = static_cast<const BlockQ4_0*>(row);
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const float d = Half(blocks[b].d);
        for (int j = 0; j < 32; ++j) {
          const int nibble = (j & 1) ? (blocks[b].qs[j / 2] >> 4)
                                     : (blocks[b].qs[j / 2] & 0x0F);
          sum += d * static_cast<double>(nibble - 8) * x[b * 32 + j];
        }
      }
      return sum;
    }
  }
}

std::size_t RowBytes(core::GgmlType type, std::size_t cols) {
  switch (type) {
    case core::GgmlType::kF32: return cols * 4;
    case core::GgmlType::kQ8_0: return cols / 32 * 34;
    case core::GgmlType::kQ4_0: return cols / 32 * 18;
    case core::GgmlType::kQ4_K: return cols / 256 * 144;
    case core::GgmlType::kQ5_K: return cols / 256 * 176;
    case core::GgmlType::kQ6_K: return cols / 256 * 210;
    default: return 0;
  }
}

/// Integer per-group dot in llama.cpp's shape: q8 activations, integer products
/// per 32-group, one float scale per group. Compared against quant::Dot to prove
/// the unpacking before using it in a kernel.
double IntegerQ4Dot(const BlockQ4K& block, const std::vector<float>& x) {
  float amax = 0.0f;
  for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(x[i]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const double d = Half(block.d);
  const double dmin = Half(block.dmin);
  double inner = 0.0;
  for (int g = 0; g < 8; ++g) {
    int sum_qa = 0;
    int sum_a = 0;
    for (int i = 0; i < 32; ++i) {
      const int index = g * 32 + i;
      const int code = index % 32;
      const std::uint8_t packed = block.qs[(g / 2) * 32 + code];
      const int q = (g & 1) ? (packed >> 4) : (packed & 0x0F);
      const float raw = std::rint(x[index] / scale);
      const int a = static_cast<int>(raw > 127.0f ? 127.0f : (raw < -127.0f ? -127.0f : raw));
      sum_qa += q * a;
      sum_a += a;
    }
    float sc = 0.0f;
    float m = 0.0f;
    ScaleMin(g, block.scales, sc, m);
    inner += (d * static_cast<double>(sc) * static_cast<double>(sum_qa) -
              dmin * static_cast<double>(m) * static_cast<double>(sum_a)) *
             static_cast<double>(scale);
  }
  return inner;
}

/// Segment-major Q6_K unpacking, the shape the kernel would use: hoist the
/// nibble/2-bit selection per 16-element group instead of choosing per element.
int Q6CodeSegmentMajor(const BlockQ6K& block, int index) {
  const int group = index / 16;
  const int i = index % 16;
  const int half = group / 8;
  const int segment = (group % 8) / 2;
  const int lane = (group * 16) % 32 + i;
  const std::uint8_t* ql = block.ql + half * 64 + (segment % 2) * 32;
  const int ql_shift = (segment / 2) * 4;
  const int qh_shift = segment * 2;
  const int low = (ql[lane] >> ql_shift) & 0x0F;
  const int high = (block.qh[half * 32 + lane] >> qh_shift) & 0x03;
  return ((high << 4) | low) - 32;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  if (argc > 1) {
    path = argv[1];
  } else if (const char* env = std::getenv("LING3_GGUF")) {
    path = env;
  }
  if (path.empty()) {
    std::printf("SKIP: set LING3_GGUF or pass the model .gguf path\n");
    return 77;
  }
  std::string error;
  const auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), error.c_str());
    return 2;
  }

  // One tensor per storage type, plus block 31's Q5_K experts.
  const char* names[] = {"blk.42.attn_q.weight", "blk.42.nextn.eh_proj.weight",
                         "blk.0.attn_q.weight", "blk.0.attn_norm.weight",
                         "blk.2.ffn_gate_exps.weight", "blk.2.ffn_up_exps.weight",
                         "blk.31.ffn_down_exps.weight", "token_embd.weight",
                         "output.weight"};
  int failures = 0;
  int checked = 0;
  for (const char* name : names) {
    const auto* info = reader->FindTensor(name);
    if (info == nullptr) {
      std::printf("missing %s\n", name);
      ++failures;
      continue;
    }
    const std::size_t cols = static_cast<std::size_t>(info->dimensions[0]);
    if (!quant::IsSupported(info->type)) {
      std::printf("skipped %s: Gufo's reference does not decode %s\n", name,
                  core::ToString(info->type).data());
      continue;
    }
    const std::size_t stride = RowBytes(info->type, cols);
    if (stride == 0) {
      std::printf("unsupported type for %s\n", name);
      ++failures;
      continue;
    }
    std::vector<float> x(cols);
    for (std::size_t i = 0; i < cols; ++i) {
      x[i] = std::sin(0.001 * static_cast<double>(i) + 1.0);
    }
    for (int row = 0; row < 2; ++row) {
      const auto* base = static_cast<const std::uint8_t*>(info->data) + row * stride;
      const double mine = HostDot(info->type, base, x, cols);
      std::vector<float> reference(cols, 0.0f);
      quant::Dequantize(info->type, base, reference.data(), cols);
      double gufo = 0.0;
      for (std::size_t i = 0; i < cols; ++i) gufo += reference[i] * x[i];
      const double scale = std::max(1.0, std::fabs(gufo));
      const double relative = std::fabs(mine - gufo) / scale;
      ++checked;
      if (relative > 1e-4) {
        ++failures;
        std::printf("MISMATCH %s row %d: mine %.6f gufo %.6f (rel %.2e)\n", name, row,
                    mine, gufo, relative);
      }
    }
  }
  {
    const auto* info = reader->FindTensor("blk.2.ffn_gate_exps.weight");
    if (info != nullptr) {
      const std::size_t cols = static_cast<std::size_t>(info->dimensions[0]);
      std::vector<float> x(cols);
      for (std::size_t i = 0; i < cols; ++i) x[i] = std::sin(0.001 * static_cast<double>(i) + 1.0);
      const std::size_t stride = RowBytes(info->type, cols) / 10;  // one 256-block
      const auto* base = static_cast<const std::uint8_t*>(info->data);
      for (int block = 0; block < 3; ++block) {
        std::vector<float> block_x(x.begin() + block * 256, x.begin() + (block + 1) * 256);
        const auto* block_data = base + block * stride;
        const double integer = IntegerQ4Dot(*reinterpret_cast<const BlockQ4K*>(block_data), block_x);
        std::vector<float> reference(256, 0.0f);
        quant::Dequantize(core::GgmlType::kQ4_K, block_data, reference.data(), 256);
        double dequant = 0.0;
        for (int i = 0; i < 256; ++i) dequant += reference[i] * block_x[i];
        const double relative = std::fabs(integer - dequant) / std::max(1.0, std::fabs(dequant));
        ++checked;
        if (relative > 0.02) {
          ++failures;
          std::printf("MISMATCH integer Q4_K block %d: integer %.6f dequant %.6f (rel %.3f)\n",
                      block, integer, dequant, relative);
        } else {
          std::printf("integer Q4_K block %d: rel %.4f\n", block, relative);
        }
      }
    }
  }

  {
    const auto* info = reader->FindTensor("blk.2.ffn_up_exps.weight");
    if (info != nullptr) {
      const auto* base = static_cast<const std::uint8_t*>(info->data);
      int mismatches = 0;
      for (int block = 0; block < 3; ++block) {
        const auto* codes = reinterpret_cast<const BlockQ6K*>(base + block * 210);
        for (int i = 0; i < 256; ++i) {
          if (Q6CodeSegmentMajor(*codes, i) != static_cast<int>(Q6Value(*codes, i)) + 32 - 32) {
            const int reference = static_cast<int>(Q6Value(*codes, i));
            // Q6Value returns the dequantised value; compare codes through it.
            (void)reference;
          }
        }
        for (int i = 0; i < 256; ++i) {
          const int mine = Q6CodeSegmentMajor(*codes, i);
          const float via_value = Q6Value(*codes, i);
          const float d = Half(codes->d);
          const int group = i / 16;
          const float expected = d * static_cast<float>(codes->scales[group]) * mine;
          if (std::fabs(expected - via_value) > 1e-6f * std::max(1.0f, std::fabs(via_value))) {
            ++mismatches;
          }
        }
      }
      ++checked;
      if (mismatches != 0) {
        ++failures;
        std::printf("MISMATCH segment-major Q6_K: %d elements differ\n", mismatches);
      } else {
        std::printf("segment-major Q6_K unpacking matches the reference\n");
      }
    }
  }

  std::printf("dot vs gufo: %d rows checked, %d failures\n", checked, failures);
  return failures == 0 ? 0 : 1;
}
