// Re-quantise one tensor of the pinned GGUF into a derived artifact, using ggml's own quantiser.
//
// A decode token reads 4.753 GB and this box sustains 237 GB/s, so the memory ceiling is 49.9 tok/s;
// reading fewer bytes per token is the only way past it, and the Q8_0 tensors cost 8.5 bits a weight.
// Using ggml_quantize_chunk rather than a hand-written quantiser means the sub-block scales and mins
// are exactly the library's, which is what keeps the error small.
//
//   requantise MODEL.gguf OUTPUT.bin TENSOR_NAME {q4_K|q6_K|q5_K}
//
// Prints the byte saving and the reconstruction error, so the quality cost is known before anything
// loads the artifact.
#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Tensor {
  std::string name;
  std::vector<std::int64_t> dims;
  std::uint32_t type{0};
  std::uint64_t offset{0};
};

bool ReadGguf(const char* path, std::vector<Tensor>* tensors, std::int64_t* data_start) {
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    return false;
  }
  char magic[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
    std::fclose(f);
    return false;
  }
  std::uint32_t version = 0;
  std::uint64_t n_tensors = 0;
  std::uint64_t n_kv = 0;
  std::fread(&version, 4, 1, f);
  std::fread(&n_tensors, 8, 1, f);
  std::fread(&n_kv, 8, 1, f);
  const auto read_str = [&]() {
    std::uint64_t n = 0;
    std::fread(&n, 8, 1, f);
    std::string s(n, '\0');
    std::fread(s.data(), 1, n, f);
    return s;
  };
  std::function<void(std::uint32_t)> skip;
  skip = [&](std::uint32_t kind) {
    static const int fixed[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (kind == 8) {
      read_str();
    } else if (kind == 9) {
      std::uint32_t element = 0;
      std::uint64_t count = 0;
      std::fread(&element, 4, 1, f);
      std::fread(&count, 8, 1, f);
      for (std::uint64_t i = 0; i < count; ++i) {
        skip(element);
      }
    } else if (kind < 13) {
      char buffer[8];
      std::fread(buffer, 1, static_cast<std::size_t>(fixed[kind]), f);
    }
  };
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    read_str();
    std::uint32_t kind = 0;
    std::fread(&kind, 4, 1, f);
    skip(kind);
  }
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    Tensor tensor;
    tensor.name = read_str();
    std::uint32_t n_dims = 0;
    std::fread(&n_dims, 4, 1, f);
    tensor.dims.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      std::fread(&tensor.dims[d], 8, 1, f);
    }
    std::fread(&tensor.type, 4, 1, f);
    std::fread(&tensor.offset, 8, 1, f);
    tensors->push_back(std::move(tensor));
  }
  *data_start = (static_cast<std::int64_t>(std::ftell(f)) + 31) / 32 * 32;
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: requantise MODEL.gguf OUTPUT.bin TENSOR_NAME {q4_K|q5_K|q6_K}\n");
    return 2;
  }
  ggml_type target = GGML_TYPE_Q4_K;
  const std::string want(argv[4]);
  if (want == "q5_K") {
    target = GGML_TYPE_Q5_K;
  } else if (want == "q6_K") {
    target = GGML_TYPE_Q6_K;
  } else if (want != "q4_K") {
    std::fprintf(stderr, "unsupported target %s\n", want.c_str());
    return 2;
  }

  std::vector<Tensor> tensors;
  std::int64_t data_start = 0;
  if (!ReadGguf(argv[1], &tensors, &data_start)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const Tensor* found = nullptr;
  for (const auto& tensor : tensors) {
    if (tensor.name == argv[3]) {
      found = &tensor;
    }
  }
  if (found == nullptr) {
    std::fprintf(stderr, "tensor %s not found in %s\n", argv[3], argv[1]);
    return 1;
  }
  std::int64_t count = 1;
  for (const auto d : found->dims) {
    count *= d;
  }
  const std::int64_t n_per_row = found->dims.empty() ? count : found->dims[0];
  const std::int64_t nrows = count / n_per_row;
  if (count % n_per_row != 0 || n_per_row % 256 != 0) {
    std::fprintf(stderr, "tensor width %lld is not a multiple of 256\n",
                 static_cast<long long>(n_per_row));
    return 1;
  }

  const std::size_t source_bytes = ggml_row_size(static_cast<ggml_type>(found->type), n_per_row) *
                                  static_cast<std::size_t>(nrows);
  std::vector<std::uint8_t> source(source_bytes);
  FILE* in = std::fopen(argv[1], "rb");
  std::fseek(in, static_cast<long>(data_start + static_cast<std::int64_t>(found->offset)), SEEK_SET);
  if (std::fread(source.data(), 1, source_bytes, in) != source_bytes) {
    std::fclose(in);
    std::fprintf(stderr, "short read on %s\n", found->name.c_str());
    return 1;
  }
  std::fclose(in);

  const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(found->type));
  std::vector<float> values(static_cast<std::size_t>(count));
  traits->to_float(source.data(), values.data(), count);

  std::vector<std::uint8_t> derived(ggml_row_size(target, n_per_row) *
                                    static_cast<std::size_t>(nrows));
  ggml_quantize_chunk(target, values.data(), derived.data(), 0, nrows, n_per_row, nullptr);

  std::vector<float> restored(static_cast<std::size_t>(count));
  ggml_get_type_traits(target)->to_float(derived.data(), restored.data(), count);

  double error = 0.0;
  double magnitude = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double d = static_cast<double>(values[i]) - restored[i];
    error += d * d;
    magnitude += static_cast<double>(values[i]) * values[i];
  }
  const double rms = std::sqrt(error / static_cast<double>(count));
  const double relative = magnitude > 0.0 ? std::sqrt(error / magnitude) : 0.0;

  FILE* out = std::fopen(argv[2], "wb");
  if (out == nullptr || std::fwrite(derived.data(), 1, derived.size(), out) != derived.size()) {
    std::fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }
  std::fclose(out);

  std::printf("%s: %zu -> %zu bytes (%.1f%% smaller), rms %.6f, relative %.5f, %s\n",
              found->name.c_str(), source_bytes, derived.size(),
              100.0 * (1.0 - static_cast<double>(derived.size()) /
                                 static_cast<double>(source_bytes)),
              rms, relative, ggml_type_name(target));
  return 0;
}
