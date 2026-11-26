// End-to-end forward: load the model on the GPU, decode a few tokens and
// report the token ids and the decode rate.
//
// Usage: forward_test [model.gguf] [steps] [start-token]
//   (defaults: $LING3_GGUF, 8 steps, 156891 = <|startoftext|>)

#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include <cmath>
#include <cstring>

#include <gufo/ext/experimental/ggml_dequant.hpp>

#include "src/config.hpp"
#include "src/engine.hpp"
#include "src/text/tokenizer.hpp"
#include "src/weights.hpp"

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;

  std::string model_path;
  if (argc > 1) {
    model_path = argv[1];
  } else if (const char* env = std::getenv("LING3_GGUF")) {
    model_path = env;
  }
  if (model_path.empty()) {
    std::printf("SKIP: set LING3_GGUF or pass the model .gguf path\n");
    return 77;
  }
  const int steps = argc > 2 ? std::atoi(argv[2]) : 8;
  std::string prompt = argc > 3 ? argv[3] : std::string();
  // A file avoids shell quoting for prompts with markers like <|role_end|>.
  if (const char* file = std::getenv("LING3_PROMPT_FILE")) {
    std::ifstream input(file, std::ios::binary);
    if (!input) {
      std::fprintf(stderr, "cannot read %s\n", file);
      return 2;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    prompt = buffer.str();
  }
  const bool use_prompt = !prompt.empty();

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(), error.c_str());
    return 2;
  }
  ling::Ling3Config config;
  if (!ling::ParseConfig(*reader, &config, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  ling::Ling3Weights weights;
  if (!ling::LoadWeights(*reader, config, &weights, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  ling::EngineOptions options;
  options.max_context = 4096;
  if (const char* ctx = std::getenv("LING3_CONTEXT")) {
    options.max_context = static_cast<std::uint32_t>(std::atoi(ctx));
  }
  const auto create_start = std::chrono::steady_clock::now();
  auto engine = ling::Engine::Create(*reader, config, weights, options, &error);
  if (!engine) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - create_start)
          .count();
  std::printf("loaded %.2f GiB of weights in %.1f s (vocab %u)\n",
              engine->DeviceBytes() / 1073741824.0, load_seconds,
              engine->VocabSize());

  std::vector<ling::text::TokenId> prompt_tokens;
  auto tokenizer = ling::text::Tokenizer::CreateFromGguf(*reader, &error);
  if (use_prompt) {
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
      return 1;
    }
    const std::string framed =
        std::getenv("LING3_RAW_PROMPT") != nullptr
            ? prompt
            : ("<role>SYSTEM</role>detailed thinking off<|role_end|>"
               "<role>HUMAN</role>" + prompt +
               "<|role_end|><role>ASSISTANT</role>\n<think" ">");
    prompt_tokens = tokenizer->Encode(framed, /*parse_special=*/true);
    std::printf("prompt: %zu tokens\n", prompt_tokens.size());
  }
  // The embedding row is the engine's only custom dequantization path.
  {
    std::vector<float> device_row(weights.token_embd.Cols() == 0 ? config.n_embd
                                                                 : config.n_embd);
    const std::int32_t probe_token = 1000;
    engine->DebugEmbedding(probe_token, device_row.data());
    std::vector<float> reference(config.n_embd, 0.0f);
    const std::size_t row_bytes =
        weights.token_embd.bytes / static_cast<std::size_t>(config.n_vocab);
    const auto* base = static_cast<const std::uint8_t*>(weights.token_embd.data) +
                       static_cast<std::size_t>(probe_token) * row_bytes;
    std::printf("embedding row %d: %zu bytes/row\n", probe_token, row_bytes);
    double worst = 0.0;
    double magnitude = 0.0;
    if (weights.token_embd.type == gufo::core::GgmlType::kQ8_0) {
      for (std::size_t b = 0; b < config.n_embd / 32; ++b) {
        const auto* block = base + b * 34;
        std::uint16_t bits;
        std::memcpy(&bits, block, sizeof(bits));
        const float d = gufo::quant::Fp16ToFloat(bits);
        for (int j = 0; j < 32; ++j) {
          const float value =
              d * static_cast<float>(static_cast<std::int8_t>(block[2 + j]));
          const std::size_t index = b * 32 + j;
          worst = std::fabs(static_cast<double>(device_row[index]) - value) > worst
                      ? std::fabs(static_cast<double>(device_row[index]) - value)
                      : worst;
          magnitude = std::fabs(value) > magnitude ? std::fabs(value) : magnitude;
        }
      }
    }
    std::printf("embedding check: worst abs error %.3g (values up to %.3g)\n", worst,
                magnitude);
  }

  // Chunked path, stage 1: embeddings plus the M-row RMSNorm, against a host
  // computation from the same embeddings and block 0's norm weight.
  if (use_prompt && std::getenv("LING3_CHUNK_STAGE1") != nullptr) {
    const std::size_t rows = prompt_tokens.size() < 8 ? prompt_tokens.size() : 8;
    std::vector<std::int32_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) ids[i] = static_cast<std::int32_t>(prompt_tokens[i]);
    std::vector<float> x(rows * config.n_embd), h(rows * config.n_embd);
    engine->DebugChunkStage1(ids.data(), rows, x.data(), h.data());
    const auto* norm_weight =
        static_cast<const float*>(weights.layers[0].attn_norm.data);
    double worst = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (std::size_t i = 0; i < config.n_embd; ++i) {
        const double v = x[r * config.n_embd + i];
        sum += v * v;
      }
      const double scale = 1.0 / std::sqrt(sum / config.n_embd + config.rms_eps);
      for (std::size_t i = 0; i < config.n_embd; ++i) {
        const double expected = x[r * config.n_embd + i] * scale * norm_weight[i];
        const double diff = std::fabs(h[r * config.n_embd + i] - expected);
        worst = diff > worst ? diff : worst;
      }
    }
    std::printf("chunk stage1: %zu rows, norm worst abs diff %.3g\n", rows, worst);
    const float projection = engine->DebugChunkProjectionDiff(rows);
    std::printf("chunk stage2: batched projection vs per-row GEMV worst abs diff %.3g\n",
                projection);
    return (worst <= 1e-5 && projection <= 1e-4) ? 0 : 1;
  }

  std::int32_t token = use_prompt ? prompt_tokens.front() : 156891;
  std::vector<ling::text::TokenId> produced;
  std::size_t position = 0;
  const auto start_time = std::chrono::steady_clock::now();
  if (use_prompt) {
    for (std::size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
      if (!engine->ForwardToken(static_cast<std::int32_t>(prompt_tokens[i]),
                                static_cast<std::uint32_t>(position++),
                                &error)) {
        std::fprintf(stderr, "prefill: %s\n", error.c_str());
        return 1;
      }
    }
    token = static_cast<std::int32_t>(prompt_tokens.back());
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
            .count();
    std::printf("prefill: %zu tokens in %.3f s = %.2f tok/s\n",
                prompt_tokens.size() - 1, prefill_seconds,
                prompt_tokens.size() > 1 ? (prompt_tokens.size() - 1) / prefill_seconds
                                         : 0.0);
  }
  // Optional depth: fill the cache with one repeated token before timing, so
  // decode is measured with a long context already in place.
  std::size_t depth = 0;
  if (const char* arg = std::getenv("LING3_DEPTH")) {
    depth = static_cast<std::size_t>(std::atoi(arg));
  }
  for (std::size_t i = 0; i < depth; ++i) {
    if (!engine->ForwardToken(156891, static_cast<std::uint32_t>(position++), &error)) {
      std::fprintf(stderr, "prefill: %s\n", error.c_str());
      return 1;
    }
  }
  // MTP: after a trunk step, block 42 predicts the token *after* the one the
  // trunk just picked. Compare that prediction with what the trunk produces on
  // its next step; agreement is what speculative decoding needs to be exact.
  const bool check_mtp = std::getenv("LING3_MTP") != nullptr;
  std::int32_t mtp_prediction = -1;
  std::size_t mtp_compared = 0;
  std::size_t mtp_agreed = 0;
  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < steps; ++step) {
    const auto step_start = std::chrono::steady_clock::now();
    if (!engine->ForwardToken(token, static_cast<std::uint32_t>(position++), &error)) {
      std::fprintf(stderr, "step %d: %s\n", step, error.c_str());
      return 1;
    }
    token = engine->Argmax();
    if (check_mtp) {
      if (mtp_prediction >= 0) {
        ++mtp_compared;
        if (mtp_prediction == token) {
          ++mtp_agreed;
        }
      }
      // Draft from this step's hidden state: block 42 at the next position.
      if (!engine->ForwardMtp(token, static_cast<std::uint32_t>(position), &error)) {
        std::fprintf(stderr, "mtp: %s\n", error.c_str());
        return 1;
      }
      mtp_prediction = engine->Argmax();
    }
    produced.push_back(static_cast<ling::text::TokenId>(token));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - step_start)
                          .count();
    std::printf("step %d: token %d in %.1f ms\n", step, static_cast<int>(token), ms);
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start)
          .count();
  const double total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
          .count();
  (void)total_seconds;

  std::printf("tokens:");
  for (const auto id : produced) {
    std::printf(" %d", static_cast<int>(id));
  }
  std::printf("\n%.2f tok/s over %d steps (%.3f s)\n", steps / seconds, steps,
              seconds);
  if (check_mtp) {
    std::printf("mtp: %zu of %zu predictions matched the trunk\n", mtp_agreed,
                mtp_compared);
  }
  if (use_prompt && tokenizer != nullptr) {
    std::printf("text: %s\n", tokenizer->Decode(produced, false).c_str());
    std::printf("tokens with control:");
    for (const auto id : produced) {
      const std::string piece = tokenizer->Decode({id}, true);
      std::printf(" [%u]%s", static_cast<unsigned>(id), piece.c_str());
    }
    std::printf("\n");
  }
  {
    std::vector<float> logits(engine->VocabSize());
    engine->ReadLogits(logits.data());
    std::size_t nan_count = 0;
    float minimum = logits[0];
    float maximum = logits[0];
    for (const float value : logits) {
      if (!(value == value)) {
        ++nan_count;
        continue;
      }
      minimum = value < minimum ? value : minimum;
      maximum = value > maximum ? value : maximum;
    }
    std::vector<std::size_t> order(logits.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                      [&logits](std::size_t a, std::size_t b) {
                        return logits[a] > logits[b];
                      });
    std::printf("logits: nan %zu, min %.3f, max %.3f; top5:", nan_count, minimum,
                maximum);
    for (int i = 0; i < 5; ++i) {
      std::printf(" %zu=%.3f", order[i], logits[order[i]]);
    }
    std::printf("\n");
  }
  return 0;
}
