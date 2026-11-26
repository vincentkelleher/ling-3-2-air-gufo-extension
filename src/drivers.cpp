// `gufo prompt`, `gufo chat` and `gufo bench` for Ling 3.2 Air.
//
// These drivers run the model directly, without the component cache, so the
// extension is usable from the CLI while the `TextModelRunner` (which the server
// needs, see notes/runner-plan.md) is still being built. Sampling is done here on
// the host from the logits the engine copies back.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gufo/ext/generated/ling3_air_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/config.hpp"
#include "src/engine.hpp"
#include "src/text/chat_template.hpp"
#include "src/text/tokenizer.hpp"
#include "src/probe.hpp"
#include "src/weights.hpp"

namespace gufo::ext::ling3_air {
namespace {

using Clock = std::chrono::steady_clock;

/// Everything a driver needs, loaded once.
struct Loaded {
  Ling3Config config;
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::unique_ptr<Engine> engine;
};

bool Load(const core::GgufReader& reader, std::uint32_t context, Loaded* out,
          std::string* error) {
  if (!ParseConfig(reader, &out->config, error)) {
    return false;
  }
  Ling3Weights weights;
  if (!LoadWeights(reader, out->config, &weights, error)) {
    return false;
  }
  out->tokenizer = text::Tokenizer::CreateFromGguf(reader, error);
  if (out->tokenizer == nullptr) {
    return false;
  }
  EngineOptions options;
  options.max_context = context;
  out->engine = Engine::Create(reader, out->config, weights, options, error);
  return out->engine != nullptr;
}

/// Greedy when the configuration asks for it, otherwise temperature with top-k
/// and top-p over the copied logits. Integer seeds keep runs reproducible.
std::int32_t Sample(std::vector<float>& logits, const sampling::SamplingConfig& cfg,
                    std::mt19937_64& rng) {
  if (!cfg.uses_random_sampling() || cfg.temperature <= 0.0f) {
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }
  const float inverse = 1.0f / cfg.temperature;
  for (float& value : logits) {
    value *= inverse;
  }
  // top-k
  if (cfg.top_k > 0 && static_cast<std::size_t>(cfg.top_k) < logits.size()) {
    std::vector<float> sorted(logits.begin(), logits.end());
    std::nth_element(sorted.begin(), sorted.begin() + (cfg.top_k - 1), sorted.end(),
                     std::greater<float>());
    const float threshold = sorted[static_cast<std::size_t>(cfg.top_k) - 1];
    for (float& value : logits) {
      if (value < threshold) {
        value = -INFINITY;
      }
    }
  }
  // softmax
  const float maximum = *std::max_element(logits.begin(), logits.end());
  double total = 0.0;
  for (float& value : logits) {
    value = std::exp(value - maximum);
    total += value;
  }
  if (total <= 0.0) {
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }
  for (float& value : logits) {
    value = static_cast<float>(value / total);
  }
  // top-p
  if (cfg.top_p < 1.0f) {
    std::vector<std::size_t> order(logits.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + std::min<std::size_t>(256, order.size()),
                      order.end(),
                      [&logits](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
    double cumulative = 0.0;
    for (const std::size_t index : order) {
      cumulative += logits[index];
      if (cumulative >= cfg.top_p) {
        const float keep = logits[index];
        for (float& value : logits) {
          if (value < keep) {
            value = 0.0f;
          }
        }
        break;
      }
    }
    double renorm = 0.0;
    for (const float value : logits) renorm += value;
    if (renorm > 0.0) {
      for (float& value : logits) value = static_cast<float>(value / renorm);
    }
  }
  std::discrete_distribution<std::size_t> pick(logits.begin(), logits.end());
  return static_cast<std::int32_t>(pick(rng));
}

/// Frame a single user turn with the model's template.
std::string FrameSingleTurn(const std::string& system, const std::string& user,
                            bool thinking, bool use_template) {
  if (!use_template) {
    return user;
  }
  std::vector<text::ChatTurn> turns;
  if (!system.empty()) {
    text::ChatTurn turn;
    turn.role = "system";
    turn.parts.push_back({system, false});
    turns.push_back(std::move(turn));
  }
  text::ChatTurn turn;
  turn.role = "user";
  turn.parts.push_back({user, false});
  turns.push_back(std::move(turn));
  text::ChatTemplateOptions options;
  options.enable_thinking = thinking;
  return text::RenderChat(turns, options);
}

/// Prefill in chunks, then decode, calling `on_token` for every emitted id.
struct RunResult {
  std::size_t prompt_tokens{0};
  std::size_t generated{0};
  double prefill_seconds{0.0};
  double decode_seconds{0.0};
};

bool Generate(Loaded& model, const std::vector<text::TokenId>& prompt,
              const sampling::SamplingConfig& sampling, std::size_t max_tokens,
              const std::function<void(text::TokenId, const std::string&)>& on_token,
              RunResult* result, std::string* error) {
  Engine& engine = *model.engine;
  const auto& config = model.config;
  std::mt19937_64 rng(sampling.seed >= 0 ? static_cast<std::uint64_t>(sampling.seed)
                                         : std::random_device{}());
  std::vector<float> logits(config.n_vocab);
  std::size_t position = 0;
  const auto prefill_start = Clock::now();
  // Leave the last prompt token for the first decode step, as the server does.
  std::vector<std::int32_t> ids(prompt.begin(), prompt.end());
  std::size_t consumed = 0;
  const std::size_t last = ids.empty() ? 0 : ids.size() - 1;
  while (consumed < last) {
    const std::size_t chunk = std::min<std::size_t>(32, last - consumed);
    if (!engine.PrefillChunk(ids.data() + consumed, chunk,
                             static_cast<std::uint32_t>(consumed), -1, error)) {
      return false;
    }
    consumed += chunk;
  }
  position = consumed;
  result->prefill_seconds =
      std::chrono::duration<double>(Clock::now() - prefill_start).count();
  if (ids.empty()) {
    return true;
  }
  std::int32_t token = ids.back();
  const auto decode_start = Clock::now();
  for (std::size_t step = 0; step < max_tokens; ++step) {
    if (!engine.ForwardToken(token, static_cast<std::uint32_t>(position), error)) {
      return false;
    }
    ++position;
    engine.ReadLogits(logits.data());
    token = Sample(logits, sampling, rng);
    if (token == static_cast<std::int32_t>(model.tokenizer->Eos()) ||
        token == 156892 /* <|endoftext|> */) {
      break;
    }
    const std::string piece = model.tokenizer->Decode({static_cast<text::TokenId>(token)});
    on_token(static_cast<text::TokenId>(token), piece);
    ++result->generated;
    std::fflush(stdout);
  }
  result->decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();
  result->prompt_tokens = ids.size();
  return true;
}

int RunPromptLike(const TextPromptOptions& options, const core::GgufReader& reader,
                  const std::string& prompt_text) {
  Loaded model;
  std::string error;
  if (!Load(reader, options.max_tokens > 0 ? 4096 : 4096, &model, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto thinking = options.thinking_mode().value_or(true);
  const std::string framed = FrameSingleTurn(options.system_prompt, prompt_text, thinking,
                                             options.use_chat_template);
  const auto ids = model.tokenizer->Encode(framed, /*parse_special=*/true);
  if (options.display_prompt) {
    std::printf("%s", framed.c_str());
  }
  RunResult result;
  const bool ok = Generate(
      model, ids, options.sampling, options.max_tokens,
      [](text::TokenId, const std::string& piece) { std::fputs(piece.c_str(), stdout); },
      &result, &error);
  std::printf("\n");
  if (!ok) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::fprintf(stderr, "\nprefill %.2f tok/s, decode %.2f tok/s (%zu prompt, %zu generated)\n",
               result.prefill_seconds > 0 ? result.prompt_tokens / result.prefill_seconds : 0.0,
               result.decode_seconds > 0 ? result.generated / result.decode_seconds : 0.0,
               result.prompt_tokens, result.generated);
  return 0;
}

}  // namespace

int RunLing3Prompt(const TextPromptOptions& options, const core::GgufReader& reader,
                   std::chrono::steady_clock::time_point) {
  std::string prompt = options.prompt_text;
  if (!options.prompt_file.empty()) {
    FILE* file = std::fopen(options.prompt_file.c_str(), "rb");
    if (file != nullptr) {
      prompt.clear();
      char buffer[4096];
      std::size_t read = 0;
      while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        prompt.append(buffer, read);
      }
      std::fclose(file);
    }
  }
  return RunPromptLike(options, reader, prompt);
}

int RunLing3Chat(const TextPromptOptions& options, const core::GgufReader& reader,
                 std::chrono::steady_clock::time_point load_start) {
  return RunLing3Prompt(options, reader, load_start);
}

int RunLing3Bench(const TextBenchOptions& options,
                  const std::shared_ptr<const core::GgufReader>& reader,
                  std::chrono::steady_clock::time_point) {
  if (reader == nullptr) {
    std::fprintf(stderr, "bench needs the model file\n");
    return 1;
  }
  Loaded model;
  std::string error;
  const std::size_t context = options.n_depths.empty()
                                  ? 4096
                                  : std::max<std::size_t>(4096, options.n_depths.back() + 4096);
  if (!Load(*reader, static_cast<std::uint32_t>(context), &model, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  // A synthetic prompt of the requested length, repeated from a fixed sentence.
  const std::string seed_text =
      "The quick brown fox jumps over the lazy dog and keeps running through the field. ";
  std::vector<text::TokenId> seed = model.tokenizer->Encode(seed_text, true);
  for (const std::size_t n_prompt : options.n_prompts) {
    for (const std::size_t n_gen : options.n_gens) {
      for (const std::size_t depth : options.n_depths) {
        std::vector<text::TokenId> prompt;
        prompt.reserve(n_prompt + depth);
        while (prompt.size() < n_prompt + depth) {
          prompt.insert(prompt.end(), seed.begin(), seed.end());
        }
        prompt.resize(n_prompt + depth);
        RunResult result;
        const bool ok = Generate(
            model, prompt, options.sampling, n_gen,
            [](text::TokenId, const std::string&) {}, &result, &error);
        if (!ok) {
          std::fprintf(stderr, "%s\n", error.c_str());
          return 1;
        }
        std::printf("pp%-6zu tg%-5zu depth %-6zu: prefill %8.2f tok/s, decode %8.2f tok/s\n",
                    n_prompt, n_gen, depth,
                    result.prefill_seconds > 0 ? result.prompt_tokens / result.prefill_seconds : 0.0,
                    result.decode_seconds > 0 ? result.generated / result.decode_seconds : 0.0);
      }
    }
  }
  return 0;
}

}  // namespace gufo::ext::ling3_air

void ling3_air_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_ling3_air::kProvider_ling3_air);
  registry.AddTextProvider({
      .id = "ling3-air",
      .probe = &gufo::ext::ling3_air::ProbeLing3,
      .probe_gguf = &gufo::ext::ling3_air::ProbeLing3Gguf,
      .create = nullptr,
      .bench_driver = &gufo::ext::ling3_air::RunLing3Bench,
      .prompt_driver = &gufo::ext::ling3_air::RunLing3Prompt,
      .chat_driver = &gufo::ext::ling3_air::RunLing3Chat,
  });
}
