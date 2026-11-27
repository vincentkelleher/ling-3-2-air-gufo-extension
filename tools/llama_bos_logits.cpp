// Dump one full CPU logit row after exactly one BOS (no chat or CLI framing).
#include <llama.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

// Optional: an explicit id list, so multi-token contexts can be compared too.
static std::vector<llama_token> ParseIds(const char* text) {
  std::vector<llama_token> ids;
  for (const char* p = text; *p != '\0';) {
    char* end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p) { if (*p == ',') { ++p; continue; } break; }
    ids.push_back(static_cast<llama_token>(value));
    p = end;
  }
  return ids;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: llama_bos_logits MODEL OUTPUT.f32 [IDS]\n"); return 2; }
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  auto* model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 1;
  auto params = llama_context_default_params();
  params.n_ctx = 4096;
  params.n_threads = params.n_threads_batch = 16;
  auto* ctx = llama_init_from_model(model, params);
  const auto* vocab = llama_model_get_vocab(model);
  const auto ids = argc > 3 ? ParseIds(argv[3])
                            : std::vector<llama_token>{llama_vocab_bos(vocab)};
  if (!ctx || ids.empty() || llama_decode(ctx, llama_batch_get_one(
                                  const_cast<llama_token*>(ids.data()),
                                  static_cast<int32_t>(ids.size()))) != 0) return 1;
  const auto* row = llama_get_logits_ith(ctx, static_cast<int32_t>(ids.size()) - 1);
  const auto n = llama_vocab_n_tokens(vocab);
  if (!row || n <= 0) return 1;
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(row), n * sizeof(float));
  out.close();
  std::printf("tokens=%zu vocab=%d argmax=%td\n", ids.size(), n, std::max_element(row, row+n)-row);
  llama_free(ctx); llama_model_free(model); llama_backend_free();
  return out ? 0 : 1;
}
