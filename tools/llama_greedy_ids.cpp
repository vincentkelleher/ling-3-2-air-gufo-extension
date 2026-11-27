// Greedy decode from an explicit token-id list, printing ids, so the comparison against
// this engine starts from a byte-identical context instead of two framings of a prompt.
#include <llama.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<llama_token> ParseIds(const char* text) {
  std::vector<llama_token> ids;
  for (const char* p = text; *p != '\0';) {
    char* end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p) {
      if (*p == ',') { ++p; continue; }
      break;
    }
    ids.push_back(static_cast<llama_token>(value));
    p = end;
  }
  return ids;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: llama_greedy_ids MODEL IDS_CSV STEPS\n");
    return 2;
  }
  const int steps = std::atoi(argv[3]);
  auto ids = ParseIds(argv[2]);
  if (ids.empty()) return 2;
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  auto* model = llama_model_load_from_file(argv[1], mp);
  if (model == nullptr) return 1;
  auto cp = llama_context_default_params();
  cp.n_ctx = 4096;
  cp.n_threads = cp.n_threads_batch = 16;
  auto* ctx = llama_init_from_model(model, cp);
  if (ctx == nullptr) return 1;
  if (llama_decode(ctx, llama_batch_get_one(ids.data(), static_cast<int32_t>(ids.size()))) != 0) return 1;
  const auto* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  std::printf("ids");
  for (int step = 0; step < steps; ++step) {
    const float* row = llama_get_logits_ith(ctx, -1);
    if (row == nullptr) return 1;
    llama_token next = static_cast<llama_token>(std::max_element(row, row + n_vocab) - row);
    std::printf(" %d", static_cast<int>(next));
    if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) return 1;
  }
  std::printf("\n");
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
