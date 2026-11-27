// Dump one full CPU logit row after exactly one BOS (no chat or CLI framing).
#include <llama.h>
#include <algorithm>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: llama_bos_logits MODEL OUTPUT.f32\n"); return 2; }
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
  llama_token bos = llama_vocab_bos(vocab);
  if (!ctx || bos < 0 || llama_decode(ctx, llama_batch_get_one(&bos, 1)) != 0) return 1;
  const auto* row = llama_get_logits_ith(ctx, 0);
  const auto n = llama_vocab_n_tokens(vocab);
  if (!row || n <= 0) return 1;
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(row), n * sizeof(float));
  out.close();
  std::printf("bos=%d vocab=%d argmax=%td\n", bos, n, std::max_element(row, row+n)-row);
  llama_free(ctx); llama_model_free(model); llama_backend_free();
  return out ? 0 : 1;
}
