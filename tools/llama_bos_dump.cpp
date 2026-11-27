// Dump llama.cpp's per-block residual stream after a single BOS, as raw FP32, so
// the first layer where this extension and the reference disagree can be named.
#include <llama.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace {
std::string g_dir;
int g_written = 0;

bool Dump(struct ggml_tensor* t) {
  if (t == nullptr || ggml_n_dims(t) != 1 || t->type != GGML_TYPE_F32) return true;
  const std::string name(t->name);
  // One file per block's residual stream, plus the final norm and logits.
  if (name.rfind("l_out-", 0) != 0 && name != "result_norm" && name != "result_output" &&
      name != "model.input_embed")
    return true;
  std::ofstream out(g_dir + "/" + name + ".f32", std::ios::binary);
  out.write(static_cast<const char*>(t->data), static_cast<std::streamsize>(ggml_nbytes(t)));
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", name.c_str());
    return false;
  }
  ++g_written;
  return true;
}

bool Wanted(const struct ggml_tensor* t) {
  const std::string name(t->name);
  return name.rfind("l_out-", 0) == 0 || name == "result_norm" || name == "result_output" ||
         name == "model.input_embed";
}

bool Callback(struct ggml_tensor* t, bool ask, void*) {
  // llama.cpp calls once with ask=true to learn what to report, then again after
  // evaluation, when the tensor's data is valid: dumping on the first call reads
  // an uninitialised buffer.
  if (!Wanted(t)) return false;
  if (ask) return true;
  if (!Dump(t)) {
    std::fprintf(stderr, "dump failed at %s\n", t->name);
    return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: llama_bos_dump MODEL OUTDIR\n");
    return 2;
  }
  g_dir = argv[2];
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  auto* model = llama_model_load_from_file(argv[1], mp);
  if (model == nullptr) return 1;
  auto params = llama_context_default_params();
  params.n_ctx = 4096;
  params.n_threads = params.n_threads_batch = 16;
  params.cb_eval = &Callback;
  params.cb_eval_user_data = nullptr;
  auto* ctx = llama_init_from_model(model, params);
  const auto* vocab = llama_model_get_vocab(model);
  llama_token bos = llama_vocab_bos(vocab);
  if (ctx == nullptr || bos < 0 || llama_decode(ctx, llama_batch_get_one(&bos, 1)) != 0) return 1;
  std::printf("bos=%d tensors=%d\n", bos, g_written);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return g_written > 0 ? 0 : 1;
}
