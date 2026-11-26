// Does reading eight scattered 1.4 MB windows out of a registered 50 GiB mapping run as fast as
// reading one contiguous span of the same total size? The MoE sits at ~40 GB/s while the
// sequential probe reaches 237 GB/s, and this decides whether that is the kernel or the memory.
//
// build: hipcc -O3 tools/scatter_probe.hip -o scatter_probe
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__global__ void SumBytes(const unsigned char* base, std::size_t bytes, const std::size_t* offsets,
                         int windows, float* out) {
  std::size_t total = 0;
  // One window per block: each block walks its window in 16-byte steps, so the read pattern
  // matches what an expert GEMV sees (contiguous inside a window, scattered across windows).
  const int window = blockIdx.x;
  const std::size_t begin = window < windows ? offsets[window] : 0;
  const std::size_t length = window < windows ? bytes : 0;
  const auto* data = reinterpret_cast<const unsigned int*>(base + begin);
  unsigned int sum = 0;
  for (std::size_t i = threadIdx.x; i < length / 4; i += blockDim.x) {
    sum += data[i];
  }
  atomicAdd(out, static_cast<float>(sum & 0xFFFF));
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t total_mb = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int windows = argc > 2 ? std::atoi(argv[2]) : 8;
  const std::size_t window_bytes = argc > 3 ? std::atoll(argv[3]) : (1400 * 1024);
  const std::size_t total = total_mb * 1024ull * 1024ull;
  std::vector<unsigned char> host(total, 1);
  // Register like the engine does, so the read path is the same one decode uses.
  if (hipHostRegister(host.data(), total, hipHostRegisterMapped) != hipSuccess) {
    std::fprintf(stderr, "hipHostRegister failed for %zu MiB\n", total_mb);
    return 1;
  }
  void* device = nullptr;
  if (hipHostGetDevicePointer(&device, host.data(), 0) != hipSuccess || device == nullptr) {
    std::fprintf(stderr, "hipHostGetDevicePointer failed\n");
    return 1;
  }
  std::vector<std::size_t> offsets(windows);
  for (int i = 0; i < windows; ++i) {
    offsets[i] = (total / windows) * static_cast<std::size_t>(i);
  }
  std::size_t* device_offsets = nullptr;
  float* device_out = nullptr;
  hipMalloc(reinterpret_cast<void**>(&device_offsets), windows * sizeof(std::size_t));
  hipMalloc(reinterpret_cast<void**>(&device_out), sizeof(float));
  hipMemcpy(device_offsets, offsets.data(), windows * sizeof(std::size_t),
            hipMemcpyHostToDevice);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  for (int repeat = 0; repeat < 2; ++repeat) {
    hipEventRecord(start, nullptr);
    for (int step = 0; step < 10; ++step) {
      SumBytes<<<windows, 256>>>(static_cast<const unsigned char*>(device), window_bytes,
                                 device_offsets, windows, device_out);
    }
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    const double gb = static_cast<double>(windows) * window_bytes * 10 / 1e9;
    std::printf("scattered: %d windows x %zu KiB, %d rounds: %.1f ms -> %.1f GB/s%s\n", windows,
                window_bytes / 1024, 10, ms, gb / (ms / 1000.0), repeat == 0 ? " (warm-up)" : "");
  }
  hipHostUnregister(host.data());
  return 0;
}
