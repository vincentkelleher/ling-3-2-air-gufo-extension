// Can the GPU read the model file's read-only mmap in place (zero copy)?
//
// On Strix Halo host memory is device-accessible. If hipHostRegister accepts the
// read-only mapping, the engine can skip copying 80 GiB into a second buffer
// (which is what made the first end-to-end run thrash).
//
// hipcc -O3 --offload-arch=gfx1151 tools/zerocopy_probe.hip -o zerocopy_probe
// ./zerocopy_probe <file> [GiB]
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

__global__ void ReadKernel(const float4* in, std::size_t count, float* out) {
  float4 sum = make_float4(0, 0, 0, 0);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const float4 v = in[i];
    sum.x += v.x;
    sum.y += v.y;
    sum.z += v.z;
    sum.w += v.w;
  }
  const float total = sum.x + sum.y + sum.z + sum.w;
  if (total == -0.0f) {
    *out = total;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <file> [GiB]\n", argv[0]);
    return 2;
  }
  const double gib = argc > 2 ? std::atof(argv[2]) : 0.25;
  const std::size_t bytes = static_cast<std::size_t>(gib * 1073741824.0);
  const int fd = open(argv[1], O_RDONLY);
  if (fd < 0) {
    std::perror("open");
    return 2;
  }
  void* map = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    std::perror("mmap");
    return 2;
  }

  hipError_t status = hipHostRegister(map, bytes, hipHostRegisterMapped);
  std::printf("hipHostRegister(read-only mmap): %s\n", hipGetErrorString(status));
  if (status != hipSuccess) {
    return 3;
  }
  void* device = nullptr;
  status = hipHostGetDevicePointer(&device, map, 0);
  std::printf("hipHostGetDevicePointer: %s, pointer %s the mapping\n",
              hipGetErrorString(status), device == map ? "is" : "differs from");

  float* sink = nullptr;
  if (hipMalloc(&sink, sizeof(float)) != hipSuccess) {
    std::printf("sink allocation failed\n");
    return 3;
  }
  const std::size_t count = bytes / sizeof(float4);
  auto read_gbs = [&](const float4* pointer) {
    hipEvent_t a;
    hipEvent_t b;
    hipEventCreate(&a);
    hipEventCreate(&b);
    hipEventRecord(a);
    ReadKernel<<<256 * 32, 256>>>(pointer, count, sink);
    hipEventRecord(b);
    hipEventSynchronize(b);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a);
    hipEventDestroy(b);
    return bytes / (ms / 1000.0) / 1e9;
  };
  const double first = read_gbs(static_cast<const float4*>(device));
  const double second = read_gbs(static_cast<const float4*>(device));

  double copied = 0.0;
  float4* buffer = nullptr;
  if (hipMalloc(&buffer, bytes) == hipSuccess &&
      hipMemcpy(buffer, device, bytes, hipMemcpyDeviceToDevice) == hipSuccess) {
    copied = read_gbs(buffer);
    hipFree(buffer);
  }
  std::printf("zero-copy read %.1f GB/s, warm %.1f GB/s, device copy read %.1f GB/s (%.2f GiB)\n",
              first, second, copied, gib);
  hipHostUnregister(map);
  munmap(map, bytes);
  close(fd);
  return 0;
}
