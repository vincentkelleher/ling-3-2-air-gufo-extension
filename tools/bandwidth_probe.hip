// Unified-memory bandwidth probe for the roofline the extension is measured
// against.
//
// Three kernels over a large buffer: a read-only reduction (the shape of a
// decode step reading weights), a copy (read + write), and a write stream.
// Each is timed with HIP events and reported in GB/s of the bytes the memory
// system actually moves, so the read number is the ceiling a GEMV can reach.
//
// Build (in Gufo's devshell, which provides hipcc):
//   hipcc -O3 --offload-arch=gfx1151 tools/bandwidth_probe.hip -o bandwidth_probe
// Run:
//   ./bandwidth_probe [GiB] [iterations]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

/// Reads every element of the buffer once and folds it into `out` so the
/// compiler cannot drop the loads.
__global__ void ReadKernel(const float4* __restrict__ in, std::size_t count,
                           float* __restrict__ out) {
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const float4 value = in[i];
    sum.x += value.x;
    sum.y += value.y;
    sum.z += value.z;
    sum.w += value.w;
  }
  const float total = sum.x + sum.y + sum.z + sum.w;
  if (total == -0.0f) {
    *out = total;  // unreachable in practice; keeps the reads alive
  }
}

__global__ void CopyKernel(float4* __restrict__ dst, const float4* __restrict__ src,
                           std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void WriteKernel(float4* __restrict__ dst, std::size_t count) {
  const float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    dst[i] = value;
  }
}

struct Result {
  double read_gbs;
  double copy_gbs;
  double write_gbs;
};

double Millis(hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0f;
  Check(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
  return static_cast<double>(ms);
}

}  // namespace

int main(int argc, char** argv) {
  const double gib = argc > 1 ? std::atof(argv[1]) : 8.0;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 3;

  int device = 0;
  Check(hipGetDevice(&device), "hipGetDevice");
  hipDeviceProp_t properties{};
  Check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
  std::printf("device: %s (%s), %.0f GiB visible\n", properties.name,
              properties.gcnArchName, properties.totalGlobalMem / 1073741824.0);

  const std::size_t bytes = static_cast<std::size_t>(gib * 1073741824.0);
  const std::size_t count = bytes / sizeof(float4);
  float4* a = nullptr;
  float4* b = nullptr;
  float* sink = nullptr;
  Check(hipMalloc(&a, bytes), "hipMalloc a");
  Check(hipMalloc(&b, bytes), "hipMalloc b");
  Check(hipMalloc(&sink, sizeof(float)), "hipMalloc sink");
  Check(hipMemset(a, 0, bytes), "hipMemset a");
  Check(hipMemset(b, 0, bytes), "hipMemset b");

  const int blocks = properties.multiProcessorCount * 32;
  const int threads = 256;
  const double read_gb = static_cast<double>(bytes) / 1e9;
  const double copy_gb = read_gb * 2.0;  // one read plus one write

  hipEvent_t start, stop;
  Check(hipEventCreate(&start), "hipEventCreate");
  Check(hipEventCreate(&stop), "hipEventCreate");

  std::vector<Result> results;
  for (int run = 0; run < iterations; ++run) {
    Result result{};
    Check(hipEventRecord(start), "hipEventRecord");
    ReadKernel<<<blocks, threads>>>(a, count, sink);
    Check(hipEventRecord(stop), "hipEventRecord");
    Check(hipEventSynchronize(stop), "hipEventSynchronize");
    result.read_gbs = read_gb / (Millis(start, stop) / 1000.0);

    Check(hipEventRecord(start), "hipEventRecord");
    CopyKernel<<<blocks, threads>>>(b, a, count);
    Check(hipEventRecord(stop), "hipEventRecord");
    Check(hipEventSynchronize(stop), "hipEventSynchronize");
    result.copy_gbs = copy_gb / (Millis(start, stop) / 1000.0);

    Check(hipEventRecord(start), "hipEventRecord");
    WriteKernel<<<blocks, threads>>>(b, count);
    Check(hipEventRecord(stop), "hipEventRecord");
    Check(hipEventSynchronize(stop), "hipEventSynchronize");
    result.write_gbs = read_gb / (Millis(start, stop) / 1000.0);

    std::printf("run %d: read %.1f GB/s, copy %.1f GB/s, write %.1f GB/s\n", run + 1,
                result.read_gbs, result.copy_gbs, result.write_gbs);
    results.push_back(result);
  }

  auto median = [&results](double Result::*field) {
    std::vector<double> values;
    values.reserve(results.size());
    for (const auto& result : results) {
      values.push_back(result.*field);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  const double read = median(&Result::read_gbs);
  const double copy = median(&Result::copy_gbs);
  const double write = median(&Result::write_gbs);
  std::printf("median: read %.1f GB/s, copy %.1f GB/s, write %.1f GB/s (%.1f GiB, %d runs)\n",
              read, copy, write, gib, iterations);
  std::printf("ROOFLINE_READ_GBPS %.1f\n", read);

  Check(hipFree(a), "hipFree a");
  Check(hipFree(b), "hipFree b");
  Check(hipFree(sink), "hipFree sink");
  return 0;
}
