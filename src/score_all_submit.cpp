#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace cartographer_parallel {
namespace {

const char* kLogPath = "/tmp/score_all_runtime.csv";

void WriteLog(const int call, const int n, const int p,
              const double total_ms, const double alloc_ms,
              const float h2d_ms, const float kernel_ms,
              const float d2h_ms, const bool used_cuda) {
  static bool header_written = false;
  std::ofstream f(kLogPath, std::ios::app);
  if (!f) return;
  if (!header_written) {
    f << "call,total_ms,alloc_ms,h2d_ms,kernel_ms,d2h_ms,"
         "candidates,scan_points,work_items,used_cuda\n";
    header_written = true;
  }
  f << call << "," << total_ms << "," << alloc_ms << ","
    << h2d_ms << "," << kernel_ms << "," << d2h_ms << ","
    << n << "," << p << "," << static_cast<long long>(n) * p << ","
    << (used_cuda ? 1 : 0) << "\n";
}

void score_all_cpu_fallback(const std::vector<unsigned char>& grid,
                             const int w, const int h,
                             const std::vector<int>& px,
                             const std::vector<int>& py,
                             const std::vector<int>& cx,
                             const std::vector<int>& cy,
                             std::vector<float>* const score) {
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h))
    return;
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cx[i];
      const int y = py[j] + cy[i];
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += grid[y * w + x];
    }
    (*score)[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
  }
}

#ifdef __CUDACC__

// 1 block = 1 candidate, 256 threads split scan-point work (shared mem reduction).
// __restrict__ : no pointer aliasing -> better instruction scheduling
// __ldg()      : read-only texture cache on Jetson Nano Maxwell SM5.3
__global__ void score_all_kernel(
    const unsigned char* __restrict__ grid, const int w, const int h,
    const int* __restrict__ px, const int* __restrict__ py, const int p,
    const int* __restrict__ cx, const int* __restrict__ cy, const int n,
    float* __restrict__ score) {
  extern __shared__ int shmem[];

  const int cand = blockIdx.x;
  const int tid  = threadIdx.x;
  if (cand >= n) return;

  const int cx_i = __ldg(&cx[cand]);
  const int cy_i = __ldg(&cy[cand]);
  int local_sum = 0;

  for (int j = tid; j < p; j += blockDim.x) {
    const int x = __ldg(&px[j]) + cx_i;
    const int y = __ldg(&py[j]) + cy_i;
    if (x >= 0 && x < w && y >= 0 && y < h)
      local_sum += __ldg(&grid[y * w + x]);
  }

  shmem[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shmem[tid] += shmem[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    score[cand] = static_cast<float>(shmem[0]) /
                  (255.0f * static_cast<float>(p));
}

// Static GPU workspace: allocated once on first call, reused on every subsequent call.
// Eliminates ~80 ms of cudaMalloc overhead per call on Jetson Nano.
struct GpuWorkspace {
  unsigned char* grid  = nullptr;  size_t grid_cap  = 0;
  int*           px    = nullptr;  size_t px_cap    = 0;
  int*           py    = nullptr;  size_t py_cap    = 0;
  int*           cx    = nullptr;  size_t cx_cap    = 0;
  int*           cy    = nullptr;  size_t cy_cap    = 0;
  float*         score = nullptr;  size_t score_cap = 0;

  cudaEvent_t h2d_start = nullptr, h2d_stop  = nullptr;
  cudaEvent_t ker_start = nullptr, ker_stop  = nullptr;
  cudaEvent_t d2h_start = nullptr, d2h_stop  = nullptr;
  bool events_ready = false;
};

GpuWorkspace& get_workspace() {
  static GpuWorkspace g;
  return g;
}

bool ensure_events(GpuWorkspace& g) {
  if (g.events_ready) return true;
  bool ok = true;
  ok = ok && cudaEventCreate(&g.h2d_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.h2d_stop)  == cudaSuccess;
  ok = ok && cudaEventCreate(&g.ker_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.ker_stop)  == cudaSuccess;
  ok = ok && cudaEventCreate(&g.d2h_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.d2h_stop)  == cudaSuccess;
  g.events_ready = ok;
  return ok;
}

template <typename T>
bool grow_if_needed(T** ptr, size_t* cap, const size_t need) {
  if (*cap >= need) return true;
  if (*ptr) { cudaFree(*ptr); *ptr = nullptr; }
  if (cudaMalloc(reinterpret_cast<void**>(ptr), need) != cudaSuccess)
    return false;
  *cap = need;
  return true;
}

bool ensure_capacity(GpuWorkspace& g, const size_t gbytes,
                     const size_t pbytes, const size_t cbytes,
                     const size_t sbytes) {
  bool ok = ensure_events(g);
  ok = ok && grow_if_needed(&g.grid,  &g.grid_cap,  gbytes);
  ok = ok && grow_if_needed(&g.px,    &g.px_cap,    pbytes);
  ok = ok && grow_if_needed(&g.py,    &g.py_cap,    pbytes);
  ok = ok && grow_if_needed(&g.cx,    &g.cx_cap,    cbytes);
  ok = ok && grow_if_needed(&g.cy,    &g.cy_cap,    cbytes);
  ok = ok && grow_if_needed(&g.score, &g.score_cap, sbytes);
  return ok;
}

bool score_all_cuda(const std::vector<unsigned char>& grid, const int w,
                    const int h, const std::vector<int>& px,
                    const std::vector<int>& py, const std::vector<int>& cx,
                    const std::vector<int>& cy,
                    std::vector<float>* const score,
                    double* total_ms, double* alloc_ms,
                    float* h2d_ms, float* kernel_ms, float* d2h_ms) {
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);

  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h) || n == 0) {
    if (total_ms) *total_ms = 0.0;
    if (alloc_ms) *alloc_ms = 0.0;
    return true;
  }

  const size_t gbytes = static_cast<size_t>(w) * h * sizeof(unsigned char);
  const size_t pbytes = static_cast<size_t>(p) * sizeof(int);
  const size_t cbytes = static_cast<size_t>(n) * sizeof(int);
  const size_t sbytes = static_cast<size_t>(n) * sizeof(float);

  GpuWorkspace& g = get_workspace();

  const auto alloc_t0 = std::chrono::high_resolution_clock::now();
  const bool cap_ok = ensure_capacity(g, gbytes, pbytes, cbytes, sbytes);
  const auto alloc_t1 = std::chrono::high_resolution_clock::now();
  if (alloc_ms)
    *alloc_ms = std::chrono::duration<double, std::milli>(
                    alloc_t1 - alloc_t0).count();
  if (!cap_ok) return false;

  const auto total_t0 = std::chrono::high_resolution_clock::now();

  bool ok = true;
  cudaEventRecord(g.h2d_start);
  ok = ok && cudaMemcpy(g.grid, grid.data(), gbytes,
                        cudaMemcpyHostToDevice) == cudaSuccess;
  ok = ok && cudaMemcpy(g.px, px.data(), pbytes,
                        cudaMemcpyHostToDevice) == cudaSuccess;
  ok = ok && cudaMemcpy(g.py, py.data(), pbytes,
                        cudaMemcpyHostToDevice) == cudaSuccess;
  ok = ok && cudaMemcpy(g.cx, cx.data(), cbytes,
                        cudaMemcpyHostToDevice) == cudaSuccess;
  ok = ok && cudaMemcpy(g.cy, cy.data(), cbytes,
                        cudaMemcpyHostToDevice) == cudaSuccess;
  cudaEventRecord(g.h2d_stop);
  cudaEventSynchronize(g.h2d_stop);

  if (ok) {
    constexpr int kTPB = 256;
    cudaEventRecord(g.ker_start);
    score_all_kernel<<<n, kTPB, kTPB * sizeof(int)>>>(
        g.grid, w, h, g.px, g.py, p, g.cx, g.cy, n, g.score);
    cudaEventRecord(g.ker_stop);
    cudaEventSynchronize(g.ker_stop);
    ok = cudaGetLastError() == cudaSuccess;
  }

  cudaEventRecord(g.d2h_start);
  ok = ok && cudaMemcpy(score->data(), g.score, sbytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaEventRecord(g.d2h_stop);
  cudaEventSynchronize(g.d2h_stop);

  if (h2d_ms)    cudaEventElapsedTime(h2d_ms,    g.h2d_start, g.h2d_stop);
  if (kernel_ms) cudaEventElapsedTime(kernel_ms, g.ker_start,  g.ker_stop);
  if (d2h_ms)    cudaEventElapsedTime(d2h_ms,    g.d2h_start, g.d2h_stop);

  const auto total_t1 = std::chrono::high_resolution_clock::now();
  if (total_ms)
    *total_ms = std::chrono::duration<double, std::milli>(
                    total_t1 - total_t0).count();
  return ok;
}

#endif  // __CUDACC__

}  // namespace

void make_cand(const int min_x, const int max_x, const int min_y,
               const int max_y, const int step, std::vector<int>* const cx,
               std::vector<int>* const cy) {
  if (cx == nullptr || cy == nullptr || step <= 0) return;
  for (int x = min_x; x <= max_x; x += step)
    for (int y = min_y; y <= max_y; y += step) {
      cx->push_back(x);
      cy->push_back(y);
    }
}

void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;

  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());

  double total_ms = 0.0, alloc_ms = 0.0;
  float  h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
  bool used_cuda = false;

#ifdef __CUDACC__
  used_cuda = score_all_cuda(grid, w, h, px, py, cx, cy, score,
                             &total_ms, &alloc_ms,
                             &h2d_ms, &kernel_ms, &d2h_ms);
  if (!used_cuda) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    score_all_cpu_fallback(grid, w, h, px, py, cx, cy, score);
    total_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0).count();
  }
#else
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    score_all_cpu_fallback(grid, w, h, px, py, cx, cy, score);
    total_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0).count();
  }
#endif

  static int call_count = 0;
  ++call_count;

  std::fprintf(stderr,
               "[score_all] call=%d total=%.3fms alloc=%.3fms "
               "h2d=%.3fms kernel=%.3fms d2h=%.3fms cuda=%d\n",
               call_count, total_ms, alloc_ms,
               h2d_ms, kernel_ms, d2h_ms, used_cuda ? 1 : 0);
  std::fflush(stderr);
  WriteLog(call_count, n, p, total_ms, alloc_ms,
           h2d_ms, kernel_ms, d2h_ms, used_cuda);
}

}  // namespace cartographer_parallel
