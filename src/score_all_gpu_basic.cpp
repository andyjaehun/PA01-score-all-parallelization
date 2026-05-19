// score_all_gpu_basic.cpp
// GPU 기본 CUDA 구현
//
// 최적화 내용:
//   1. Shared Memory Reduction
//      - 1 block = 1 candidate, 256 threads = scan point 분담
//      - shared memory로 partial sum 합산 후 candidate별 score 계산
//
//   2. GpuWorkspace (Device Memory 재사용)
//      - 매 호출마다 cudaMalloc/cudaFree 반복 시 약 109ms overhead 발생
//      - static singleton으로 최초 1회만 할당하고 이후 재사용
//
//   3. __ldg__() Read-only Cache
//      - px, py는 모든 block이 동일하게 반복 읽는 데이터
//      - read-only texture cache를 경유해 global memory 접근 비용 감소

#include "cartographer_parallel/assignment.h"
#include <algorithm>
#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace cartographer_parallel {
namespace {

void cpu_fallback(const std::vector<unsigned char>& grid, const int w,
                  const int h, const std::vector<int>& px,
                  const std::vector<int>& py, const std::vector<int>& cx,
                  const std::vector<int>& cy, std::vector<float>* score) {
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cx[i];
      const int y = py[j] + cy[i];
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += grid[y * w + x];
    }
    (*score)[i] = static_cast<float>(sum) / (255.0f * p);
  }
}

#ifdef __CUDACC__

// 1 block = 1 candidate, 256 threads = scan point 분담
// shared memory reduction으로 candidate별 최종 score 계산
__global__ void score_all_kernel(
    const unsigned char* __restrict__ grid, const int w, const int h,
    const int* __restrict__ px, const int* __restrict__ py, const int p,
    const int* __restrict__ cx, const int* __restrict__ cy, const int n,
    float* __restrict__ score) {
  extern __shared__ int shmem[];
  const int cand = blockIdx.x;
  const int tid  = threadIdx.x;
  if (cand >= n) return;

  const int cxi = __ldg(&cx[cand]);
  const int cyi = __ldg(&cy[cand]);
  int local_sum = 0;

  for (int j = tid; j < p; j += blockDim.x) {
    const int x = __ldg(&px[j]) + cxi;
    const int y = __ldg(&py[j]) + cyi;
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
    score[cand] = static_cast<float>(shmem[0]) / (255.0f * p);
}

// GpuWorkspace: device memory를 최초 1회 할당 후 재사용
struct GpuWorkspace {
  unsigned char* grid  = nullptr; size_t grid_cap  = 0;
  int*           px    = nullptr; size_t px_cap    = 0;
  int*           py    = nullptr; size_t py_cap    = 0;
  int*           cx    = nullptr; size_t cx_cap    = 0;
  int*           cy    = nullptr; size_t cy_cap    = 0;
  float*         score = nullptr; size_t score_cap = 0;
};

GpuWorkspace& get_workspace() {
  static GpuWorkspace g;
  return g;
}

template<typename T>
bool grow(T** p, size_t* cap, size_t need) {
  if (*cap >= need) return true;
  if (*p) { cudaFree(*p); *p = nullptr; }
  bool ok = cudaMalloc(reinterpret_cast<void**>(p), need) == cudaSuccess;
  if (ok) *cap = need;
  return ok;
}

#endif  // __CUDACC__

}  // namespace

void make_cand(const int min_x, const int max_x, const int min_y,
               const int max_y, const int step,
               std::vector<int>* const cx, std::vector<int>* const cy) {
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
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h) || n == 0) return;

#ifdef __CUDACC__
  {
    const size_t gb = (size_t)w * h * sizeof(unsigned char);
    const size_t pb = (size_t)p * sizeof(int);
    const size_t cb = (size_t)n * sizeof(int);
    const size_t sb = (size_t)n * sizeof(float);

    GpuWorkspace& g = get_workspace();
    bool ok = true;
    ok = ok && grow(&g.grid,  &g.grid_cap,  gb);
    ok = ok && grow(&g.px,    &g.px_cap,    pb);
    ok = ok && grow(&g.py,    &g.py_cap,    pb);
    ok = ok && grow(&g.cx,    &g.cx_cap,    cb);
    ok = ok && grow(&g.cy,    &g.cy_cap,    cb);
    ok = ok && grow(&g.score, &g.score_cap, sb);

    if (ok) {
      ok = ok && cudaMemcpy(g.grid, grid.data(), gb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.px,   px.data(),   pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.py,   py.data(),   pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cx,   cx.data(),   cb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cy,   cy.data(),   cb, cudaMemcpyHostToDevice)==cudaSuccess;

      if (ok) {
        constexpr int kTPB = 256;
        score_all_kernel<<<n, kTPB, kTPB * sizeof(int)>>>(
            g.grid, w, h, g.px, g.py, p, g.cx, g.cy, n, g.score);
        ok = cudaGetLastError() == cudaSuccess;
      }

      if (ok)
        cudaMemcpy(score->data(), g.score, sb, cudaMemcpyDeviceToHost);
      return;
    }
  }
#endif

  cpu_fallback(grid, w, h, px, py, cx, cy, score);
}

}  // namespace cartographer_parallel
