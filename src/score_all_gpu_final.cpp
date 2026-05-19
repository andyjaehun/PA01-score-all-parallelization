// score_all_gpu_final.cpp
// GPU 최종 최적화 구현
//
// 적용된 최적화 (기본 GPU에서 추가):
//
//   1. Grid Map GPU 상주
//      - grid map은 프레임 간 변하지 않는 고정 데이터
//      - 최초 1회만 GPU에 업로드하고 이후 재사용
//      - H2D 복사량: 1MB grid 제거 → px/py/cx/cy (수 KB)만 전송
//
//   2. Warp Shuffle Reduction (__shfl_down_sync__)
//      - 기존 shared memory reduction: __syncthreads() 8회 필요
//      - Warp 내 32 threads는 항상 동시 실행 → 동기화 없이 레지스터 통신
//      - __syncthreads() 8회 → 1회로 감소
//
//   3. Block Size 128 (256 → 128 threads)
//      - Jetson Nano SM 1개 최대 2048 threads 동시 실행
//      - 256 threads/block → 동시 8 blocks
//      - 128 threads/block → 동시 16 blocks (SM occupancy 2배)

#include "cartographer_parallel/assignment.h"
#include <algorithm>

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

// Warp Shuffle Reduction + Block 128
// - 128 threads = 4 warps → shared memory 4 int (기존 256 int 대비 1/64)
// - warp 내부는 __shfl_down_sync__으로 register 통신 (syncthreads 불필요)
// - warp 간 합산에만 __syncthreads__ 1회 사용
__global__ void score_all_kernel_final(
    const unsigned char* __restrict__ grid, const int w, const int h,
    const int* __restrict__ px, const int* __restrict__ py, const int p,
    const int* __restrict__ cx, const int* __restrict__ cy, const int n,
    float* __restrict__ score) {
  __shared__ int warp_sums[4];  // 128 threads / 32 = 4 warps

  const int cand = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & 31;   // warp 내 위치
  const int wid  = tid >> 5;   // warp 번호

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

  // Warp 내부 reduction (syncthreads 불필요)
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

  if (lane == 0) warp_sums[wid] = local_sum;
  __syncthreads();  // 딱 1번

  // Warp 0이 4개 warp_sums 최종 합산
  if (wid == 0) {
    local_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0;
    for (int offset = 2; offset > 0; offset >>= 1)
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  if (tid == 0)
    score[cand] = static_cast<float>(local_sum) / (255.0f * p);
}

// GpuWorkspace: device memory 재사용 + grid 상주
struct GpuWorkspace {
  // Grid: 최초 1회만 업로드
  unsigned char* d_grid     = nullptr; size_t d_grid_cap = 0;
  bool           grid_ready = false;

  // 나머지: 매 호출마다 업데이트
  int*   d_px    = nullptr; size_t d_px_cap    = 0;
  int*   d_py    = nullptr; size_t d_py_cap    = 0;
  int*   d_cx    = nullptr; size_t d_cx_cap    = 0;
  int*   d_cy    = nullptr; size_t d_cy_cap    = 0;
  float* d_score = nullptr; size_t d_score_cap = 0;
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

    // Grid 상주: 최초 1회만 H2D 복사
    if (!g.grid_ready) {
      ok = ok && grow(&g.d_grid, &g.d_grid_cap, gb);
      if (ok) {
        ok = ok && cudaMemcpy(g.d_grid, grid.data(), gb,
                              cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) g.grid_ready = true;
      }
    }

    ok = ok && grow(&g.d_px,    &g.d_px_cap,    pb);
    ok = ok && grow(&g.d_py,    &g.d_py_cap,    pb);
    ok = ok && grow(&g.d_cx,    &g.d_cx_cap,    cb);
    ok = ok && grow(&g.d_cy,    &g.d_cy_cap,    cb);
    ok = ok && grow(&g.d_score, &g.d_score_cap, sb);

    if (ok) {
      // H2D: px/py/cx/cy만 복사 (grid 제외)
      ok = ok && cudaMemcpy(g.d_px, px.data(), pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.d_py, py.data(), pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.d_cx, cx.data(), cb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.d_cy, cy.data(), cb, cudaMemcpyHostToDevice)==cudaSuccess;

      if (ok) {
        constexpr int kTPB = 128;  // Block Size 128
        const size_t shmem = (kTPB / 32) * sizeof(int);  // 4 warps × 4 bytes
        score_all_kernel_final<<<n, kTPB, shmem>>>(
            g.d_grid, w, h, g.d_px, g.d_py, p,
            g.d_cx, g.d_cy, n, g.d_score);
        ok = cudaGetLastError() == cudaSuccess;
      }

      if (ok)
        cudaMemcpy(score->data(), g.d_score, sb, cudaMemcpyDeviceToHost);
      return;
    }
  }
#endif

  cpu_fallback(grid, w, h, px, py, cx, cy, score);
}

}  // namespace cartographer_parallel
