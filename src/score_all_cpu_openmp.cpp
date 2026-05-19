// score_all_cpu_openmp.cpp
// CPU 병렬화 구현 — -O3 컴파일러 최적화 + OpenMP 멀티코어 병렬화
//
// 최적화 내용:
//   - #pragma GCC optimize("O3"): auto-vectorization, loop invariant code motion 등
//   - #pragma omp parallel for schedule(static): candidate 단위 4코어 분산 처리
//   - cx[i], cy[i] 지역 변수화: inner loop 반복 접근 제거
//   - inv_norm 사전 계산: 반복 나눗셈 제거

#pragma GCC optimize("O3", "unroll-loops")

#include "cartographer_parallel/assignment.h"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cartographer_parallel {

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
      grid.size() < static_cast<size_t>(w * h)) return;

  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned char* const grid_ptr = grid.data();

  // candidate 단위로 4코어에 분산 (각 candidate 계산은 독립적)
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const int cxi = cx[i];
    const int cyi = cy[i];
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cxi;
      const int y = py[j] + cyi;
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += grid_ptr[y * w + x];
    }
    (*score)[i] = static_cast<float>(sum) * inv_norm;
  }
}

}  // namespace cartographer_parallel
