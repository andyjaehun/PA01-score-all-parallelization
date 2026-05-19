#pragma GCC optimize("O3", "unroll-loops")
#pragma GCC target("native")

// Boost 2: -O3 + OpenMP parallel for with static scheduling.
//
// Why static: every candidate processes exactly p scan points → balanced load.
// static schedule divides n candidates evenly across threads with no overhead.
// Micro-opts applied:
//   - cx[i], cy[i] hoisted outside inner loop
//   - inv_norm precomputed (replaces repeated division)
//   - grid accessed via raw pointer (avoids vector bounds check in debug)

#include "cartographer_parallel/assignment.h"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cartographer_parallel {

void score_all_boost2(const std::vector<unsigned char>& grid, const int w,
                      const int h, const std::vector<int>& px,
                      const std::vector<int>& py, const std::vector<int>& cx,
                      const std::vector<int>& cy,
                      std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h))
    return;

  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned char* const grid_ptr = grid.data();

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
