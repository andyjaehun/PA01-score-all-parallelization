#pragma GCC optimize("O3", "unroll-loops")
#pragma GCC target("native")

// Boost 1: -O3 compiler optimization only.
// Code is identical to baseline. Only compiler flags change.
// Effect: auto-vectorization, loop unrolling, instruction scheduling.

#include "cartographer_parallel/assignment.h"
#include <algorithm>

namespace cartographer_parallel {

void score_all_boost1(const std::vector<unsigned char>& grid, const int w,
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

}  // namespace cartographer_parallel
