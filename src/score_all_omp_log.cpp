#pragma GCC optimize("O3", "unroll-loops")

#include "cartographer_parallel/assignment.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cartographer_parallel {

namespace {
const char* kLogPath = "/tmp/score_all_omp_runtime.csv";
void WriteLog(int call, int n, int p, double total_ms) {
  static bool header = false;
  std::ofstream f(kLogPath, std::ios::app);
  if (!f) return;
  if (!header) {
    f << "call,total_ms,candidates,scan_points,work_items\n";
    header = true;
  }
  f << call << "," << total_ms << "," << n << "," << p << ","
    << static_cast<long long>(n) * p << "\n";
}
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
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h)) return;

  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned char* const grid_ptr = grid.data();

  const auto t0 = std::chrono::high_resolution_clock::now();

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

  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();

  static int call_count = 0;
  ++call_count;
  std::fprintf(stderr,
               "[score_all_omp] call=%d total=%.3fms candidates=%d\n",
               call_count, ms, n);
  std::fflush(stderr);
  WriteLog(call_count, n, p, ms);
}

}  // namespace cartographer_parallel
