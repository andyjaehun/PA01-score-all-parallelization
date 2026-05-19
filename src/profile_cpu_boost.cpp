// CPU optimization profiler
// Measures 4 versions with same multi-call method (call 1 = warm-up excluded)
//
// Versions:
//   baseline  : sequential double loop, no optimization
//   boost1    : -O3 compiler optimization only
//   boost2    : -O3 + OpenMP static scheduling
//   boost3    : -O3 + OpenMP dynamic scheduling
//
// Run with different thread counts:
//   OMP_NUM_THREADS=1 ./profile_cpu_boost
//   OMP_NUM_THREADS=2 ./profile_cpu_boost
//   OMP_NUM_THREADS=4 ./profile_cpu_boost

#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations from score_all_CPU_boost*.cpp
namespace cartographer_parallel {
void score_all_boost1(const std::vector<unsigned char>&, int, int,
                      const std::vector<int>&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&,
                      std::vector<float>*);
void score_all_boost2(const std::vector<unsigned char>&, int, int,
                      const std::vector<int>&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&,
                      std::vector<float>*);
void score_all_boost3(const std::vector<unsigned char>&, int, int,
                      const std::vector<int>&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&,
                      std::vector<float>*);
}  // namespace cartographer_parallel

namespace {

constexpr int   kMapW       = 1000;
constexpr int   kMapH       = 1000;
constexpr int   kScanPoints = 10000;
constexpr int   kCandMin    = -20;
constexpr int   kCandMax    =  20;
constexpr int   kCandStep   = 1;
constexpr int   kNumCalls   = 10;
constexpr float kCorrectTol = 1e-5f;

std::vector<unsigned char> make_grid() {
  std::vector<unsigned char> g(kMapW * kMapH);
  for (int y = 0; y < kMapH; ++y)
    for (int x = 0; x < kMapW; ++x)
      g[y * kMapW + x] = static_cast<unsigned char>((x + y) % 128 + 64);
  return g;
}

void make_scan(std::vector<int>* px, std::vector<int>* py) {
  px->resize(kScanPoints);
  py->resize(kScanPoints);
  const double radius = 200.0;
  for (int i = 0; i < kScanPoints; ++i) {
    const double a = 2.0 * M_PI * i / kScanPoints;
    (*px)[i] = kMapW / 2 + static_cast<int>(radius * std::cos(a));
    (*py)[i] = kMapH / 2 + static_cast<int>(radius * std::sin(a));
  }
}

// Sequential baseline — no optimization, used as reference
void score_all_baseline(const std::vector<unsigned char>& grid, const int w,
                        const int h, const std::vector<int>& px,
                        const std::vector<int>& py, const std::vector<int>& cx,
                        const std::vector<int>& cy, std::vector<float>* score) {
  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0) return;
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

using ScoreFunc = void (*)(const std::vector<unsigned char>&, int, int,
                           const std::vector<int>&, const std::vector<int>&,
                           const std::vector<int>&, const std::vector<int>&,
                           std::vector<float>*);

// Runs kNumCalls, excludes call 1 (warm-up), returns steady-state mean ms.
// Also checks correctness against ref_score.
double run_benchmark(const char* name, ScoreFunc fn,
                     const std::vector<unsigned char>& grid,
                     const std::vector<int>& px, const std::vector<int>& py,
                     const std::vector<int>& cx, const std::vector<int>& cy,
                     const std::vector<float>& ref_score) {
  std::vector<float> score;
  std::vector<double> wall(kNumCalls);

  for (int i = 0; i < kNumCalls; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn(grid, kMapW, kMapH, px, py, cx, cy, &score);
    wall[i] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr, "  [%s] call %2d  %.3f ms%s\n",
                 name, i + 1, wall[i], i == 0 ? "  <-- warm-up" : "");
  }

  // correctness check vs baseline
  float max_diff = 0.0f;
  if (score.size() == ref_score.size()) {
    for (size_t i = 0; i < score.size(); ++i)
      max_diff = std::max(max_diff, std::abs(score[i] - ref_score[i]));
  }
  const bool ok = max_diff < kCorrectTol;
  std::fprintf(stderr, "  [%s] correctness: max_diff=%.6f  %s\n\n",
               name, max_diff, ok ? "PASS" : "FAIL");

  const std::vector<double> steady(wall.begin() + 1, wall.end());
  return std::accumulate(steady.begin(), steady.end(), 0.0) / steady.size();
}

}  // namespace

int main() {
#ifdef _OPENMP
  std::fprintf(stderr, "[profile_cpu_boost] OMP_NUM_THREADS=%d\n\n",
               omp_get_max_threads());
#else
  std::fprintf(stderr, "[profile_cpu_boost] OpenMP not enabled\n\n");
#endif

  // Build input
  std::vector<unsigned char> grid = make_grid();
  std::vector<int> px, py;
  make_scan(&px, &py);
  std::vector<int> cx, cy;
  cartographer_parallel::make_cand(
      kCandMin, kCandMax, kCandMin, kCandMax, kCandStep, &cx, &cy);

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  std::fprintf(stderr, "[profile_cpu_boost] candidates=%d  scan_points=%d"
               "  work_items=%lld  calls=%d\n\n",
               n, p, static_cast<long long>(n) * p, kNumCalls);

  // Run baseline first to get reference scores
  std::fprintf(stderr, "── baseline (sequential) ──────────────────────────\n");
  std::vector<float> ref_score;
  std::vector<double> base_wall(kNumCalls);
  for (int i = 0; i < kNumCalls; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    score_all_baseline(grid, kMapW, kMapH, px, py, cx, cy, &ref_score);
    base_wall[i] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr, "  [baseline] call %2d  %.3f ms%s\n",
                 i + 1, base_wall[i], i == 0 ? "  <-- warm-up" : "");
  }
  const std::vector<double> base_steady(base_wall.begin() + 1, base_wall.end());
  const double base_ms = std::accumulate(base_steady.begin(),
                                         base_steady.end(), 0.0)
                         / base_steady.size();
  std::fprintf(stderr, "  [baseline] steady mean: %.3f ms\n\n", base_ms);

  // Run boost versions
  std::fprintf(stderr, "── boost1 (-O3 only) ──────────────────────────────\n");
  const double b1_ms = run_benchmark("boost1", cartographer_parallel::score_all_boost1,
                                     grid, px, py, cx, cy, ref_score);

  std::fprintf(stderr, "── boost2 (-O3 + OpenMP static) ───────────────────\n");
  const double b2_ms = run_benchmark("boost2", cartographer_parallel::score_all_boost2,
                                     grid, px, py, cx, cy, ref_score);

  std::fprintf(stderr, "── boost3 (-O3 + OpenMP dynamic,32) ──────────────\n");
  const double b3_ms = run_benchmark("boost3", cartographer_parallel::score_all_boost3,
                                     grid, px, py, cx, cy, ref_score);

  // Summary
  std::fprintf(stderr,
               "\n[profile_cpu_boost] ─── summary ───────────────────────\n"
               "  baseline (sequential) : %7.3f ms\n"
               "  boost1  (-O3)         : %7.3f ms   speedup: %.2fx\n"
               "  boost2  (+OMP static) : %7.3f ms   speedup: %.2fx\n"
               "  boost3  (+OMP dynamic): %7.3f ms   speedup: %.2fx\n"
               "────────────────────────────────────────────────────────\n",
               base_ms,
               b1_ms, base_ms / b1_ms,
               b2_ms, base_ms / b2_ms,
               b3_ms, base_ms / b3_ms);

  std::printf("baseline_ms=%.3f  b1_ms=%.3f  b2_ms=%.3f  b3_ms=%.3f"
              "  speedup_b2=%.2fx\n",
              base_ms, b1_ms, b2_ms, b3_ms, base_ms / b2_ms);

  return 0;
}
