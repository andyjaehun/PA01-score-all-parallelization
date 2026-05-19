// Multi-call profiler for score_all().
//
// Call 1  = warm-up (CUDA init + first alloc — always slow, excluded from stats).
// Calls 2..N = steady-state performance.
//
// Also runs a CPU reference once and:
//   - Reports CPU baseline time.
//   - Checks that GPU scores match CPU scores (max absolute diff < 1e-4).
//
// Input size matches the assignment benchmark:
//   candidates  = 41 * 41 = 1681   (make_cand -20..20 step 1)
//   scan points = 10000
//   work items  = 16,810,000
//
// Build (add to CMakeLists_GPU.txt — already added):
//   add_executable(profile_score_all_multi src/profile_score_all_multi.cpp)
//   target_link_libraries(profile_score_all_multi assignment_cpu_lib ${catkin_LIBRARIES})
//
// Run:
//   rm -f /tmp/score_all_claude_runtime.csv
//   time /root/catkin_ws/devel/lib/cartographer_parallel/profile_score_all_multi

#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

constexpr int    kMapW       = 1000;
constexpr int    kMapH       = 1000;
constexpr int    kScanPoints = 10000;
constexpr int    kCandMin    = -20;
constexpr int    kCandMax    =  20;
constexpr int    kCandStep   = 1;
constexpr int    kNumCalls   = 10;   // call 1 = warm-up, calls 2..N = measured
constexpr float  kCorrectTol = 1e-4f;

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
  const int cx = kMapW / 2;
  const int cy = kMapH / 2;
  const double radius = 200.0;
  for (int i = 0; i < kScanPoints; ++i) {
    const double a = 2.0 * M_PI * i / kScanPoints;
    (*px)[i] = cx + static_cast<int>(radius * std::cos(a));
    (*py)[i] = cy + static_cast<int>(radius * std::sin(a));
  }
}

// CPU reference — same double-loop logic as score_all() baseline.
// Used purely for correctness checking and baseline timing.
void score_all_ref(const std::vector<unsigned char>& grid, const int w,
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

}  // namespace

int main() {
  std::vector<unsigned char> grid = make_grid();
  std::vector<int> px, py;
  make_scan(&px, &py);

  std::vector<int> cx, cy;
  cartographer_parallel::make_cand(
      kCandMin, kCandMax, kCandMin, kCandMax, kCandStep, &cx, &cy);

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));

  std::fprintf(stderr,
               "[profile_multi] candidates=%d  scan_points=%d"
               "  work_items=%lld  calls=%d\n\n",
               n, p, static_cast<long long>(n) * p, kNumCalls);

  // ── CPU multi-call (same structure as GPU: call 1 = warm-up, 2..N = measured)
  std::vector<float> cpu_score;
  std::vector<double> cpu_wall_ms(kNumCalls);
  for (int i = 0; i < kNumCalls; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    score_all_ref(grid, kMapW, kMapH, px, py, cx, cy, &cpu_score);
    cpu_wall_ms[i] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr, "[profile_multi] CPU call %2d  wall=%.3f ms%s\n",
                 i + 1, cpu_wall_ms[i], i == 0 ? "  <-- warm-up" : "");
  }
  const std::vector<double> cpu_steady(cpu_wall_ms.begin() + 1, cpu_wall_ms.end());
  const double cpu_ms = std::accumulate(cpu_steady.begin(), cpu_steady.end(), 0.0)
                        / cpu_steady.size();
  std::fprintf(stderr, "[profile_multi] CPU steady mean : %.3f ms\n\n", cpu_ms);

  // ── GPU multi-call ──────────────────────────────────────────────────────────
  std::vector<float> gpu_score;
  std::vector<float> steady_gpu_score;  // saved from call 2 for correctness
  std::vector<double> wall_ms(kNumCalls);

  for (int i = 0; i < kNumCalls; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    cartographer_parallel::score_all(grid, kMapW, kMapH, px, py, cx, cy,
                                     &gpu_score);
    wall_ms[i] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (i == 1) steady_gpu_score = gpu_score;  // first steady-state result

    std::fprintf(stderr, "[profile_multi] call %2d  wall=%.3f ms%s\n",
                 i + 1, wall_ms[i], i == 0 ? "  <-- warm-up" : "");
    std::fflush(stderr);
  }

  // ── Correctness check ───────────────────────────────────────────────────────
  std::fprintf(stderr, "\n");
  float max_diff = 0.0f;
  int   diff_idx = -1;
  if (steady_gpu_score.size() == cpu_score.size()) {
    for (int i = 0; i < static_cast<int>(cpu_score.size()); ++i) {
      const float d = std::abs(cpu_score[i] - steady_gpu_score[i]);
      if (d > max_diff) { max_diff = d; diff_idx = i; }
    }
  } else {
    std::fprintf(stderr, "[profile_multi] ERROR: size mismatch cpu=%zu gpu=%zu\n",
                 cpu_score.size(), steady_gpu_score.size());
  }
  const bool correct = max_diff < kCorrectTol;
  std::fprintf(stderr,
               "[profile_multi] correctness : max_diff=%.6f at idx=%d  --> %s\n",
               max_diff, diff_idx, correct ? "PASS" : "FAIL");

  // ── Summary ─────────────────────────────────────────────────────────────────
  if (kNumCalls > 1) {
    const std::vector<double> steady(wall_ms.begin() + 1, wall_ms.end());
    const double mean = std::accumulate(steady.begin(), steady.end(), 0.0) /
                        steady.size();
    const double mn = *std::min_element(steady.begin(), steady.end());
    const double mx = *std::max_element(steady.begin(), steady.end());
    const double speedup = cpu_ms / mean;

    std::fprintf(stderr,
                 "\n[profile_multi] ─── summary ───────────────────────\n"
                 "  CPU ref          : %7.3f ms\n"
                 "  GPU warm-up      : %7.3f ms  (call 1, excluded)\n"
                 "  GPU steady mean  : %7.3f ms  (calls 2..%d)\n"
                 "  GPU steady min   : %7.3f ms\n"
                 "  GPU steady max   : %7.3f ms\n"
                 "  speedup (mean)   : %.2fx\n"
                 "  correctness      : %s\n"
                 "────────────────────────────────────────────────────\n",
                 cpu_ms, wall_ms[0], mean, kNumCalls, mn, mx,
                 speedup, correct ? "PASS" : "FAIL");

    std::printf("cpu_ms=%.3f  warmup_ms=%.3f  gpu_mean_ms=%.3f"
                "  gpu_min_ms=%.3f  speedup=%.2fx  correct=%s\n",
                cpu_ms, wall_ms[0], mean, mn, speedup,
                correct ? "yes" : "no");
  }

  return 0;
}
