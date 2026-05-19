// profile_synthetic.cpp
// 합성 데이터 기반 멀티콜 프로파일러
//
// CPU / GPU 구현을 동일한 조건으로 비교합니다:
//   - 합성 데이터: Grid 1000×1000, Candidates 1681개, Scan points 10000개
//   - 10회 반복 호출, Call 1 warm-up 제외, Call 2~10 평균 = steady-state
//   - CPU reference와 GPU 결과의 correctness 검증 포함
//
// 빌드 후 실행:
//   OMP_NUM_THREADS=4 ./profile_synthetic

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

namespace {

constexpr int   kMapW       = 1000;
constexpr int   kMapH       = 1000;
constexpr int   kScanPoints = 10000;
constexpr int   kCandMin    = -20;
constexpr int   kCandMax    =  20;
constexpr int   kCandStep   = 1;
constexpr int   kNumCalls   = 10;
constexpr float kCorrectTol = 1e-4f;

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

// CPU 기준값 계산 (correctness 검증용)
void score_all_ref(const std::vector<unsigned char>& grid, int w, int h,
                   const std::vector<int>& px, const std::vector<int>& py,
                   const std::vector<int>& cx, const std::vector<int>& cy,
                   std::vector<float>* score) {
  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
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

}  // namespace

int main() {
#ifdef _OPENMP
  std::fprintf(stderr, "[profile] OMP_NUM_THREADS=%d\n\n",
               omp_get_max_threads());
#endif

  // 합성 데이터 생성
  std::vector<unsigned char> grid = make_grid();
  std::vector<int> px, py;
  make_scan(&px, &py);
  std::vector<int> cx, cy;
  cartographer_parallel::make_cand(
      kCandMin, kCandMax, kCandMin, kCandMax, kCandStep, &cx, &cy);

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  std::fprintf(stderr,
               "[profile] candidates=%d  scan_points=%d"
               "  work_items=%lld  calls=%d\n\n",
               n, p, static_cast<long long>(n) * p, kNumCalls);

  // CPU reference (correctness 검증용)
  std::vector<float> ref_score;
  score_all_ref(grid, kMapW, kMapH, px, py, cx, cy, &ref_score);

  // 멀티콜 측정: Call 1 warm-up 제외, Call 2~10 평균
  std::vector<float> score;
  std::vector<double> wall_ms(kNumCalls);

  for (int i = 0; i < kNumCalls; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    cartographer_parallel::score_all(grid, kMapW, kMapH, px, py, cx, cy, &score);
    wall_ms[i] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr, "  call %2d  %.3f ms%s\n",
                 i + 1, wall_ms[i], i == 0 ? "  <-- warm-up" : "");
  }

  // Correctness 검증
  float max_diff = 0.0f;
  for (size_t i = 0; i < ref_score.size(); ++i)
    max_diff = std::max(max_diff, std::abs(ref_score[i] - score[i]));
  const bool correct = max_diff < kCorrectTol;

  // 결과 출력
  const std::vector<double> steady(wall_ms.begin() + 1, wall_ms.end());
  const double mean = std::accumulate(steady.begin(), steady.end(), 0.0) / steady.size();
  const double mn   = *std::min_element(steady.begin(), steady.end());
  const double mx   = *std::max_element(steady.begin(), steady.end());

  std::fprintf(stderr,
               "\n[profile] ─── summary ───────────────────────\n"
               "  warm-up       : %7.3f ms\n"
               "  steady mean   : %7.3f ms  (calls 2..%d)\n"
               "  steady min    : %7.3f ms\n"
               "  steady max    : %7.3f ms\n"
               "  correctness   : %s  (max_diff=%.6f)\n"
               "────────────────────────────────────────────\n",
               wall_ms[0], mean, kNumCalls, mn, mx,
               correct ? "PASS" : "FAIL", max_diff);

  return 0;
}
