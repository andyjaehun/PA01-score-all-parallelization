# PA_01 — score_all() 병렬화

**OpenMP, CUDA를 이용한 Cartographer score_all() 병렬화 및 성능 비교**

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| 장비 | NVIDIA Jetson Nano Developer Kit |
| CPU | ARM Cortex-A57, 4코어 |
| GPU | Maxwell SM 5.3, 128 CUDA cores |
| OS | Ubuntu 18.04 (Docker: dustynv/ros:melodic-ros-base-l4t-r32.7.1) |
| CUDA | 10.2 (compute capability 5.3) |

---

## 파일 구조

```
PA01_submission/
├── include/
│   └── cartographer_parallel/
│       └── assignment.h          # score_all() 함수 인터페이스
├── src/
│   ├── score_all_baseline.cpp    # 원본 순차 CPU 구현
│   ├── score_all_cpu_openmp.cpp  # CPU -O3 + OpenMP 병렬화
│   ├── score_all_gpu_basic.cpp   # GPU 기본 (GpuWorkspace + __ldg__)
│   └── score_all_gpu_final.cpp   # GPU 최종 (Grid상주 + Warp + Block128)
├── benchmark/
│   └── profile_synthetic.cpp     # 합성 데이터 벤치마크
└── cmake/
    ├── CMakeLists_cpu.txt
    └── CMakeLists_gpu.txt
```

---

## 구현 설명

### score_all_baseline.cpp
원본 순차 이중 루프 구현. 성능 비교의 기준값입니다.

### score_all_cpu_openmp.cpp
- `-O3` 컴파일러 최적화 (auto-vectorization, loop invariant code motion 등)
- `#pragma omp parallel for schedule(static)` 로 candidate 단위 4코어 병렬화
- candidate 계산이 독립적이므로 data race 없음

### score_all_gpu_basic.cpp
- **Shared Memory Reduction**: 1 block = 1 candidate, 256 threads = scan point 분담
- **GpuWorkspace**: device memory 최초 1회 할당 후 재사용 (cudaMalloc overhead 제거)
- **`__ldg__()`**: read-only texture cache 활용으로 global memory 접근 비용 감소

### score_all_gpu_final.cpp
기본 GPU에 2가지 최적화를 추가한 최종 버전입니다.
- **Warp Shuffle Reduction**: `__shfl_down_sync__()` 사용, syncthreads 8회 → 1회, kernel time -8%
- **Block Size 128**: 256 → 128 threads/block, SM 동시 block 수 2배 증가

---

## 빌드 방법

### CPU 버전
```bash
cp cmake/CMakeLists_cpu.txt CMakeLists.txt
cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

### GPU 버전
```bash
cp cmake/CMakeLists_gpu.txt CMakeLists.txt
cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

---

## 실행 — 합성 데이터 벤치마크

```bash
# CPU 버전
OMP_NUM_THREADS=4 /root/catkin_ws/devel/lib/cartographer_parallel/profile_synthetic

# GPU 버전 (동일 바이너리)
/root/catkin_ws/devel/lib/cartographer_parallel/profile_synthetic
```

합성 데이터: Grid 1000x1000 / Candidates 1681개 / Scan points 10000개
= **16,810,000 연산** / 10회 반복 / Call 1 warm-up 제외 / Call 2~10 평균

---

## 주요 결과

### 합성 데이터 (16.8M 연산)

| 버전 | Steady Mean | Speedup |
|------|------------|---------|
| CPU baseline (-O0) | 699.990 ms | 1.00x |
| CPU -O3 | 73.496 ms | 9.52x |
| CPU OpenMP | 35.946 ms | 19.47x |
| **GPU 최종** | **15.750 ms** | **44.44x** |

### Scaled Rosbag (candidates=1024)

| 버전 | Steady Mean |
|------|------------|
| CPU -O0 | 43.38 ms |
| CPU OpenMP dynamic | 1.34 ms |
| **GPU 최종** | **1.25 ms** |

> 모든 버전 Correctness: **PASS** (max_diff = 0.000000)

---

## 핵심 발견

GPU는 데이터 규모가 충분히 클 때 효과적입니다.
실제 rosbag(candidates=256)에서는 H2D/D2H overhead로 CPU가 유리하였지만,
candidates를 1024개로 확장하자 GPU가 CPU를 역전하였고
합성 데이터(16.8M 연산)에서는 **44.44x** 성능을 달성하였습니다.
