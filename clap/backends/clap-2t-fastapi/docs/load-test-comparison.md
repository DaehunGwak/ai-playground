# clap-fastapi vs clap-triton vs clap-2t-fastapi 성능 비교 리포트

- **테스트 일시**: 2026-03-08
- **테스트 도구**: k6 (동일 스크립트, BASE_URL 환경변수로 대상 전환)

---

## 1. 아키텍처 비교

| 항목 | clap-fastapi | clap-triton | clap-2t-fastapi |
|------|-------------|-------------|-----------------|
| 포트 | 8100 | 8080 | 8105 |
| 구조 | FastAPI 단일 서버 (1-tier) | FastAPI 게이트웨이 → Triton Inference Server (2-tier) | FastAPI 게이트웨이 → FastAPI 모델 서비스 (2-tier) |
| 추론 방식 | `laion_clap.CLAP_Module` 직접 로드 | gRPC → Triton (dynamic batching) | HTTP → FastAPI model (직접 CLAP 호출) |
| 동시성 제약 | `asyncio.Semaphore(2)` (gateway) | `threading.Semaphore(2)` (gateway) | `threading.Semaphore(2)` (model 서비스) |
| 텍스트 모델 | CLAP_Module (in-process) | Triton `clap_text` (max_batch=32, delay 100ms) | CLAP_Module (model 서비스 in-process) |
| 오디오 모델 | CLAP_Module + librosa (in-process) | CLAP_Module + librosa → Triton `clap_audio` (max_batch=8, delay 200ms) | librosa (gateway) + CLAP_Module (model 서비스) |

---

## 2. 텍스트 임베딩 비교 (`POST /embed/text`)

### 시나리오 (공통)

| 단계 | 시간 | VU |
|------|------|----|
| Warm-up | 10s | 5 |
| 압박 시작 | 20s | 15 |
| 피크 스트레스 | 30s | **30** |
| 회복 구간 | 20s | 15 |
| Ramp-down | 10s | 0 |

### 결과 비교

| 지표 | clap-triton | clap-fastapi | clap-2t-fastapi |
|------|-------------|--------------|-----------------|
| 총 요청 수 | 733 | 1,102 | **1,108** |
| 처리량 | 8.14 req/s | 12.24 req/s | **12.31 req/s** |
| 평균 레이턴시 | 1.97s | 1.30s | **1.29s** |
| 중앙값 (p50) | 1.93s | 1.35s | **1.30s** |
| p(90) | 3.40s | 2.24s | **2.23s** |
| p(95) | 3.61s | 2.32s | **2.33s** |
| 최대 레이턴시 | 3.66s | 2.69s | **2.82s** |
| 최소 레이턴시 | 147ms | 85.53ms | **83.5ms** |
| 에러율 | 0.00% | 0.00% | **0.00%** |
| Checks 통과율 | 100% (2199/2199) | 100% (3306/3306) | **100% (3324/3324)** |
| 수신 데이터 | 8.0 MB (89 kB/s) | 12 MB (134 kB/s) | **12 MB (134 kB/s)** |
| 송신 데이터 | 127 kB (1.4 kB/s) | 191 kB (2.1 kB/s) | **192 kB (2.1 kB/s)** |

### 임계값 판정

| 서비스 | 임계값 | 실제 p(95) | 판정 | 여유 |
|--------|--------|-----------|------|------|
| clap-triton | p(95) < 5,000ms | 3,610ms | ✓ | 28% |
| clap-fastapi | p(95) < 5,000ms | 2,320ms | ✓ | 54% |
| clap-2t-fastapi | p(95) < 5,000ms | 2,330ms | ✓ | **53%** |

---

## 3. 오디오 임베딩 비교 (`POST /embed/audio`)

### 시나리오 (공통)

| 단계 | 시간 | VU |
|------|------|----|
| Warm-up | 10s | 3 |
| 안정 구간 | 20s | 3 |
| 증설 | 10s | 5 |
| 안정 구간 | 20s | 5 |
| 증설 | 10s | **10** |
| 안정 구간 | 20s | **10** |
| Ramp-down | 10s | 0 |

### 결과 비교

| 지표 | clap-triton | clap-fastapi | clap-2t-fastapi |
|------|-------------|--------------|-----------------|
| 총 요청 수 | 352 | 475 | **395** |
| 처리량 | 3.52 req/s | 4.74 req/s | **3.94 req/s** |
| 평균 레이턴시 | 1.53s | 1.13s | **1.37s** |
| 중앙값 (p50) | 1.37s | 994ms | **1.17s** |
| p(90) | 2.80s | 2.08s | **2.34s** |
| p(95) | 2.82s | 2.11s | **2.36s** |
| 최대 레이턴시 | 3.17s | 2.92s | **7.24s** |
| 최소 레이턴시 | 264ms | 239ms | **233ms** |
| 에러율 | 0.00% | 0.00% | **0.00%** |
| Checks 통과율 | 100% (1056/1056) | 100% (1425/1425) | **100% (1185/1185)** |
| 수신 데이터 | 3.8 MB (38 kB/s) | 5.2 MB (52 kB/s) | **4.3 MB (43 kB/s)** |
| 송신 데이터 | 466 MB (4.7 MB/s) | 629 MB (6.3 MB/s) | **523 MB (5.2 MB/s)** |

### 임계값 판정

| 서비스 | 임계값 | 실제 p(95) | 판정 | 여유 |
|--------|--------|-----------|------|------|
| clap-triton | p(95) < 20,000ms | 2,820ms | ✓ | 86% |
| clap-fastapi | p(95) < 20,000ms | 2,110ms | ✓ | 89% |
| clap-2t-fastapi | p(95) < 20,000ms | 2,360ms | ✓ | **88%** |

---

## 4. 종합 분석

### 전체 성능 요약

| 항목 | 텍스트 우위 | 오디오 우위 |
|------|------------|------------|
| 처리량 | clap-2t-fastapi (12.31 req/s) ≈ clap-fastapi (12.24) | clap-fastapi (4.74 req/s) |
| p(95) 레이턴시 | clap-2t-fastapi (2.33s) ≈ clap-fastapi (2.32s) | clap-fastapi (2.11s) |
| 안정성 (에러율) | 3개 서비스 동일 (0%) | 3개 서비스 동일 (0%) |
| Checks 통과율 | 3개 서비스 동일 (100%) | 3개 서비스 동일 (100%) |

### 주요 관찰

1. **텍스트: clap-2t-fastapi ≈ clap-fastapi > clap-triton**: 텍스트 처리량에서 clap-2t-fastapi(12.31 req/s)와 clap-fastapi(12.24 req/s)가 사실상 동등하며, clap-triton(8.14 req/s)을 51% 상회. 2-tier HTTP 포워딩 오버헤드가 단일 서버와 비교해 무시할 수준임을 확인.

2. **오디오: clap-fastapi > clap-2t-fastapi > clap-triton**: 오디오에서는 clap-fastapi(4.74 req/s)가 가장 높고, clap-2t-fastapi(3.94 req/s)가 중간, clap-triton(3.52 req/s)이 가장 낮음. clap-2t-fastapi에서 gateway→model 간 HTTP로 전처리된 데이터를 전달하는 오버헤드가 텍스트보다 오디오에서 더 두드러짐.

3. **clap-2t-fastapi 오디오 최대 레이턴시 7.24s**: 다른 서비스 대비 최대값이 높음(clap-fastapi 2.92s, clap-triton 3.17s). p(95)는 2.36s로 안정적이므로 일부 요청에서 간헐적 지연 발생. gateway와 model 서비스 간 HTTP 연결 경합 또는 WAV 데이터 직렬화 지연이 원인으로 추정.

4. **semaphore 위치 차이**: clap-fastapi와 clap-triton은 semaphore가 gateway에 위치하여 gateway 자체가 병목. clap-2t-fastapi는 semaphore가 model 서비스에 위치하여 gateway 동시성 제한 없음. 이 구조는 gateway 수평 확장 시 model 서비스의 semaphore만 조정하면 되는 운영 이점이 있음.

5. **Triton dynamic batching의 실질 효과 부재**: clap-triton의 dynamic batching(텍스트: max_batch=32, delay 100ms / 오디오: max_batch=8, delay 200ms)은 `semaphore=2` 제약으로 동시 요청이 최대 2개로 제한되어 배치 효과가 미미. 오히려 배치 지연(100~200ms)이 레이턴시를 높이는 역효과가 있었던 것으로 판단.

6. **동일 안정성**: 3개 서비스 모두 에러율 0%, Checks 100% 통과로 기능적 안정성 동등. 운영 관점에서 clap-fastapi가 단일 컨테이너로 가장 단순하고, clap-2t-fastapi는 Triton 없이 2-tier를 구현하여 관리 복잡성을 줄이면서도 관심사 분리(gateway/model) 이점을 유지.

### 아키텍처 선택 가이드

| 상황 | 권장 아키텍처 |
|------|--------------|
| 낮은 레이턴시 / 높은 처리량 우선 | **clap-fastapi** 또는 **clap-2t-fastapi** |
| 운영 단순성 / 빠른 배포 | **clap-fastapi** |
| gateway/model 관심사 분리 필요 | **clap-2t-fastapi** |
| gateway 수평 확장 계획 | **clap-2t-fastapi** (semaphore가 model에 위치) |
| 모델 관리 / A/B 테스트 / 버전 관리 필요 | **clap-triton** (Triton 모델 저장소 기능 활용) |
| 대규모 배치 추론 (batch_size > 2) | **clap-triton** (semaphore 제약 제거 시 dynamic batching 효과) |
| GPU 클러스터 스케일아웃 | **clap-triton** (Triton 멀티 인스턴스 지원) |

### 임계값 달성 여부 전체 요약

| 서비스 | 테스트 | 임계값 | 실제 값 | 판정 |
|--------|--------|--------|---------|------|
| clap-triton | 텍스트 p(95) | < 5,000ms | 3,610ms | ✓ (여유 28%) |
| clap-triton | 텍스트 에러율 | < 5% | 0.00% | ✓ |
| clap-triton | 오디오 p(95) | < 20,000ms | 2,820ms | ✓ (여유 86%) |
| clap-triton | 오디오 에러율 | < 5% | 0.00% | ✓ |
| clap-fastapi | 텍스트 p(95) | < 5,000ms | 2,320ms | ✓ (여유 54%) |
| clap-fastapi | 텍스트 에러율 | < 5% | 0.00% | ✓ |
| clap-fastapi | 오디오 p(95) | < 20,000ms | 2,110ms | ✓ (여유 89%) |
| clap-fastapi | 오디오 에러율 | < 5% | 0.00% | ✓ |
| clap-2t-fastapi | 텍스트 p(95) | < 5,000ms | 2,330ms | ✓ (여유 53%) |
| clap-2t-fastapi | 텍스트 에러율 | < 5% | 0.00% | ✓ |
| clap-2t-fastapi | 오디오 p(95) | < 20,000ms | 2,360ms | ✓ (여유 88%) |
| clap-2t-fastapi | 오디오 에러율 | < 5% | 0.00% | ✓ |
