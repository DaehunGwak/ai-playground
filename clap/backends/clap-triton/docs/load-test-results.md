# clap-triton k6 부하 테스트 결과

- **테스트 일시**: 2026-03-08
- **대상 서비스**: `http://localhost:8080` (FastAPI 게이트웨이 → Triton Inference Server)
- **환경**: Docker Compose (clap-triton-api:latest + clap-triton-server:latest)
- **제약 조건**: `_inference_semaphore = threading.Semaphore(2)` (최대 2개 동시 추론)

---

## 1. 텍스트 임베딩 스트레스 테스트 (`load-test-text.js`)

### 시나리오

| 단계 | 시간 | VU |
|------|------|----|
| Warm-up | 10s | 5 |
| 압박 시작 | 20s | 15 |
| 피크 스트레스 | 30s | **30** |
| 회복 구간 | 20s | 15 |
| Ramp-down | 10s | 0 |

- 엔드포인트: `POST /embed/text`
- 요청 형태: `{"texts": ["<랜덤 텍스트>"]}`  (100개 쿼리 풀에서 단건 랜덤 선택)

### 결과 요약

| 지표 | 값 |
|------|----|
| 총 요청 수 | 733 |
| 처리량 | **8.14 req/s** |
| 평균 레이턴시 | 1.97s |
| 중앙값 (p50) | 1.93s |
| p(90) | 3.40s |
| p(95) | **3.61s** ✓ (임계값 < 5s) |
| 최대 레이턴시 | 3.66s |
| 최소 레이턴시 | 147ms |
| 에러율 | **0.00%** ✓ (임계값 < 5%) |
| Checks 통과율 | **100%** (2199/2199) |
| 수신 데이터 | 8.0 MB (89 kB/s) |
| 송신 데이터 | 127 kB (1.4 kB/s) |

### Checks 상세

| Check | 결과 |
|-------|------|
| status 200 | ✓ 전체 통과 |
| embeddings length 1 | ✓ 전체 통과 |
| dimension 512 | ✓ 전체 통과 |

### 임계값 판정

```
http_req_duration  ✓ p(95)=3.61s < 5000ms
http_req_failed    ✓ rate=0.00% < 5%
```

---

## 2. 오디오 임베딩 부하 테스트 (`load-test-audio.js`)

### 시나리오

| 단계 | 시간 | VU |
|------|------|----|
| Warm-up | 10s | 3 |
| 안정 구간 | 20s | 3 |
| 증설 | 10s | 5 |
| 안정 구간 | 20s | 5 |
| 증설 | 10s | **10** |
| 안정 구간 | 20s | **10** |
| Ramp-down | 10s | 0 |

- 엔드포인트: `POST /embed/audio`
- 요청 형태: multipart/form-data WAV 파일 업로드 (10개 장르 샘플에서 랜덤 선택)
- 오디오 소스: `clap-fastapi/resources/audio/*.wav` (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

### 결과 요약

| 지표 | 값 |
|------|----|
| 총 요청 수 | 352 |
| 처리량 | **3.52 req/s** |
| 평균 레이턴시 | 1.53s |
| 중앙값 (p50) | 1.37s |
| p(90) | 2.80s |
| p(95) | **2.82s** ✓ (임계값 < 20s) |
| 최대 레이턴시 | 3.17s |
| 최소 레이턴시 | 264ms |
| 에러율 | **0.00%** ✓ (임계값 < 5%) |
| Checks 통과율 | **100%** (1056/1056) |
| 수신 데이터 | 3.8 MB (38 kB/s) |
| 송신 데이터 | **466 MB** (4.7 MB/s) |

### Checks 상세

| Check | 결과 |
|-------|------|
| status 200 | ✓ 전체 통과 |
| embeddings length 1 | ✓ 전체 통과 |
| dimension 512 | ✓ 전체 통과 |

### 임계값 판정

```
http_req_duration  ✓ p(95)=2.82s < 20000ms
http_req_failed    ✓ rate=0.00% < 5%
```

---

## 3. 종합 분석

### 성능 비교

| 항목 | 텍스트 | 오디오 |
|------|--------|--------|
| 피크 VU | 30 | 10 |
| 처리량 | 8.14 req/s | 3.52 req/s |
| p(95) | 3.61s | 2.82s |
| 에러율 | 0% | 0% |
| 네트워크 (송신) | 127 kB | 466 MB |

### 주요 관찰

1. **안정성**: 두 엔드포인트 모두 에러율 0%, Checks 100% 통과로 완전한 안정성 확인.

2. **텍스트 레이턴시 특성**: 피크 30 VU에서 p(95) 3.61s. `semaphore=2` 제약으로 최대 2개 추론이 병렬 처리되고, 나머지는 큐에서 대기. Triton `clap_text` 모델 (max_batch_size=32, preferred [4,8,16], delay 100ms)의 dynamic batching이 큐 대기 중인 요청들을 묶어 처리하여 처리량 향상에 기여.

3. **오디오 레이턴시 특성**: 피크 10 VU임에도 p(95) 2.82s로 텍스트보다 낮음. librosa 전처리가 포함되었지만 `clap_audio` 모델 (max_batch_size=8, preferred [1,2,4], delay 200ms)과 Python semaphore 조합이 VU 수에 비례하여 효율적으로 처리됨.

4. **네트워크 비대칭**: 오디오는 송신 466 MB (WAV 파일 업로드) vs 수신 3.8 MB (임베딩 응답). 텍스트는 역방향으로 송신 127 kB (텍스트) vs 수신 8.0 MB (임베딩 응답). 오디오 테스트 시 업로드 대역폭이 병목이 될 수 있음.

5. **Triton 2-tier 효과**: FastAPI gateway → Triton 구조에서도 단일 서버 구성(clap-fastapi)과 유사한 레이턴시 범위 달성. Triton의 dynamic batching이 동시 요청을 효율적으로 묶어 GPU/CPU 활용도를 높임.

### 임계값 달성 여부

| 테스트 | 임계값 | 실제 값 | 판정 |
|--------|--------|---------|------|
| 텍스트 p(95) | < 5,000ms | 3,610ms | ✓ 통과 (margin 28%) |
| 텍스트 에러율 | < 5% | 0.00% | ✓ 통과 |
| 오디오 p(95) | < 20,000ms | 2,820ms | ✓ 통과 (margin 86%) |
| 오디오 에러율 | < 5% | 0.00% | ✓ 통과 |
