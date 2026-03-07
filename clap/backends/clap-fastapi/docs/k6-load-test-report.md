# k6 부하 테스트 결과 보고서

- **테스트 일시**: 2026-03-06
- **환경**: Rancher Desktop (Moby engine + k3s v1.32.5)
- **체크포인트**: `music_audioset_epoch_15_esc_90.14.pt`
- **접근 방식**: `kubectl port-forward service/clap-fastapi 8000:80`

---

## 배포 환경

| 항목 | 값 |
|------|-----|
| 클러스터 | k3s v1.32.5 (lima-rancher-desktop) |
| 이미지 | `clap-fastapi:latest` (Docker → nerdctl k8s.io 네임스페이스) |
| Replicas | 1 |
| CPU limit | 4000m |
| Memory limit | 8000Mi |
| Memory 사용량 (실측) | ~3500Mi |
| Pod 재시작 | 0회 (초기 3000Mi 한도로 OOMKilled 2회 → 8000Mi로 조정 후 안정) |

> **메모**: 초기 배포 시 memory limit 3000Mi로 OOMKilled 발생. CLAP 모델 자체가 ~2.35GB 체크포인트 + PyTorch 오버헤드로 인해 8000Mi로 상향 후 안정화.

---

## 1. 텍스트 임베딩 부하 테스트 (`load-test-text.js`)

### 설정

| 항목 | 값 |
|------|-----|
| 스크립트 | `k6/load-test-text.js` |
| 엔드포인트 | `POST /embed/text` |
| 데이터 | 100개 음악 텍스트 쿼리 (장르/악기/분위기/보컬 등 10개 카테고리) |
| 배치 크기 | 랜덤 2~5개 / iteration |
| Stages | ramp-up 10s → 10 VUs, sustained 60s, ramp-down 10s |
| Threshold | `p(95) < 3000ms`, `error rate < 1%` |

### 결과 요약

| 지표 | 값 | 목표 | 판정 |
|------|-----|------|------|
| 총 요청 수 | 293 | - | - |
| Throughput | 3.64 req/s | - | - |
| 에러율 | 0.00% | < 1% | ✅ |
| avg | 1.93s | - | - |
| median | 2.02s | - | - |
| p(90) | 2.89s | - | - |
| **p(95)** | **3.02s** | **< 3.00s** | ❌ |
| min | 159ms | - | - |
| max | 3.45s | - | - |

### checks

| check | 결과 |
|-------|------|
| status 200 | ✅ 293/293 (100%) |
| embeddings length match | ✅ 293/293 (100%) |
| dimension 512 | ✅ 293/293 (100%) |

### 네트워크

| 항목 | 값 |
|------|-----|
| 수신 데이터 | 11 MB (135 kB/s) |
| 송신 데이터 | 71 kB (887 B/s) |

### 분석

- p(95) = **3.02s** — 임계값(3.00s) 대비 **+20ms 초과**. 사실상 경계선 수준.
- 10 VUs 동시 처리 시 CPU 경합으로 P95 지연 발생. 단일 Pod CPU 4코어 공유 환경에서 예상된 결과.
- 에러 0건으로 안정성은 완벽.
- **권고**: 임계값을 `p(95) < 3500ms`로 완화하거나, replicas 증설 또는 배치 크기 제한 시 통과 가능.

---

## 2. 오디오 임베딩 부하 테스트 (`load-test-audio.js`)

### 설정

| 항목 | 값 |
|------|-----|
| 스크립트 | `k6/load-test-audio.js` |
| 엔드포인트 | `POST /embed/audio` |
| 데이터 | 장르별 `.00.wav` 10개 (blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock) |
| 업로드 방식 | `multipart/form-data` (init 단계에서 binary 로드) |
| Stages | ramp-up 10s → 3 VUs, sustained 60s, ramp-down 10s |
| Threshold | `p(95) < 15000ms`, `error rate < 1%` |

### 결과 요약

| 지표 | 값 | 목표 | 판정 |
|------|-----|------|------|
| 총 요청 수 | 151 | - | - |
| Throughput | 1.87 req/s | - | - |
| 에러율 | 0.00% | < 1% | ✅ |
| avg | 433ms | - | - |
| median | 435ms | - | - |
| p(90) | 570ms | - | - |
| **p(95)** | **596ms** | **< 15000ms** | ✅ |
| min | 237ms | - | - |
| max | 2.02s | - | - |

### checks

| check | 결과 |
|-------|------|
| status 200 | ✅ 151/151 (100%) |
| embeddings length 1 | ✅ 151/151 (100%) |
| dimension 512 | ✅ 151/151 (100%) |

### 네트워크

| 항목 | 값 |
|------|-----|
| 수신 데이터 | 1.6 MB (21 kB/s) |
| 송신 데이터 | 200 MB (2.5 MB/s) |

### 분석

- p(95) = **596ms** — 임계값(15s) 대비 **극히 여유있는 통과** (4% 수준).
- 예상과 달리 오디오가 텍스트보다 훨씬 빠름. GTZAN 30초 WAV가 고정 길이 청크로 전처리되어 추론 시간이 안정적.
- 송신 데이터 200MB — WAV 파일 업로드 비용이 지배적. 프로덕션에서는 오디오 압축(MP3) 또는 스트리밍 전처리 검토 필요.
- 3 VUs에서도 처리량 충분 (CPU 경합 없이 안정).

---

## 종합 비교

| 항목 | 텍스트 (`/embed/text`) | 오디오 (`/embed/audio`) |
|------|----------------------|------------------------|
| VUs | 10 | 3 |
| 총 요청 | 293 | 151 |
| Throughput | 3.64 req/s | 1.87 req/s |
| avg 지연 | 1.93s | 433ms |
| p(95) 지연 | 3.02s | 596ms |
| 에러율 | 0% | 0% |
| 임계값 통과 | ❌ p95 (+20ms 초과) | ✅ |
| 모든 checks | ✅ 100% | ✅ 100% |

---

## 트러블슈팅 기록

### 1. nerdctl 소켓 경로 오류
- **증상**: `nerdctl --namespace k8s.io load` 실행 시 `/run/k3s/containerd/containerd.sock` 접근 불가
- **원인**: Rancher Desktop Moby 엔진은 Docker containerd 소켓 사용
- **해결**: `nerdctl --address /var/run/docker/containerd/containerd.sock --namespace k8s.io load`

### 2. OOMKilled (memory limit 3000Mi)
- **증상**: Pod 기동 직후 OOMKilled, 2회 재시작
- **원인**: CLAP 체크포인트 2.35GB + PyTorch + librosa 메모리 = ~3.5GB 필요
- **해결**: `deployment.yaml` memory limit 3000Mi → 8000Mi, request 1500Mi → 3000Mi, readinessProbe initialDelay 30s → 60s

### 3. /etc/hosts 미등록
- **증상**: `curl http://clap-fastapi.local/health` 타임아웃
- **원인**: `/etc/hosts`에 `clap-fastapi.local` 미등록
- **해결**: `kubectl port-forward service/clap-fastapi 8000:80` 사용, `BASE_URL=http://localhost:8000`으로 k6 실행

---

## 개선 제안

1. **텍스트 p95 초과**: replicas 2로 증설하면 10 VUs에서 p95 < 2s 예상
2. **오디오 업로드 대역폭**: WAV → MP3 변환 후 업로드하면 송신 데이터 200MB → ~20MB로 감소
3. **/etc/hosts 자동화**: k8s ConfigMap + ExternalDNS 또는 `hostAliases` 설정으로 로컬 DNS 해소
4. **메모리 최적화**: `torch.no_grad()` + half precision(fp16)으로 메모리 ~40% 절감 가능 (현재 3500Mi → 예상 ~2100Mi) → **`torch.no_grad()` 적용 완료** (스트레스 테스트 OOMKilled 해결, 메모리 3,154Mi 안정화)
