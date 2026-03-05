# fastapi-test

Kubernetes 환경에서 FastAPI의 성능 한계를 탐색하는 실험 프로젝트.
단일 Pod에서 CPU 리소스 제한에 따른 TPS 변화를 측정한다.

## 기술 스택

| 영역 | 기술 |
|------|------|
| Runtime | Python 3.13, FastAPI, Uvicorn (ASGI) |
| 패키지 관리 | uv |
| 컨테이너 | Docker (`python:3.13-slim`) |
| 오케스트레이션 | Kubernetes (Deployment + Service + Ingress/Traefik) |
| 부하 테스트 | k6 |

## 프로젝트 구조

```
fastapi-test/
├── app/
│   └── main.py          # FastAPI 애플리케이션
├── k6/
│   ├── load-test.js     # 일반 부하 테스트 (50 VU)
│   └── stress-test.js   # 스트레스 테스트 (300 VU)
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── Dockerfile
└── pyproject.toml
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 |
| GET | `/echo/{message}` | 메시지 에코 |
| GET | `/heavy` | CPU-bound 작업 시뮬레이션 (`sum(i*i for i in range(10000))`) |

## 빠른 시작 (Quick Start)

### 1. Docker 이미지 빌드

```bash
docker build -t fastapi-test:latest .
```

### 2. Kubernetes 배포

```bash
kubectl apply -f k8s/
```

### 3. Port-forward

```bash
kubectl port-forward svc/fastapi-test 8003:80
```

### 4. 테스트 실행

```bash
# 부하 테스트
k6 run -e BASE_URL=http://localhost:8003 k6/load-test.js

# 스트레스 테스트
k6 run k6/stress-test.js
```

## 테스트 시나리오

### Load Test (`k6/load-test.js`)

- VU: 최대 50, ramp-up 10s → sustained 30s → ramp-down 10s
- 엔드포인트: `/health`, `/echo/hello`, `/heavy` (순차 호출)
- sleep: 0.5s
- 임계값: p95 < 200ms, 에러율 < 1%

### Stress Test (`k6/stress-test.js`)

- VU: 최대 300, ramp-up 10s → sustained 50s → cooldown 15s
- 엔드포인트: `/heavy` 집중
- sleep 없음 (최대 부하)
- 임계값: p95 < 500ms, 에러율 < 10%

## 성능 테스트 결과

### Round 1 — CPU 500m

| 지표 | 값 |
|------|----|
| TPS | ~1,476 req/s |
| 병목 | CPU throttling |
| 비고 | CPU 제한이 주요 병목임을 확인 |

### Round 2 — CPU 2000m

| 지표 | 값 |
|------|----|
| TPS | **2,873 req/s** (+95% 향상) |
| p95 응답시간 | 110ms |
| 에러율 | 0% |

### 결론

CPU가 주요 병목이었으며, 2000m으로 증설 시 처리량이 약 2배 향상되었다.
단, port-forward 오버헤드와 단일 replica 구성으로 인해 추가 병목이 존재할 수 있다.

## 향후 과제

- [ ] 수평 확장 (replicas 증가) 후 TPS 재측정
- [ ] Port-forward 제거 후 직접 Ingress 접근으로 테스트
- [ ] `/heavy` 엔드포인트 프로파일링 (CPU 사용 패턴 분석)
