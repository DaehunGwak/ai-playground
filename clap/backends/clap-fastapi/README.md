# clap-fastapi

[LAION CLAP](https://github.com/LAION-AI/CLAP) 모델을 서빙하는 FastAPI 기반 임베딩 API 서버.
텍스트와 오디오를 각각 512차원 벡터로 임베딩한다.

## 모델 정보

| 항목 | 값 |
|------|----|
| 모델 | LAION CLAP (`larger_clap_music_and_speech`) |
| 아키텍처 | `HTSAT-base` + `enable_fusion=False` |
| 체크포인트 | `music_speech_audioset_epoch_15_esc_89.98.pt` (~2.35GB) |
| 임베딩 차원 | 512 (text, audio 동일) |

## 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 헬스체크 |
| `POST` | `/embed/text` | 텍스트 → 512차원 임베딩 |
| `POST` | `/embed/audio` | 오디오 파일 → 512차원 임베딩 |

### POST /embed/text

**Request**
```json
{
  "texts": ["a dog barking", "piano music"]
}
```

**Response**
```json
{
  "embeddings": [[0.012, -0.034, ...], [0.056, 0.021, ...]],
  "dimension": 512,
  "count": 2
}
```

### POST /embed/audio

- Content-Type: `multipart/form-data`
- 파라미터: `file` (오디오 파일)
- 지원 형식: WAV, MP3, FLAC 등 librosa 지원 형식
- 최대 크기: 50MB
- 내부적으로 48kHz mono로 리샘플링

```bash
curl -X POST http://localhost:8000/embed/audio \
  -F "file=@test.wav"
```

**Response**
```json
{
  "embeddings": [[0.023, -0.011, ...]],
  "dimension": 512,
  "count": 1
}
```

---

## 로컬 실행

### 사전 요구사항

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

### 1. 의존성 설치

```bash
uv sync
```

### 2. 체크포인트 다운로드

```bash
mkdir -p checkpoints
wget -O checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt \
  "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt"
```

> 체크포인트를 지정하지 않으면 서버 시작 시 HuggingFace Hub에서 자동 다운로드된다.

### 3. 서버 실행

```bash
# 체크포인트 경로 지정
CLAP_CHECKPOINT_PATH=./checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 경로 미지정 시 자동 다운로드
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 동작 확인

```bash
# 헬스체크
curl http://localhost:8000/health

# 텍스트 임베딩
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{"texts": ["a dog barking", "piano music"]}'

# 오디오 임베딩
curl -X POST http://localhost:8000/embed/audio \
  -F "file=@test.wav"
```

---

## Docker 실행

### 빌드

체크포인트가 빌드 타임에 포함된다 (~2.35GB).

```bash
docker build -t clap-fastapi:latest .
```

### 실행

```bash
docker run -p 8000:8000 clap-fastapi:latest
```

---

## Kubernetes 배포

[minikube](https://minikube.sigs.k8s.io/) 기준:

```bash
# 이미지를 minikube 내부에 로드
minikube image load clap-fastapi:latest

# 배포
kubectl apply -f k8s/deployment.yaml

# 서비스가 있다면
kubectl apply -f k8s/service.yaml

# 상태 확인
kubectl rollout status deployment/clap-fastapi
kubectl get pods
```

모델 로드에 시간이 걸리므로 readiness probe의 `initialDelaySeconds: 30`이 설정되어 있다.

---

## 부하 테스트

[k6](https://k6.io/) 필요.

```bash
# 일반 부하 테스트 (ramp-up → 50 VU 유지 → ramp-down)
k6 run k6/load-test.js

# 스트레스 테스트 (300 VU)
k6 run k6/stress-test.js

# 텍스트 전용 스트레스 테스트 (단건 요청, 60 VU 피크)
k6 run k6/load-test-text.js

# 오디오 VU 증설 테스트 (5→10→20 VU 단계적 증설)
k6 run k6/load-test-audio.js

# 대상 서버 오버라이드
BASE_URL=http://localhost:8000 k6 run k6/load-test.js
```

---

## 프로젝트 구조

```
clap-fastapi/
├── app/
│   ├── main.py          # FastAPI 앱, lifespan으로 모델 로드
│   ├── model.py         # CLAP 모델 싱글톤 로드/추론 래퍼
│   ├── schemas.py       # Pydantic 요청/응답 모델
│   └── routers/
│       └── embed.py     # /embed/text, /embed/audio 엔드포인트
├── k6/
│   ├── load-test.js          # 일반 부하 테스트 (ramp-up → 50 VU)
│   ├── stress-test.js        # 스트레스 테스트 (300 VU)
│   ├── load-test-text.js     # 텍스트 단건 스트레스 (60 VU 피크)
│   └── load-test-audio.js    # 오디오 VU 증설 (5→10→20 VU)
├── k8s/
│   └── deployment.yaml  # Kubernetes Deployment
├── docs/                # 테스트 결과 보고서
├── scripts/             # 유틸리티 스크립트
├── checkpoints/         # 모델 체크포인트 (gitignore)
├── Dockerfile
└── pyproject.toml
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CLAP_CHECKPOINT_PATH` | (없음) | 체크포인트 파일 경로. 미설정 시 HuggingFace Hub에서 자동 다운로드 |
