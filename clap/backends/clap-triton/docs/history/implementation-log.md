# clap-triton 구현 로그

> 작성일: 2026-03-08

---

## 1. 프로젝트 배경

### clap-fastapi → clap-triton 전환 목적

`clap-fastapi`는 LAION CLAP 모델(HTSAT-base + RoBERTa)을 FastAPI 내에서 직접 PyTorch로 로드하여 text/audio 임베딩을 서빙하는 구조이다.

이 구조를 **FastAPI(전처리) + Triton Inference Server(모델 추론)**으로 분리하는 새 프로젝트 `clap-triton`을 만들었다. 동일한 API 스펙을 유지하면서 모델 서빙을 Triton에 위임한다.

**전환 목적**:
- 전처리/추론 관심사 분리
- Triton의 dynamic batching / model management / metrics 활용
- 향후 ONNX/TensorRT 최적화 기반 마련

---

## 2. 핵심 설계 결정

### 2-1. Triton Backend 선택: Hybrid (ONNX + Python)

| 모델 | Backend | 이유 |
|------|---------|------|
| `clap_text` | **ONNX Runtime** | RoBERTa + projection은 ONNX export가 잘 지원됨. 텐서 입력이 단순 (input_ids, attention_mask) |
| `clap_audio` | **Python Backend** | HTSAT forward()가 dict 입력을 받고, torchlibrosa STFT의 ONNX 호환성 이슈, reshape_wav2img의 dynamic interpolation 등 ONNX export 리스크가 높음 |

### 2-2. 전처리/추론 분리 경계

| 위치 | Text | Audio |
|------|------|-------|
| **FastAPI** | RobertaTokenizer → `input_ids` (B,77) int64 + `attention_mask` (B,77) int64 | librosa resample 48kHz → int16 quantization → repeat-pad/crop 480000 samples → `waveform` (B,480000) float32 |
| **Triton** | RoBERTa → text_projection → L2 norm → (B,512) | HTSAT(STFT→mel→Swin) → audio_projection → L2 norm → (B,512) |

### 2-3. int16 quantization 처리 방식

`clap-fastapi`에서는 `laion_clap` 내부에서 자동으로 처리되던 int16 quantization을,
`clap-triton`의 FastAPI 전처리 단계에서 명시적으로 재현해야 한다.

**결정**: numpy로 직접 재구현

```python
def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32767.0
```

**Trade-off 논의**:
- `laion_clap` 직접 사용: 코드 재사용 가능하지만 FastAPI 이미지에 torch/laion_clap 의존성이 추가되어 이미지 경량화 목적에 반함
- numpy 재구현: torch 의존성 없이 동일한 양자화 동작을 재현 가능, FastAPI 이미지를 경량하게 유지

**선택**: numpy 재구현. 로직이 단순하고 명확하며, 이미지 경량화 이점이 크다.

---

## 3. Phase별 구현 요약

### Phase 1: 프로젝트 스캐폴딩

- `clap-triton/` 디렉토리 생성
- `pyproject.toml` 생성 — API gateway 의존성만 포함 (torch 제외)
  - 포함: `fastapi`, `uvicorn[standard]`, `tritonclient[grpc]`, `librosa`, `transformers`, `numpy`, `python-multipart`
  - 제외: `torch`, `laion-clap` (FastAPI 이미지 경량화)
- `app/schemas.py` — clap-fastapi와 동일하게 복사
- `.gitignore`, `.dockerignore`, `.python-version` 생성

### Phase 2: 모델 Export 스크립트

#### `scripts/export_text_model.py`

1. `laion_clap.CLAP_Module` 로드 (checkpoint 사용)
2. 래퍼 `CLAPTextONNX(nn.Module)` 생성:
   - `self.text_branch` = RobertaModel (CLAP의 `model.text_branch`)
   - `self.text_projection` = nn.Sequential(Linear(768,512), ReLU, Linear(512,512))
   - `forward(input_ids, attention_mask)` → pooler_output → projection → L2 norm → (B,512)
3. `torch.onnx.export()` with opset 17, dynamic batch axis
4. 출력: `triton/model_repository/clap_text/1/model.onnx`

참조:
- `laion_clap/clap_module/model.py` line 629-636 (`encode_text` roberta 분기)
- `laion_clap/clap_module/model.py` line 510-514 (`text_projection` 정의)

#### `scripts/export_audio_model.py`

1. CLAP 모델 로드
2. 오디오 관련 가중치 추출:
   - `audio_branch` (HTSAT-Swin-Transformer: spectrogram + logmel + BN + Swin)
   - `audio_projection` (Linear(1024,512) → ReLU → Linear(512,512))
3. state_dict + audio_cfg JSON으로 저장
4. 출력: `triton/model_repository/clap_audio/1/clap_audio_weights.pt`, `audio_cfg.json`

참조:
- `laion_clap/clap_module/htsat.py` line 866 (`forward` 메서드)
- `laion_clap/clap_module/model.py` line 720-742 (`get_audio_embedding`)

### Phase 3: Triton 모델 구성

#### `clap_text` (ONNX Runtime)

`config.pbtxt`:
- backend: `onnxruntime`
- input: `input_ids` INT64 [77], `attention_mask` INT64 [77]
- output: `embeddings` FP32 [512]
- `max_batch_size: 32`
- `instance_group: KIND_CPU, count 1`
- dynamic_batching: preferred [4, 8, 16], max_queue_delay 100ms

#### `clap_audio` (Python Backend)

`config.pbtxt`:
- backend: `python`
- input: `waveform` FP32 [480000], `longer` BOOL [1]
- output: `embeddings` FP32 [512]
- `max_batch_size: 8` (오디오가 메모리 집약적)
- `instance_group: KIND_CPU, count 1`
- dynamic_batching: preferred [1, 2, 4], max_queue_delay 200ms

`model.py` — `TritonPythonModel` 클래스:
- `initialize()`: HTSAT 모델 + audio_projection 재구성, 가중치 로드, eval 모드
- `execute()`: waveform/longer numpy → torch tensor → dict 입력 → HTSAT forward → projection → L2 norm → numpy 반환

### Phase 4: FastAPI Gateway

#### `app/triton_client.py`

`TritonCLAPClient` 클래스:
- `tritonclient.grpc.InferenceServerClient`로 Triton 연결
- `embed_text(input_ids, attention_mask)` → gRPC `clap_text` 모델 호출 → (B,512) numpy
- `embed_audio(waveform, longer)` → gRPC `clap_audio` 모델 호출 → (B,512) numpy
- `is_ready()` → 서버 상태 확인

#### `app/main.py`

- lifespan에서 `RobertaTokenizer` 로드 + `TritonCLAPClient` 초기화
- `TRITON_URL` 환경변수 (기본값: `localhost:8001`)
- `/health` — Triton 서버 상태까지 체크
- `/echo/{message}`, `/heavy` — clap-fastapi와 동일

#### `app/routers/embed.py`

**`POST /embed/text`**:
1. `request.texts` 검증
2. `RobertaTokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="np")` → numpy int64
3. `triton_client.embed_text(input_ids, attention_mask)` 호출
4. `EmbeddingResponse` 반환

**`POST /embed/audio`**:
1. 파일 크기 검증 (50MB)
2. `librosa.load(BytesIO(raw), sr=48000, mono=True)` — 리샘플링
3. 10초 초과 시 center-crop 또는 `start_sec` 기준 crop
4. int16 quantization: `int16_to_float32(float32_to_int16(waveform))`
5. 480000 미만 시 repeat-pad
6. `triton_client.embed_audio(waveform[None,:], longer=np.array([[False]]))` 호출
7. `EmbeddingResponse` 반환

> **주의**: int16 quantization과 repeat-pad 로직을 정확히 재현하지 않으면 clap-fastapi 대비 임베딩 결과가 달라짐.

### Phase 5: Docker 구성

#### `docker/Dockerfile.api` (경량)
- Base: `python:3.11-slim`
- 시스템 패키지: `libsndfile1`, `ffmpeg`
- uv로 의존성 설치 (torch 없음 → 이미지 경량화)
- 빌드 시 RobertaTokenizer 캐싱

#### `docker/Dockerfile.triton`
- Base: `nvcr.io/nvidia/tritonserver:24.01-py3-cpu` (CPU 전용)
- pip install: torch (CPU), laion-clap, transformers, torchlibrosa
- 빌드 시 checkpoint 다운로드 + export scripts 실행
- 최종 model_repository를 `/models/`에 배치

#### `docker/docker-compose.yaml`
- `triton` 서비스: gRPC 8001, HTTP 8000, metrics 8002
- `api` 서비스: port 8000, `TRITON_URL=triton:8001`, depends_on triton

### Phase 6: Kubernetes 구성

- `triton-deployment.yaml`: Triton 서버 Pod (CPU 전용, requests: CPU 1/메모리 4Gi, limits: CPU 4/메모리 8Gi)
- `api-deployment.yaml`: FastAPI Gateway Pod (CPU 500m, 메모리 1Gi)
- `triton-service.yaml`: ClusterIP, gRPC 8001
- `api-service.yaml`: ClusterIP, 80→8000
- `ingress.yaml`: `clap-triton.local` → api-service

### Phase 7: 검증

`scripts/verify_triton.py`:
1. 동일 텍스트 입력 → clap-fastapi vs clap-triton 임베딩 비교 (cosine similarity > 0.999)
2. 동일 오디오 파일 → 양쪽 임베딩 비교
3. L2 norm 검증 (≈ 1.0)
4. 응답 스키마 동일성 확인

---

## 4. 생성된 파일 목록

```
clap-triton/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── triton_client.py
│   └── routers/
│       ├── __init__.py
│       └── embed.py
├── triton/
│   └── model_repository/
│       ├── clap_text/
│       │   ├── config.pbtxt
│       │   └── 1/
│       │       └── model.onnx          # export script로 생성 (gitignore)
│       └── clap_audio/
│           ├── config.pbtxt
│           └── 1/
│               ├── model.py
│               ├── clap_audio_weights.pt  # export script로 생성 (gitignore)
│               └── audio_cfg.json         # export script로 생성 (gitignore)
├── scripts/
│   ├── export_text_model.py
│   ├── export_audio_model.py
│   └── verify_triton.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.triton
│   └── docker-compose.yaml
├── k8s/
│   ├── triton-deployment.yaml
│   ├── api-deployment.yaml
│   ├── triton-service.yaml
│   ├── api-service.yaml
│   └── ingress.yaml
├── docs/
│   └── history/
│       └── implementation-log.md       # 이 파일
├── checkpoints/                        # .gitignore
├── .gitignore
├── .dockerignore
├── .python-version
└── pyproject.toml
```

---

## 5. 구현 순서 및 의존관계

```
Phase 1 (스캐폴딩) ──────────────────────┐
Phase 2 (Export 스크립트) ───┐            │
Phase 3 (Triton 모델 구성) ──┤ 병렬 가능  │ 의존
Phase 4 (FastAPI Gateway) ───┘            │
Phase 5 (Docker) ─────────── Phase 2,3,4 완료 후
Phase 6 (Kubernetes) ──────── Phase 5 완료 후
Phase 7 (검증) ────────────── Phase 5 + clap-fastapi 구동 필요
```

---

## 6. 검증 방법

1. **단위 검증**: export script 실행 후 ONNX 모델 로드하여 PyTorch 결과와 비교
2. **통합 검증**: docker-compose로 양 서비스 기동 → `scripts/verify_triton.py`로 clap-fastapi 대비 결과 일치 확인
3. **부하 테스트**: 기존 k6 스크립트 적용하여 성능 비교
4. **수동 검증**: `curl`로 `/embed/text`, `/embed/audio` 호출, `/health` 응답 확인
