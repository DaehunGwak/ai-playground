# CLAP 오디오 전처리 동작 방식

## 핵심 수치

| 항목 | 값 |
|---|---|
| 모델 | HTSAT-base |
| Sample Rate | 48,000 Hz |
| Max Length | 480,000 samples = **10초** |

---

## `enable_fusion=False` (현재 설정)

### 10초 초과 오디오 → `rand_trunc`

- 랜덤 시작 위치를 골라 **10초(480,000 samples)만큼 crop**
- 매 호출마다 다른 구간이 선택될 수 있음 (비결정적)
- 10초를 초과하는 구간은 모두 버려짐

```
[=====원본 오디오 (예: 60초)=====]
         [←10초→]  ← 랜덤 위치
```

### 10초 미만 오디오 → `repeatpad`

- 오디오를 `int(480000 / len)` 번 반복한 뒤, 남은 부분을 zero-pad

```
[3초][3초][3초][ 1초 zero-pad ] → 10초
```

---

## `enable_fusion=True`

### 10초 초과 오디오 → `fusion`

전체 mel spectrogram을 기반으로 **4개의 representation**을 생성하여 모델에 입력:

| 채널 | 설명 |
|---|---|
| `mel_shrink` | 전체 오디오를 10초 크기로 리사이즈(축소) - 전체 맥락 보존 |
| `front` | 앞쪽 1/3 구간에서 랜덤 10초 crop |
| `middle` | 중간 1/3 구간에서 랜덤 10초 crop |
| `back` | 뒤쪽 1/3 구간에서 랜덤 10초 crop |

```
[=====원본 오디오 (예: 60초)=====]
[───────shrink(전체 축소)────────]
[front ] [  middle  ] [  back  ]
```

### 10초 미만 오디오

동일한 mel을 4번 스택: `stack([mel, mel, mel, mel])`

---

## 두 모드 비교

| | `enable_fusion=False` | `enable_fusion=True` |
|---|---|---|
| 긴 오디오 처리 | 랜덤 10초 1개 사용 | 전체 축소 + 3구간 crop |
| 정보 손실 | 큼 (10초 초과분 전부 버림) | 작음 (전체 맥락 보존) |
| 메모리 사용 | 적음 | 약 4배 증가 |
| 결정성 | 비결정적 (랜덤 crop) | 부분 비결정적 (shrink는 결정적) |
| 체크포인트 | 현재 체크포인트 사용 가능 | fusion용 체크포인트 필요할 수 있음 |

---

## 관련 소스 위치

| 파일 | 설명 |
|---|---|
| `.venv/lib/python3.11/site-packages/laion_clap/hook.py:180` | `get_audio_embedding_from_data` - max_len, truncating/filling 방식 결정 |
| `.venv/lib/python3.11/site-packages/laion_clap/training/data.py:416` | `get_audio_features` - 실제 truncation/padding 로직 |
| `app/model.py` | `enable_fusion=False`, `amodel="HTSAT-base"` 설정 |
| `app/routers/embed.py` | 오디오 로드 (`librosa`, sr=48000) 후 CLAP에 전달 |
