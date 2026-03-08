import threading
from io import BytesIO

import librosa
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile

from app.schemas import EmbeddingResponse, TextEmbedRequest

router = APIRouter(prefix="/embed", tags=["embed"])

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
CLIP_SECONDS = 10
SAMPLE_RATE = 48000
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE  # 480,000

_inference_semaphore = threading.Semaphore(2)


def _int16_quantize(audio: np.ndarray) -> np.ndarray:
    """laion_clap 내부의 int16 quantization 재현.

    float32 → int16 → float32 변환으로 quantization noise 추가.
    get_audio_embedding_from_data()에서 use_tensor=False일 때 수행하는 동일 로직.
    """
    clipped = np.clip(audio, a_min=-1.0, a_max=1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    return (int16 / 32767.0).astype(np.float32)


def _repeat_pad(audio: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    """laion_clap의 get_audio_features repeatpad 로직 재현.

    max_len 미만이면 repeat 후 zero-pad. max_len 이상이면 그대로 반환.
    """
    if len(audio) >= max_len:
        return audio[:max_len]
    n_repeat = int(max_len / len(audio))
    audio = np.tile(audio, n_repeat)
    return np.pad(audio, (0, max_len - len(audio)), mode="constant")


@router.post("/text", response_model=EmbeddingResponse)
def embed_text(request: Request, body: TextEmbedRequest):
    if not body.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    tokenizer = request.app.state.tokenizer
    triton_client = request.app.state.triton_client

    encoded = tokenizer(
        body.texts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    with _inference_semaphore:
        embeddings = triton_client.embed_text(input_ids, attention_mask)

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )


@router.post("/audio", response_model=EmbeddingResponse)
def embed_audio(
    request: Request,
    file: UploadFile,
    start_sec: float | None = Query(default=None),
):
    raw = file.file.read()
    if len(raw) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다")
    if start_sec is not None and start_sec < 0:
        raise HTTPException(status_code=422, detail="start_sec은 0 이상이어야 합니다")

    audio_array, _ = librosa.load(BytesIO(raw), sr=SAMPLE_RATE, mono=True)
    del raw

    if len(audio_array) > CLIP_SAMPLES:
        if start_sec is not None:
            start_sample = int(start_sec * SAMPLE_RATE)
            if start_sample + CLIP_SAMPLES > len(audio_array):
                raise HTTPException(
                    status_code=422,
                    detail="start_sec + 10초가 오디오 길이를 초과합니다",
                )
        else:
            center = len(audio_array) // 2
            start_sample = center - CLIP_SAMPLES // 2
        audio_array = audio_array[start_sample : start_sample + CLIP_SAMPLES]

    # int16 quantization — laion_clap get_audio_embedding_from_data 내부 로직 재현
    audio_array = _int16_quantize(audio_array)

    # repeat-pad to 480000 samples — laion_clap get_audio_features repeatpad 재현
    audio_array = _repeat_pad(audio_array, max_len=CLIP_SAMPLES)

    waveform = audio_array[np.newaxis, :].astype(np.float32)  # (1, 480000)
    longer = np.array([[False]], dtype=bool)  # (1, 1)

    triton_client = request.app.state.triton_client

    with _inference_semaphore:
        embeddings = triton_client.embed_audio(waveform, longer)

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )
