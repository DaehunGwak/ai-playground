import threading
from io import BytesIO

import librosa
import numpy as np
from fastapi import APIRouter, HTTPException, Query, UploadFile

from app import model
from app.schemas import EmbeddingResponse, TextEmbedRequest

router = APIRouter(prefix="/embed", tags=["embed"])

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
_inference_semaphore = threading.Semaphore(2)


@router.post("/text", response_model=EmbeddingResponse)
def embed_text(request: TextEmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    with _inference_semaphore:
        embeddings: np.ndarray = model.embed_text(request.texts)
    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )


CLIP_SECONDS = 10
SAMPLE_RATE = 48000
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE  # 480,000


@router.post("/audio", response_model=EmbeddingResponse)
def embed_audio(file: UploadFile, start_sec: float | None = Query(default=None)):
    raw = file.file.read()
    if len(raw) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다")
    if start_sec is not None and start_sec < 0:
        raise HTTPException(status_code=422, detail="start_sec은 0 이상이어야 합니다")
    audio_array, _ = librosa.load(BytesIO(raw), sr=SAMPLE_RATE, mono=True)
    del raw  # 조기 해제
    if len(audio_array) > CLIP_SAMPLES:
        if start_sec is not None:
            start_sample = int(start_sec * SAMPLE_RATE)
            if start_sample + CLIP_SAMPLES > len(audio_array):
                raise HTTPException(status_code=422, detail="start_sec + 10초가 오디오 길이를 초과합니다")
        else:
            center = len(audio_array) // 2
            start_sample = center - CLIP_SAMPLES // 2
        audio_array = audio_array[start_sample : start_sample + CLIP_SAMPLES]
    with _inference_semaphore:
        embeddings: np.ndarray = model.embed_audio(audio_array, sample_rate=SAMPLE_RATE)
    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )
