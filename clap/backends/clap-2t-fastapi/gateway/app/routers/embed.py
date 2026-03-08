from io import BytesIO

import librosa
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile

from app.model_client import ModelClient
from app.schemas import EmbeddingResponse, TextEmbedRequest

router = APIRouter(prefix="/embed", tags=["embed"])

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB

CLIP_SECONDS = 10
SAMPLE_RATE = 48000
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE  # 480,000


def _get_client(request: Request) -> ModelClient:
    return request.app.state.model_client


@router.post("/text", response_model=EmbeddingResponse)
async def embed_text(request: TextEmbedRequest, req: Request):
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    client = _get_client(req)
    try:
        result = await client.infer_text(request.texts)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="모델 서비스 타임아웃")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"모델 서비스 오류: {e.response.status_code}"
        )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="모델 서비스에 연결할 수 없습니다")
    return EmbeddingResponse(**result)


@router.post("/audio", response_model=EmbeddingResponse)
async def embed_audio(
    file: UploadFile, req: Request, start_sec: float | None = Query(default=None)
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
                    status_code=422, detail="start_sec + 10초가 오디오 길이를 초과합니다"
                )
        else:
            center = len(audio_array) // 2
            start_sample = center - CLIP_SAMPLES // 2
        audio_array = audio_array[start_sample : start_sample + CLIP_SAMPLES]
    client = _get_client(req)
    try:
        result = await client.infer_audio(audio_array)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="모델 서비스 타임아웃")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"모델 서비스 오류: {e.response.status_code}"
        )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="모델 서비스에 연결할 수 없습니다")
    return EmbeddingResponse(**result)
