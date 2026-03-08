import threading

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app import model as clap_model
from app.schemas import InferResponse, TextInferRequest

router = APIRouter(prefix="/infer", tags=["infer"])

_semaphore = threading.Semaphore(2)


@router.post("/text", response_model=InferResponse)
def infer_text(request: TextInferRequest):
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    with _semaphore:
        embeddings: np.ndarray = clap_model.embed_text(request.texts)
    return InferResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )


@router.post("/audio", response_model=InferResponse)
async def infer_audio(request: Request):
    length_header = request.headers.get("X-Audio-Length")
    if length_header is None:
        raise HTTPException(status_code=422, detail="X-Audio-Length 헤더가 필요합니다")
    n_samples = int(length_header)
    body = await request.body()
    audio_array = np.frombuffer(body, dtype=np.float32)
    if len(audio_array) != n_samples:
        raise HTTPException(
            status_code=422, detail="오디오 데이터 크기가 헤더와 일치하지 않습니다"
        )
    with _semaphore:
        embeddings: np.ndarray = clap_model.embed_audio(audio_array, sample_rate=48000)
    return InferResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        count=embeddings.shape[0],
    )
