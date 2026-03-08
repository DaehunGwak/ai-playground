import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from transformers import RobertaTokenizer

from app.routers import embed
from app.triton_client import TritonCLAPClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    triton_url = os.environ.get("TRITON_URL", "localhost:8001")
    app.state.triton_client = TritonCLAPClient(url=triton_url)

    yield


app = FastAPI(title="clap-triton", lifespan=lifespan)
app.include_router(embed.router)


@app.get("/health")
async def health():
    client = app.state.triton_client if hasattr(app.state, "triton_client") else None
    if client is None or not client.is_ready():
        raise HTTPException(status_code=503, detail="Triton server not ready")
    return {"status": "ok"}


@app.get("/echo/{message}")
async def echo(message: str):
    return {"message": message}


@app.get("/heavy")
async def heavy():
    """CPU-bound 작업 시뮬레이션 (성능 측정용)"""
    total = sum(i * i for i in range(10000))
    return {"result": total}
