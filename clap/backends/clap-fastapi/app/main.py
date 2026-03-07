from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model as clap_model
from app.routers import embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    clap_model.load_model()
    yield


app = FastAPI(title="clap-fastapi", lifespan=lifespan)
app.include_router(embed.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/echo/{message}")
async def echo(message: str):
    return {"message": message}


@app.get("/heavy")
async def heavy():
    """CPU-bound 작업 시뮬레이션 (성능 측정용)"""
    total = sum(i * i for i in range(10000))
    return {"result": total}
