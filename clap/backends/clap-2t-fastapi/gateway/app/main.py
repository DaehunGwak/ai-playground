from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.model_client import ModelClient
from app.routers import embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_client = ModelClient()
    yield
    await app.state.model_client.aclose()


app = FastAPI(title="clap-2t-gateway", lifespan=lifespan)
app.include_router(embed.router)


@app.get("/health")
async def health():
    client: ModelClient = app.state.model_client
    if await client.is_ready():
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "model_unavailable"})
