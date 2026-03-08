from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import model as clap_model
from app.routers import infer


@asynccontextmanager
async def lifespan(app: FastAPI):
    clap_model.load_model()
    yield


app = FastAPI(title="clap-2t-model", lifespan=lifespan)
app.include_router(infer.router)


@app.get("/health")
async def health():
    try:
        clap_model.get_model()
        return {"status": "ok"}
    except RuntimeError:
        return JSONResponse(status_code=503, content={"status": "loading"})
