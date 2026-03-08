import numpy as np
import httpx

from app.config import MODEL_SERVICE_URL, TIMEOUT


class ModelClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=MODEL_SERVICE_URL, timeout=TIMEOUT)

    async def infer_text(self, texts: list[str]) -> dict:
        resp = await self._client.post("/infer/text", json={"texts": texts})
        resp.raise_for_status()
        return resp.json()

    async def infer_audio(self, audio_array: np.ndarray) -> dict:
        body = audio_array.astype(np.float32).tobytes()
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Audio-Length": str(len(audio_array)),
        }
        resp = await self._client.post("/infer/audio", content=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def is_ready(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
