import os

MODEL_SERVICE_URL: str = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8001")
TIMEOUT: float = float(os.environ.get("MODEL_TIMEOUT", "60"))
