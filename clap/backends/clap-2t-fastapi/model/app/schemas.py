from pydantic import BaseModel


class TextInferRequest(BaseModel):
    texts: list[str]


class InferResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int
