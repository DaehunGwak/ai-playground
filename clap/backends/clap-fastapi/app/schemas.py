from pydantic import BaseModel


class TextEmbedRequest(BaseModel):
    texts: list[str]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int
