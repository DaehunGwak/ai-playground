import os
import numpy as np
import torch
import laion_clap

_model: laion_clap.CLAP_Module | None = None


def load_model() -> None:
    global _model
    checkpoint_path = os.environ.get("CLAP_CHECKPOINT_PATH")
    _model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    if checkpoint_path:
        _model.load_ckpt(checkpoint_path)
    else:
        _model.load_ckpt()  # 자동 다운로드 (HuggingFace Hub)
    _model.eval()


def get_model() -> laion_clap.CLAP_Module:
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    return _model


def embed_text(texts: list[str]) -> np.ndarray:
    model = get_model()
    with torch.no_grad():
        embeddings = model.get_text_embedding(texts, use_tensor=False)
    return embeddings  # shape: (N, 512)


def embed_audio(audio_array: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    model = get_model()
    # laion_clap expects list of audio arrays
    audio_data = audio_array[None, :]  # (1, samples)
    with torch.no_grad():
        embeddings = model.get_audio_embedding_from_data(
            x=audio_data, use_tensor=False
        )
    return embeddings  # shape: (1, 512)
