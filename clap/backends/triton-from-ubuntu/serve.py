import os
import numpy as np
import onnxruntime as ort
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_metas = session.get_inputs()
output_metas = session.get_outputs()

DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
    "tensor(string)": np.bytes_,
}


def _shape(dims):
    return tuple(-1 if d is None or d == 0 else d for d in dims)


inputs = [
    Tensor(name=m.name, dtype=DTYPE_MAP.get(m.type, np.float32), shape=_shape(m.shape))
    for m in input_metas
]
outputs = [
    Tensor(name=m.name, dtype=DTYPE_MAP.get(m.type, np.float32), shape=(-1,))
    for m in output_metas
]


def infer_fn(**kwargs):
    feed = {name: arr for name, arr in kwargs.items()}
    results = session.run(None, feed)
    return {m.name: r for m, r in zip(output_metas, results)}


with Triton() as triton:
    triton.bind(
        model_name="model",
        infer_func=infer_fn,
        inputs=inputs,
        outputs=outputs,
        config=ModelConfig(max_batch_size=8),
    )
    triton.serve()
