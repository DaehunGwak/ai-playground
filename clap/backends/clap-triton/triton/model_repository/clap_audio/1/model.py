"""Triton Python Backend: CLAP audio embedding model.

This model receives pre-processed waveforms and returns L2-normalized
audio embeddings. It reconstructs HTSAT + audio_projection from weights
saved by scripts/export_audio_model.py.

Inputs:
    waveform (FP32, [480000]): Pre-processed audio waveform (repeat-padded,
        int16-quantized, 48kHz mono).
    longer (BOOL, [1]): Whether audio is longer than 10s (always False in
        current pipeline since FastAPI pre-crops to 10s).

Outputs:
    embeddings (FP32, [512]): L2-normalized audio embedding.
"""

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import torch
        import torch.nn.functional as F

        import laion_clap

        self._torch = torch
        self._F = F

        model_dir = os.path.join(
            args["model_repository"], args["model_version"]
        )
        weights_path = os.path.join(model_dir, "clap_audio_weights.pt")

        # CLAP_Module을 shell로 초기화 (체크포인트 없이) 후 audio 가중치 로드
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        saved = torch.load(weights_path, map_location="cpu")
        clap.model.audio_branch.load_state_dict(saved["audio_branch"])
        clap.model.audio_projection.load_state_dict(saved["audio_projection"])
        clap.model.eval()

        self.audio_branch = clap.model.audio_branch
        self.audio_projection = clap.model.audio_projection

    def execute(self, requests):
        torch = self._torch
        F = self._F
        responses = []

        for request in requests:
            waveform_np = pb_utils.get_input_tensor_by_name(
                request, "waveform"
            ).as_numpy()  # (B, 480000)
            longer_np = pb_utils.get_input_tensor_by_name(
                request, "longer"
            ).as_numpy()  # (B, 1)

            waveform_t = torch.from_numpy(waveform_np.copy())  # (B, 480000)
            longer_t = torch.from_numpy(longer_np[:, 0].copy())  # (B,)

            with torch.no_grad():
                audio_dict = {"waveform": waveform_t, "longer": longer_t}
                features = self.audio_branch(
                    audio_dict, mixup_lambda=None, device=None
                )["embedding"]  # (B, 1024)
                embeddings = self.audio_projection(features)  # (B, 512)
                embeddings = F.normalize(embeddings, dim=-1)  # (B, 512)

            embeddings_np = embeddings.numpy().astype(np.float32)
            out = pb_utils.Tensor("embeddings", embeddings_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    def finalize(self):
        pass
