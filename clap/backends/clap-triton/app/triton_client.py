import numpy as np
import tritonclient.grpc as grpcclient


class TritonCLAPClient:
    def __init__(self, url: str):
        self.client = grpcclient.InferenceServerClient(url=url)

    def is_ready(self) -> bool:
        try:
            return (
                self.client.is_server_ready()
                and self.client.is_model_ready("clap_text")
                and self.client.is_model_ready("clap_audio")
            )
        except Exception:
            return False

    def embed_text(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        input_ids: np.ndarray (B, 77) int64
        attention_mask: np.ndarray (B, 77) int64

        Returns
        -------
        np.ndarray (B, 512) float32
        """
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [grpcclient.InferRequestedOutput("embeddings")]
        response = self.client.infer("clap_text", inputs, outputs=outputs)
        return response.as_numpy("embeddings")

    def embed_audio(self, waveform: np.ndarray, longer: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        waveform: np.ndarray (B, 480000) float32
        longer: np.ndarray (B, 1) bool

        Returns
        -------
        np.ndarray (B, 512) float32
        """
        inputs = [
            grpcclient.InferInput("waveform", waveform.shape, "FP32"),
            grpcclient.InferInput("longer", longer.shape, "BOOL"),
        ]
        inputs[0].set_data_from_numpy(waveform)
        inputs[1].set_data_from_numpy(longer)

        outputs = [grpcclient.InferRequestedOutput("embeddings")]
        response = self.client.infer("clap_audio", inputs, outputs=outputs)
        return response.as_numpy("embeddings")
