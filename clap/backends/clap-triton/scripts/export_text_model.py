"""Export CLAP text encoder to ONNX for Triton ONNX Runtime backend.

Usage:
    python scripts/export_text_model.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --output triton/model_repository/clap_text/1/model.onnx

The exported model:
    Inputs:
        input_ids (INT64, [batch, 77]): RoBERTa token IDs
        attention_mask (INT64, [batch, 77]): attention mask
    Outputs:
        embeddings (FP32, [batch, 512]): L2-normalized text embeddings

Pipeline: RobertaModel → pooler_output → text_projection → L2 norm
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAPTextONNX(nn.Module):
    """Wrapper for ONNX export: RobertaModel + text_projection + L2 norm."""

    def __init__(self, text_branch, text_projection):
        super().__init__()
        self.text_branch = text_branch
        self.text_projection = text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pooler_output = self.text_branch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["pooler_output"]  # (B, 768)
        x = self.text_projection(pooler_output)  # (B, 512)
        x = F.normalize(x, dim=-1)  # (B, 512)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="CLAP checkpoint .pt path (None = auto-download)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="triton/model_repository/clap_text/1/model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    print("Loading CLAP model...")
    import laion_clap

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    if args.checkpoint:
        clap.load_ckpt(args.checkpoint)
    else:
        clap.load_ckpt()
    clap.model.eval()

    text_branch = clap.model.text_branch
    text_projection = clap.model.text_projection

    wrapper = CLAPTextONNX(text_branch, text_projection)
    wrapper.eval()

    # Dummy inputs for tracing
    batch_size = 2
    seq_len = 77
    dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)

    # Verify output shape
    with torch.no_grad():
        out = wrapper(dummy_input_ids, dummy_attention_mask)
    print(f"Wrapper output shape: {out.shape}")  # (2, 512)
    assert out.shape == (batch_size, 512), f"Unexpected shape: {out.shape}"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Exporting to {args.output} (opset={args.opset})...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        args.output,
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )
    print("ONNX export complete.")

    # Verify ONNX model
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(args.output)
        out_ort = sess.run(
            ["embeddings"],
            {
                "input_ids": dummy_input_ids.numpy(),
                "attention_mask": dummy_attention_mask.numpy(),
            },
        )[0]
        out_pt = out.numpy()
        max_diff = float(np.abs(out_ort - out_pt).max())
        print(f"PyTorch vs ONNX max diff: {max_diff:.6f}")
        assert max_diff < 1e-4, f"ONNX verification failed: max_diff={max_diff}"
        print("ONNX verification passed.")
    except ImportError:
        print("onnxruntime not installed; skipping ONNX verification.")


if __name__ == "__main__":
    main()
