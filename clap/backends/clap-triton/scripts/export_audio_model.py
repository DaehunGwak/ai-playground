"""Extract CLAP audio encoder weights for Triton Python backend.

Usage:
    python scripts/export_audio_model.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --output-dir triton/model_repository/clap_audio/1/

Saved files:
    clap_audio_weights.pt  — state dicts for audio_branch and audio_projection
    audio_cfg.json         — audio configuration (for reference / debugging)

The Triton Python backend (model.py) reconstructs CLAP_Module with
enable_fusion=False, amodel="HTSAT-base" and loads these weights.

Pipeline: HTSAT(waveform_dict) → embedding (1024) → audio_projection → L2 norm → (512)
"""

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="CLAP checkpoint .pt path (None = auto-download)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="triton/model_repository/clap_audio/1",
        help="Output directory for audio weights",
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

    # Verify audio_projection input dim matches HTSAT-base embed_dim (1024)
    proj = clap.model.audio_projection
    first_linear = proj[0]
    print(
        f"audio_projection: Linear({first_linear.in_features}, {first_linear.out_features})"
    )
    assert first_linear.in_features == 1024, (
        f"Unexpected audio_projection input dim: {first_linear.in_features}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save state dicts
    weights_path = os.path.join(args.output_dir, "clap_audio_weights.pt")
    torch.save(
        {
            "audio_branch": clap.model.audio_branch.state_dict(),
            "audio_projection": clap.model.audio_projection.state_dict(),
        },
        weights_path,
    )
    print(f"Saved audio weights to {weights_path}")

    # Save audio_cfg for reference
    cfg_path = os.path.join(args.output_dir, "audio_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(clap.model_cfg.get("audio_cfg", {}), f, indent=2)
    print(f"Saved audio_cfg to {cfg_path}")

    # Smoke test: reconstruct and run forward pass
    print("Running smoke test...")
    clap2 = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    saved = torch.load(weights_path, map_location="cpu")
    clap2.model.audio_branch.load_state_dict(saved["audio_branch"])
    clap2.model.audio_projection.load_state_dict(saved["audio_projection"])
    clap2.model.eval()

    dummy_waveform = torch.zeros((1, 480000))
    dummy_longer = torch.tensor([False])

    with torch.no_grad():
        import torch.nn.functional as F

        feats = clap2.model.audio_branch(
            {"waveform": dummy_waveform, "longer": dummy_longer},
            mixup_lambda=None,
            device=None,
        )["embedding"]
        emb = clap2.model.audio_projection(feats)
        emb = F.normalize(emb, dim=-1)

    print(f"Smoke test output shape: {emb.shape}")  # (1, 512)
    norm = float(emb.norm(dim=-1).item())
    print(f"L2 norm: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-5, f"L2 norm not 1: {norm}"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
