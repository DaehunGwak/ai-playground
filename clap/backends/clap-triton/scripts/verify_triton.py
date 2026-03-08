"""Verify clap-triton outputs match clap-fastapi outputs.

Usage:
    # Both services must be running
    python scripts/verify_triton.py \\
        --fastapi http://localhost:8000 \\
        --triton http://localhost:8001 \\
        --audio-file /path/to/audio.wav

Exit code: 0 on success, 1 on failure.
"""

import argparse
import sys

import numpy as np
import requests


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def get_text_embedding(base_url: str, texts: list[str]) -> np.ndarray:
    resp = requests.post(
        f"{base_url}/embed/text",
        json={"texts": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"], dtype=np.float32)


def get_audio_embedding(base_url: str, audio_path: str) -> np.ndarray:
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/embed/audio",
            files={"file": (audio_path.split("/")[-1], f, "audio/wav")},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"], dtype=np.float32)


def check_l2_norm(embeddings: np.ndarray, label: str, tol: float = 1e-5) -> bool:
    norms = np.linalg.norm(embeddings, axis=-1)
    ok = bool(np.all(np.abs(norms - 1.0) < tol))
    print(f"  [{label}] L2 norms: {norms} — {'OK' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fastapi", default="http://localhost:8000", help="clap-fastapi base URL"
    )
    parser.add_argument(
        "--triton", default="http://localhost:8080", help="clap-triton base URL"
    )
    parser.add_argument("--audio-file", default=None, help="Audio file for comparison")
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.999,
        help="Minimum cosine similarity to pass",
    )
    args = parser.parse_args()

    test_texts = [
        "a dog barking in the distance",
        "heavy metal guitar riff",
        "calm piano music",
    ]

    all_pass = True

    # --- Text embedding test ---
    print("=== Text Embedding Test ===")
    fa_emb = get_text_embedding(args.fastapi, test_texts)
    tr_emb = get_text_embedding(args.triton, test_texts)

    print(f"clap-fastapi shape: {fa_emb.shape}")
    print(f"clap-triton  shape: {tr_emb.shape}")

    all_pass &= check_l2_norm(fa_emb, "fastapi")
    all_pass &= check_l2_norm(tr_emb, "triton")

    for i, text in enumerate(test_texts):
        sim = cosine_similarity(fa_emb[i], tr_emb[i])
        ok = sim >= args.sim_threshold
        print(f"  [{i}] '{text}' → cosine sim: {sim:.6f} — {'OK' if ok else 'FAIL'}")
        all_pass &= ok

    # --- Audio embedding test ---
    if args.audio_file:
        print("\n=== Audio Embedding Test ===")
        fa_audio = get_audio_embedding(args.fastapi, args.audio_file)
        tr_audio = get_audio_embedding(args.triton, args.audio_file)

        print(f"clap-fastapi shape: {fa_audio.shape}")
        print(f"clap-triton  shape: {tr_audio.shape}")

        all_pass &= check_l2_norm(fa_audio, "fastapi")
        all_pass &= check_l2_norm(tr_audio, "triton")

        sim = cosine_similarity(fa_audio[0], tr_audio[0])
        ok = sim >= args.sim_threshold
        print(
            f"  Audio cosine sim: {sim:.6f} — {'OK' if ok else 'FAIL'} "
            f"(threshold={args.sim_threshold})"
        )
        all_pass &= ok
    else:
        print("\n[skip] No audio file provided (use --audio-file)")

    # --- Schema validation ---
    print("\n=== Schema Validation ===")
    resp = requests.post(
        f"{args.triton}/embed/text",
        json={"texts": ["test"]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    schema_ok = (
        "embeddings" in data
        and "dimension" in data
        and "count" in data
        and data["dimension"] == 512
        and data["count"] == 1
        and len(data["embeddings"][0]) == 512
    )
    print(f"  Response schema: {'OK' if schema_ok else 'FAIL'}")
    all_pass &= schema_ok

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
