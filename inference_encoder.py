import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

from light_multi_encoder import LightMultiEncoderModel

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D or 2-D tensors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def print_similarity_table(embeddings: dict[str, torch.Tensor]) -> None:
    """Print a formatted pairwise cosine-similarity matrix."""
    keys = list(embeddings.keys())
    n = len(keys)
    col_w = max(len(k) for k in keys) + 2

    # Header
    header = " " * col_w + "".join(f"{k:>{col_w}}" for k in keys)
    print("\nPairwise Cosine Similarity")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for ki in keys:
        row = f"{ki:<{col_w}}"
        for kj in keys:
            sim = cosine_similarity(embeddings[ki], embeddings[kj])
            row += f"{sim:>{col_w}.4f}"
        print(row)
    print("=" * len(header))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test LightMultiEncoderModel embeddings")
    p.add_argument("--ckpt_path", default="checkpoints/unilight_outputs/8_tokens_sh3-1024x1024_512/checkpoint-final")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Optional per-modality inputs
    p.add_argument("--envmap",      default=None, help="HDR/LDR envmap path (*.exr, *.hdr, *.png, *.jpg, etc.)")
    p.add_argument("--rgb",         default=None, help="RGB crop image path")
    p.add_argument("--irradiance",  default=None, help="Irradiance image path")
    p.add_argument("--text",        default=None, help="Free-form light description string")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from: {args.ckpt_path}")
    model = LightMultiEncoderModel.from_pretrained(args.ckpt_path)
    model.eval()
    model.to(device)
    print(f"Model loaded.  Modalities: {model.light_modalities}\n")

    # ------------------------------------------------------------------
    # 2. Collect (modal_name -> input) pairs from CLI / fallbacks
    # ------------------------------------------------------------------
    # Build a map from modality key in the model to the user-supplied input.
    # We try to match by substring so 'envmap', 'envmap_diffusionlightturbo',
    # 'irradiance_prism', etc. all work naturally.
    encoder_modalities = model.light_modalities

    raw_inputs: dict[str, object] = {}

    if args.envmap:
        # find an encoder key that contains 'envmap'
        for k in encoder_modalities:
            if 'envmap' in k and k not in raw_inputs:
                raw_inputs[k] = args.envmap
                break

    if args.rgb:
        for k in encoder_modalities:
            if 'rgb' in k and k not in raw_inputs:
                raw_inputs[k] = args.rgb
                break

    if args.irradiance:
        for k in encoder_modalities:
            if 'irradiance' in k and k not in raw_inputs:
                raw_inputs[k] = args.irradiance
                break

    if args.text:
        for k in encoder_modalities:
            if 'light_description' in k and k not in raw_inputs:
                raw_inputs[k] = args.text
                break

    if not raw_inputs:
        print("No inputs provided.  Pass at least one of: --envmap, "
              "--rgb, --irradiance, --text\n")
        print("Available modalities in checkpoint:", encoder_modalities)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Get features for each modality
    # ------------------------------------------------------------------
    embeddings: dict[str, torch.Tensor] = {}

    print("Computing embeddings …")
    with torch.no_grad():
        for modal, data in raw_inputs.items():
            print(f"  [{modal}]  input = {data!r}")
            modal_embeds = model.get_modal_features(
                modal=modal,
                modal_data=data,
            )
            emb = modal_embeds[modal + '_embeds']   # [1, T, D]
            embeddings[modal] = emb.cpu()  # [T*D]
            print(f"         embedding shape = {embeddings[modal].shape}")

    # ------------------------------------------------------------------
    # 4. Print similarity table
    # ------------------------------------------------------------------
    if len(embeddings) >= 2:
        print_similarity_table(embeddings)
    elif len(embeddings) == 1:
        k = list(embeddings.keys())[0]
        print(f"\nOnly one modality ({k!r}) provided – nothing to compare.")
    else:
        print("\nNo embeddings were produced.")


if __name__ == "__main__":
    main()
