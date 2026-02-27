"""Minimal inference script for envmap generation from a single lighting modality.

Example usage
-------------
python inference_envmap.py \
    --envmap examples/target_lighting/indoor_000_envmap.exr \
    --output output_envmap.png

python inference_envmap.py \
    --rgb examples/crop_and_envmap/dataset0/scene0/000_crop.png \
    --output output_envmap.png

python inference_envmap.py \
    --text "Warm sunlight coming from the upper left, golden hour." \
    --output output_envmap.png
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from pipelines.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from light_multi_encoder import LightMultiEncoderModel


PRETRAINED_MODEL = "stabilityai/stable-diffusion-3.5-medium"


def import_model_class(pretrained_model_name_or_path: str, revision: str | None, subfolder: str = "text_encoder"):
    cfg = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    arch = cfg.architectures[0]
    if arch == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif arch == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    raise ValueError(f"Unsupported text-encoder architecture: {arch}")


def get_light_embedding(
    light_model: LightMultiEncoderModel,
    modal: str,
    modal_data: object,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Return prompt_embeds [1, T, D] (mu from the encoder)."""
    with torch.no_grad():
        out = light_model.get_modal_features(modal=modal, modal_data=modal_data)
    mu = out[modal + "_mu"]
    if mu.dim() == 2:
        mu = mu.unsqueeze(1)  # [1, D] → [1, 1, D]
    return mu.to(device, dtype=weight_dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal envmap generation inference from a single lighting modality")

    # Target lighting – exactly one should be specified
    p.add_argument("--envmap",      default=None, help="Source envmap path (.exr, .hdr, .png, …)")
    p.add_argument("--rgb",         default=None, help="Source RGB lighting crop")
    p.add_argument("--irradiance",  default=None, help="Source irradiance image")
    p.add_argument("--text",        default="The primary natural light source is the sun, positioned in the upper left relative to the view, emitting warm-hued, strong brightness that casts distinct shadows across the scene. No artificial lights are active or visible. The overall color of the scene is warm with golden and earthy tones.", help="Source lighting description string")

    # Model checkpoints
    p.add_argument("--ckpt_path",         default="checkpoints/envmap_outputs/8_tokens_sh3-1024x1024_512/checkpoint-100000",
                   help="SD3 transformer checkpoint directory")
    p.add_argument("--encoder_ckpt_path", default="checkpoints/unilight_outputs/8_tokens_sh3-1024x1024_512/checkpoint-final",
                   help="LightMultiEncoderModel checkpoint directory")
    p.add_argument("--pretrained_model",  default=PRETRAINED_MODEL,
                   help="Base SD3 model identifier or local path")

    # Generation settings
    p.add_argument("--resolution",          type=int,   default=512)
    p.add_argument("--num_inference_steps", type=int,   default=30)
    p.add_argument("--guidance_scale",      type=float, default=4.0)
    p.add_argument("--seed",                type=int,   default=709)
    p.add_argument("--weight_dtype",        default="bf16", choices=["fp32", "fp16", "bf16"])

    # Output
    p.add_argument("--output", default="output_envmap.png", help="Output image path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    weight_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.weight_dtype]

    # ------------------------------------------------------------------
    # 1. Load the LightMultiEncoderModel and resolve modality key
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading LightMultiEncoderModel from: {args.encoder_ckpt_path}")
    light_model = LightMultiEncoderModel.from_pretrained(args.encoder_ckpt_path)
    light_model.eval()
    light_model.requires_grad_(False)
    light_model.to(device)

    encoder_modalities = light_model.light_modalities
    raw_inputs: dict[str, object] = {}

    if args.envmap:
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
        print("No inputs provided.  Pass at least one of: --envmap, --rgb, --irradiance, --text")
        print("Available modalities in checkpoint:", encoder_modalities)
        sys.exit(1)
    if len(raw_inputs) > 1:
        first_key = next(iter(raw_inputs))
        print(f"WARNING: Multiple modalities provided; using the first one: '{first_key}'")
        raw_inputs = {first_key: raw_inputs[first_key]}

    modal_key, modal_value = next(iter(raw_inputs.items()))
    print(f"   Resolved modality key: '{modal_key}'")

    # ------------------------------------------------------------------
    # 2. Load SD3 components
    # ------------------------------------------------------------------
    print(f"\n[2/3] Loading SD3 components from: {args.pretrained_model}")

    tokenizer_one   = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    tokenizer_two   = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_3")

    te_cls_one   = import_model_class(args.pretrained_model, revision=None, subfolder="text_encoder")
    te_cls_two   = import_model_class(args.pretrained_model, revision=None, subfolder="text_encoder_2")
    te_cls_three = import_model_class(args.pretrained_model, revision=None, subfolder="text_encoder_3")

    text_encoder_one   = te_cls_one.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    text_encoder_two   = te_cls_two.from_pretrained(args.pretrained_model, subfolder="text_encoder_2")
    text_encoder_three = te_cls_three.from_pretrained(args.pretrained_model, subfolder="text_encoder_3")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    sd3_transformer = SD3Transformer2DModel.from_pretrained(args.ckpt_path, subfolder="transformer")

    for m in [vae, sd3_transformer, text_encoder_one, text_encoder_two, text_encoder_three]:
        m.eval()
        m.requires_grad_(False)

    vae.to(device, dtype=torch.float32)  # VAE stays fp32 for stability
    sd3_transformer.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    text_encoder_three.to(device, dtype=weight_dtype)

    # ------------------------------------------------------------------
    # 3. Build pipeline
    # ------------------------------------------------------------------
    print(f"\n[3/3] Building StableDiffusion3Pipeline")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model,
        transformer=sd3_transformer,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        torch_dtype=weight_dtype,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    # ------------------------------------------------------------------
    # 4. Get light embedding
    # ------------------------------------------------------------------
    print(f"\n   Computing light embedding  [{modal_key}] ← {modal_value!r}")
    prompt_embeds = get_light_embedding(light_model, modal_key, modal_value, device, weight_dtype)
    print(f"   prompt_embeds shape: {prompt_embeds.shape}")

    # Pad to T5 hidden dimension if needed
    t5_d_model = text_encoder_three.config.d_model
    if prompt_embeds.shape[2] < t5_d_model:
        pad = t5_d_model - prompt_embeds.shape[2]
        prompt_embeds = F.pad(prompt_embeds, (0, pad), "constant", 0)

    # Pooled prompt embeddings: always zeros
    pooled_dim = text_encoder_one.config.hidden_size + text_encoder_two.config.hidden_size
    pooled_prompt_embeds = torch.zeros((1, pooled_dim), device=device, dtype=weight_dtype)

    negative_prompt_embeds        = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    # ------------------------------------------------------------------
    # 5. Generate
    # ------------------------------------------------------------------
    print(f"\n   Running inference  (steps={args.num_inference_steps}, CFG={args.guidance_scale}, seed={args.seed})")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.autocast(device.type, dtype=weight_dtype):
        with torch.inference_mode():
            result = pipeline(
                prompt=None,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

    image = result.images[0]

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image.save(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
