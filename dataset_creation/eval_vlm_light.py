import torch
import os
import sys
import json
import re
import ezexr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLMoeForConditionalGeneration, AutoProcessor, AutoTokenizer
import argparse
import random

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from light_description_prompt import get_light_description_prompt

torch.set_float32_matmul_precision('high')

# Environment setup
os.environ['HF_HOME'] = '/mnt/localssd/.cache'

# To avoid slowdown in multi-process environments
torch.set_num_threads(8)
torch.set_num_interop_threads(8)


def split_sections(text):
    return {"One Paragraph": text.strip()}


def rgb2srgb(rgb):
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def reinhard(x, max_point=16):
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein


def hdr_mapping(env_hdr, log_scale=1000):
    """Map HDR environment maps to LDR and logarithmic representations."""
    env_ldr = rgb2srgb(reinhard(env_hdr, max_point=16).clamp(0, 1))
    env_log = rgb2srgb(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
    return env_ldr, env_log


def load_envmap_as_pil(exr_path, target_size=(512, 512)):
    """Load an HDR EXR envmap, tonemap with hdr_mapping, and return a PIL Image."""
    img_np = ezexr.imread(str(exr_path)).astype(np.float32)
    img_np = img_np[:, :, :3]  # ensure 3 channels
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]
    env_ldr, _ = hdr_mapping(img_tensor)

    env_ldr_np = (env_ldr.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(env_ldr_np)
    if target_size is not None:
        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    return pil_img


def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def infer_scene_type(dataset_name):
    """Infer whether a scene is indoor or outdoor from the dataset name."""
    if 'outdoor' in dataset_name.lower():
        return 'outdoor'
    return 'indoor'


def collect_samples(root_folder, idx=0, total=1):
    """
    Walk root_folder and collect all (dataset_name, scene_name, image_name, crop_path, envmap_path) tuples.
    Only includes entries where both the crop PNG and the envmap EXR exist.
    Splits across parallel workers using idx/total.

    Expected structure:
        root_folder/dataset_name/scene_name/NNN_crop.png
        root_folder/dataset_name/scene_name/NNN_envmap.exr
    """
    root_path = Path(root_folder)
    crop_files = sorted(root_path.glob("**/*_crop.png"))

    samples = []
    for crop_path in crop_files:
        rel = crop_path.relative_to(root_path)
        parts = rel.parts
        if len(parts) < 3:
            continue
        dataset_name = parts[0]
        scene_name = parts[1]
        filename = parts[2]
        image_name = filename.replace("_crop.png", "")

        envmap_path = crop_path.parent / f"{image_name}_envmap.exr"
        if not envmap_path.exists():
            continue

        samples.append((dataset_name, scene_name, image_name, crop_path, envmap_path))

    # Distribute samples across parallel workers
    samples = samples[idx::total]
    return samples


def create_argparser():
    parser = argparse.ArgumentParser()

    # Choose from available checkpoints like 'google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'google/gemma-3-8b-it', 'OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-14B-hf', or 'OpenGVLab/InternVL3-38B-hf'
    # parser.add_argument("--model_checkpoint", default='OpenGVLab/InternVL3-38B-hf', type=str, help="Model checkpoint to use for inference")
    parser.add_argument("--model_checkpoint", default='Qwen/Qwen3-VL-30B-A3B-Thinking', type=str, help="Model checkpoint to use for inference")

    parser.add_argument("--seed", default=None, type=str, help="Random seed for reproducibility")

    parser.add_argument("--save_dir", default='light_description', type=str, help="Directory to save the light description results")
    parser.add_argument("--backup_freq", default=200, type=int, help="Frequency of saving intermediate results to avoid data loss")

    # Root folder with crop/envmap files
    # Expected structure: root_folder/dataset_name/scene_name/NNN_crop.png + NNN_envmap.exr
    parser.add_argument("--root_folder", default='output_lightmods/crop_and_envmap', type=str,
                        help="Root folder with structure: root/dataset_name/scene_name/NNN_crop.png + NNN_envmap.exr")

    # Auxiliary light info JSON (produced by find_bright_spots.py)
    parser.add_argument("--light_info_json", default='lightmods_light_info.json', type=str,
                        help="Path to JSON with pre-computed light source info (from find_bright_spots.py)")

    # Choose the level of detail for the light description
    parser.add_argument("--detail_level", default='low_one_paragraph', type=str,
                        choices=['low_one_paragraph', 'summary', 'summary_one_sentence', 'summary_few_words'],
                        help="Level of detail for the light description.")

    parser.add_argument('--use_light_info', action='store_true', help='Use auxiliary light source information to assist the model in understanding the lighting conditions')
    parser.set_defaults(use_light_info=True)

    # speed optimization stuff
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for faster processing with smaller dataset')
    parser.add_argument('--no_torch_compile', dest='use_torch_compile', action='store_false',
                        help='by default we using torch compile for faster processing speed. disable it if your environment is lower than pytorch2.0')
    parser.set_defaults(use_torch_compile=True)

    # parallel processing
    parser.add_argument("--idx", default=0, type=int, help="index of the current process, useful for running on multiple nodes")
    parser.add_argument("--total", default=1, type=int, help="total number of processes")

    # Summary mode: path to a previously-generated one-paragraph JSON to summarize
    parser.add_argument("--summary_json", default=None, type=str,
                        help="Path to existing one-paragraph light description JSON (required when detail_level contains 'summary')")

    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(int(args.seed))
    torch_dtype = torch.bfloat16

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_checkpoint = args.model_checkpoint
    processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True, use_fast=True)
    if 'qwen' in model_checkpoint.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_checkpoint,
            device_map="cuda",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_checkpoint,
            device_map="cuda",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
    if args.use_torch_compile:
        # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # --- Collect samples from disk ---
    samples = collect_samples(args.root_folder, idx=args.idx, total=args.total)
    print(f"Found {len(samples)} samples in '{args.root_folder}' (worker {args.idx}/{args.total})")

    if args.debug:
        samples = random.sample(samples, min(10, len(samples)))

    # Dictionary to store all results
    light_description = {}

    # Output JSON path
    model_short = args.model_checkpoint.split("/")[1].split("-")[0]
    light_description_json = os.path.join(
        save_dir,
        f'light_description_{model_short}_{args.detail_level}_idx{args.idx}_total{args.total}.json'
    )
    os.makedirs(os.path.dirname(light_description_json), exist_ok=True)

    # Load existing progress if available
    if os.path.exists(light_description_json):
        print(f"Loading existing progress from {light_description_json}")
        with open(light_description_json, 'r') as f:
            light_description = json.load(f)
        print(f"Loaded {sum(len(v) if isinstance(v, dict) else 0 for v in light_description.values())} existing entries")

    # For summary modes, load the source one-paragraph descriptions
    cur_light_description_json = None
    if 'summary' in args.detail_level:
        if args.summary_json is None:
            raise ValueError("--summary_json must be provided when detail_level contains 'summary'")
        cur_light_description_json = json.load(open(args.summary_json, 'r'))

    # Auxiliary info for VLM to understand lighting
    if args.use_light_info:
        if not os.path.exists(args.light_info_json):
            print(f"Warning: light_info_json '{args.light_info_json}' not found; disabling light info.")
            args.use_light_info = False
        else:
            light_info = json.load(open(args.light_info_json, 'r'))

    # --- Main Processing Loop ---
    for i, (dataset_name, scene_name, image_name, crop_path, envmap_path) in enumerate(tqdm(samples)):

        scene_type = infer_scene_type(dataset_name)

        # Skip if already processed
        if dataset_name in light_description and \
           scene_name in light_description[dataset_name] and \
           image_name in light_description[dataset_name][scene_name]:
            if args.debug:
                print(f"Skipping already processed: {dataset_name}/{scene_name}/{image_name}")
            continue

        if args.detail_level == 'summary':
            system_prompt = f"You are an expert in analyzing {scene_type} scene lighting. Given existing lighting description, your task is to summarize current descriptions according to the provided images (cropped view, panorama)."
            user_prompt_template = f"""
            Important requirements:
            - Must give the correct and faithful description based on the lighting conditions of the scene
            - Use in total of two sentences, one for direct lighting that dominates the scene lighting, must describe what where these light sources are and their positions. And the other one describe the overall lighting, focus on the color (must have) and brightness
            - Do not include any additional information or context beyond the lighting description
            - These two sentences should be clearly separated, and very short and concise, like what humans will say
            - Make sure them flows naturally, and avoid redundancy
            - Write in complete sentences without using bullet points, dashes, or numbered lists
            - Do not use words expressing uncertainty like 'appears to be', 'seems to', 'likely', or 'suggests'. State the lighting conditions as fact
            
            Current light description: {cur_light_description_json[dataset_name][scene_name][image_name]['One Paragraph']}
            """
        elif args.detail_level == 'summary_one_sentence':
            system_prompt = f"You are an expert in analyzing {scene_type} scene lighting. Given existing lighting description, your task is to summarize current descriptions to one single sentence according to the provided images (cropped view and panorama) and the direct light source position and brightness information."
            user_prompt_template = f"""
            Important requirements:
            - Must give the correct and faithful description based on the lighting conditions of the scene
            - Use one sentence to summarize the lighting condition, must include predominant direct light sources (like "a bright sun from above to the left", etc.) and overall scene lighting.
            - Do not include any additional information or context beyond the lighting description
            - This one sentence should be very short and concise, like what humans will say
            - Make sure them flows naturally, and avoid redundancy
            - Write in complete sentences without using bullet points, dashes, or numbered lists
            - Do not use words expressing uncertainty like 'appears to be', 'seems to', 'likely', or 'suggests'. State the lighting conditions as fact
            
            Current light description: {cur_light_description_json[dataset_name][scene_name][image_name]['One Paragraph']}
            """
        elif args.detail_level == 'summary_few_words':
            system_prompt = f"You are an expert in analyzing {scene_type} scene lighting. Given existing lighting description, your task is to summarize current descriptions to a few words according to the provided images (cropped view and panorama) and the direct light source position and brightness information."
            user_prompt_template = f"""
            Important requirements:
            - Must give the correct and faithful description based on the lighting conditions of the scene
            - Use a few phrases (not complete sentences) to summarize the scene lighting condition
            - Describe predominant direct light sources and overall scene lighting
            - It's very important to give the color and position of the direct light sources
            - Do not include any additional information or context beyond the lighting description
            - Separate the phrases with commas
            - Do not use words expressing uncertainty like 'appears to be', 'seems to', 'likely', or 'suggests'. State the lighting conditions as fact

            Current light description: {cur_light_description_json[dataset_name][scene_name][image_name]['One Paragraph']}
            """
        else:
            system_prompt = f"You are an expert in analyzing {scene_type} scene lighting. Your task is to describe the lighting in the image with technical accuracy."
            user_prompt_template = get_light_description_prompt(scene_type, detail_level=args.detail_level)

        if args.use_light_info:
            light_info_prompt = "Here is some auxiliary information about the light sources in the scene to help you better understand the lighting conditions, but do not decribe the lighting with numbers:\n"
            cur_light_info = light_info.get(dataset_name, {}).get(scene_name, {}).get(image_name, [])
            if len(cur_light_info) == 0:
                light_info_prompt += "There is no direct light source in this scene.\n"
            else:
                light_info_prompt += "Direct Light Sources:\n"
                for light_idx, light in enumerate(cur_light_info):
                    light_desc = (
                        f"Light {light_idx + 1}: maximum brightness {light['max_brightness']}, "
                        f"position description: {light['position_description']}, "
                        f"theta (elevation angle on envmap, center is 0): {light['theta_deg']}, "
                        f"phi (azimuthal angle on envmap, center is 0): {light['phi_deg']} \n"
                    )
                    light_info_prompt += light_desc
            user_prompt_template = user_prompt_template + '\n' + light_info_prompt

        # --- Load crop image ---
        photo = Image.open(str(crop_path)).convert('RGB')
        photo = photo.resize((512, 512), Image.LANCZOS)

        # --- Load and tonemap envmap ---
        envmap = load_envmap_as_pil(envmap_path, target_size=(512, 512))

        if args.debug:
            temp_folder = 'temp'
            os.makedirs(temp_folder, exist_ok=True)
            photo.save(os.path.join(temp_folder, f"{dataset_name}_{scene_name}_{image_name}.jpg"))
            envmap.save(os.path.join(temp_folder, f"{dataset_name}_{scene_name}_{image_name}_envmap.jpg"))

        # --- Prepare messages for the model ---
        if 'gemma' in model_checkpoint.lower():
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": user_prompt_template}
                    ],
                },
            ]
        elif 'internvl' in model_checkpoint.lower() or 'qwen' in model_checkpoint.lower():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        # {"type": "image"},
                        {"type": "text", "text": system_prompt + '\n' + user_prompt_template},
                        # {"type": "text", "text": "What's the indoor lighting of the scene in the given photograph? Provide the description only without any prelude."},
                    ],
                },
            ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            # images=[photo, envmap, coordinate_image],
            images=[photo, envmap],
            text=prompt,
            return_tensors="pt"
        ).to(model.device, dtype=torch_dtype)

        input_len = inputs["input_ids"].shape[-1]

        # --- Generate Description ---
        max_new_tokens = 200 if 'summary' in args.detail_level else 2000
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output_ids = generated_ids[0][input_len:].tolist()  # Skip the input part of the generation

        # check if the generation has already finished (151645 is <|im_end|>)
        if 'qwen' in model_checkpoint.lower():
            if 151645 not in output_ids:
                # check if the thinking process has finished (151668 is </think>)
                # and prepare the second model input
                if 151668 not in output_ids:
                    print("thinking budget is reached")
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    early_stopping_ids = tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)
                    input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
                else:
                    input_ids = generated_ids
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

                # second generation
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # max_new_tokens=input_len + max_new_tokens - input_ids.size(-1)  # could be negative if max_new_tokens is not large enough (early stopping text is 24 tokens)
                    max_new_tokens=max_new_tokens  # could be negative if max_new_tokens is not large enough (early stopping text is 24 tokens)
                )
                output_ids = generated_ids[0][input_len:].tolist()
            
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        decoded_output = processor.decode(output_ids[index:], skip_special_tokens=True)
        
        decoded_output = decoded_output.replace('\n', ' ').strip()  # Normalize output by removing newlines and extra spaces

        if 'summary_few_words' in args.detail_level:
            decoded_output = decoded_output.strip('.')
            decoded_output = decoded_output.lower()

        # --- Parse the output into sections ---
        sections = split_sections(decoded_output)

        # Save to nested dict: dataset_name -> scene_name -> image_name
        if dataset_name not in light_description:
            light_description[dataset_name] = {}
        if scene_name not in light_description[dataset_name]:
            light_description[dataset_name][scene_name] = {}
        light_description[dataset_name][scene_name][image_name] = sections

        if args.debug:
            print(f"Successfully processed {dataset_name}/{scene_name}/{image_name} - Sections: {sections}")

        # Save progress periodically to the main file (allows resuming)
        if (i + 1) % args.backup_freq == 0:
            print(f"Saving progress ({i + 1} iterations) to {light_description_json}", flush=True)
            with open(light_description_json, 'w') as f:
                json.dump(light_description, f, indent=4)

    # --- Save Final Results ---
    print(f"Saving all descriptions to {light_description_json}")
    with open(light_description_json, 'w') as f:
        json.dump(light_description, f, indent=4)

    print("Script finished.")


if __name__ == "__main__":
    main()
