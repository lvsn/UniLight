<h1 align="center">UniLight: Unified Multi-Modal Lighting Representation</h1>

<p align="center">
<a href="https://lvsn.github.io/UniLight"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://arxiv.org/abs/2512.04267"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/lvsn/UniLight" />
</a>
</p>

<p align="center"><strong>CVPR 2026</strong></p>



<p align="center">
  <a href="https://zzt76.github.io/">Zitian Zhang</a><sup>1,2</sup>&nbsp;&nbsp;
  <a href="https://iliyan.com/">Iliyan Georgiev</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://mfischer-ucl.github.io/">Michael Fischer</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://yannickhold.com/">Yannick Hold-Geoffroy</a><sup>2</sup>&nbsp;&nbsp;
  <br>
  <a href="http://www.jflalonde.ca/">Jean-François Lalonde</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://valentin.deschaintre.fr/">Valentin Deschaintre</a><sup>2*</sup>
</p>

<p align="center">
  <span><sup>1</sup>Université Laval</span>&nbsp;&nbsp;
  <span><sup>2</sup>Adobe Research</span>
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="100%">
</p>

UniLight introduces a joint latent space to unify previously incompatible lighting representation - environment maps, images, irradiance and text descriptions. Our joint lighting embedding enables applications such as retrieval, example-based light control during image generation, and environment map generation from various modalities.

## Installation

```bash
conda create -n vlm python=3.11
conda activate vlm
pip install -r requirements.txt
```

> **Note:** Pin `transformers`. Version 4.57.3 has a bug that breaks loading local model checkpoints.



## Pretrained Checkpoints

Pretrained weights are available at [🤗 Hugging Face](https://huggingface.co/zzt76/UniLight).

Download the checkpoints and place them under `checkpoints/`:

| Model | Path | Purpose |
|---|---|---|
| UniLight Encoder | `checkpoints/unilight_outputs/8_tokens_sh3-1024x1024_512/checkpoint-final` | Unified lighting encoder |
| LDR Envmap Generation | `checkpoints/envmap_outputs/8_tokens_sh3-1024x1024_512/checkpoint-100000` | SD3.5 fine-tuned for envmap generation |
| Relighting | `checkpoints/light_outputs/8_tokens_sh3-1024x1024_512/checkpoint-100000` | SD3.5 fine-tuned for scene relighting |

The envmap generation and relighting models also require the base `stabilityai/stable-diffusion-3.5-medium` model from HuggingFace Hub.



## UniLight Encoder

The core of the repo is `LightMultiEncoderModel`, a multi-modal encoder that produces a unified lighting embedding from any of the four supported modalities.

### Supported modalities

| Modality key | Input | What it captures |
|---|---|---|
| `envmap` | HDR `.exr` / `.hdr`, LDR `.png` / `.jpg` | Full environment map |
| `irradiance` | LDR `.png` / `.jpg` | Diffuse irradiance image |
| `rgb` | LDR `.png` / `.jpg` | Perspective crop / photos |
| `light_description` | String | Text description of lighting |

### Basic usage

```python
from light_multi_encoder import LightMultiEncoderModel

model = LightMultiEncoderModel.from_pretrained("checkpoints/unilight_outputs/8_tokens_sh3-1024x1024_512/checkpoint-final")
model.eval()
model.to("cuda")

# Get embeddings for any supported modality
modality = "envmap"  # or "irradiance", "rgb", "light_description"
features = model.get_modal_features(modal=modality, modal_data="path/to/envmap.exr")

# features["{modality}_embeds"]   — L2-normalised embeddings  [B, K, D]  (used for retrieval)
# features["{modality}_mu"]       — un-normalised embeddings  [B, K, D]  (used for conditional generation)
# features["{modality}_sh_pred"]  — predicted SH coefficients [B, 3, N^2]  (N = SH order, used for relighting, visualization, etc.)
```

To compute pairwise cosine similarities across modalities and verify cross-modal alignment:

```bash
python inference_encoder.py \
    --envmap     examples/crop_and_envmap/dataset0/scene0/000_envmap.exr \
    --rgb        examples/crop_and_envmap/dataset0/scene0/000_crop.png \
    --irradiance examples/irradiance/dataset0/scene0/000_irradiance.png \
    --text       "Bright midday sunlight coming from the upper right, casting hard shadows"
```



## LDR Envmap Generation

Generate a 512×512 LDR environment map from any lighting cue. Pass one of `--envmap`, `--rgb`, `--irradiance`, or `--text`.

```bash
# From an environment map
python inference_envmap_sd3.py \
    --envmap examples/target_lighting/indoor_000_envmap.exr \
    --output output_envmap.png

# From a photo
python inference_envmap_sd3.py \
    --rgb    examples/crop_and_envmap/dataset0/scene0/000_crop.png \
    --output output_envmap.png

# From a text description
python inference_envmap_sd3.py \
    --text   "Overcast sky with soft diffuse light, no visible sun" \
    --output output_envmap.png
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--num_inference_steps` | `30` | Number of diffusion steps |
| `--guidance_scale` | `4.0` | Classifier-free guidance scale |
| `--seed` | `709` | Seed for reproducibility |
| `--weight_dtype` | `bf16` | `fp32`, `fp16`, or `bf16` |
| `--ckpt_path` | `checkpoints/envmap_outputs/...` | Fine-tuned SD3 transformer |
| `--encoder_ckpt_path` | `checkpoints/unilight_outputs/...` | Light encoder checkpoint |



## Relighting

Relight a scene given its intrinsics (depth, normal, albedo) and a target lighting. The intrinsics directory should contain files named `{idx}_depth.png`, `{idx}_normal.png`, and `{idx}_albedo.png`.

```bash
# Relight using a target HDR envmap
python inference_relighting_sd3.py \
    --intrinsics_dir examples/intrinsics/dataset0/scene0 \
    --intrinsics_idx 000 \
    --envmap         examples/target_lighting/outdoor_000_envmap.exr \
    --output         output_relit.png

# Relight using a text description of the target lighting
python inference_relighting_sd3.py \
    --intrinsics_dir examples/intrinsics/dataset0/scene0 \
    --intrinsics_idx 000 \
    --text           "Golden hour sunlight from the left, warm tones, long shadows" \
    --output         output_relit.png
```

To estimate the required intrinsics from a single image, you can use [RGB-X](https://github.com/zheng95z/rgbx), [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer), or any other intrinsic estimator that outputs depth, normal, and albedo maps.



## Dataset Creation

The training data pipeline starts from HDR environment maps and produces matched tuples (crop, envmap, irradiance, light description). Follow these steps in order.

### 1. Get HDR envmaps

You can prepare your own HDR panorama or use publicly available datasets. For example, download HDR panorama from one of these sources:

- **[Polyhaven](https://polyhaven.com/hdris)** — wide variety of outdoor and indoor scenes
- **[ULaval Indoor HDR](http://hdrdb.com/indoor/)** — fill the form and request access from Prof. Jean-François Lalonde at Université Laval
- **[ULaval Outdoor HDR](http://hdrdb.com/outdoor/)** — fill the form and request access from Prof. Jean-François Lalonde at Université Laval

### 2. Generate perspective crops and aligned envmaps

`dataset_creation/rotate_and_crop.py` takes each HDR panorama, samples multiple rotations in intervals, renders a perspective LDR crop for each viewpoint, and saves the paired equirectangular envmap. It handles naming conventions from Polyhaven, ULaval Indoor, and ULaval Outdoor datasets, to this format: dataset_name/scene_name/idx_filename

Edit the input/output folder paths at the bottom of the script, then run:

```bash
python dataset_creation/rotate_and_crop.py
```

This populates the output folder with `{i:03d}_crop.png` and `{i:03d}_envmap.exr` pairs.

### 3. Estimate scene intrinsics

For each crop, estimate depth, normal, and albedo maps using an intrinsics estimator. We recommend:

- **[RGB-X](https://github.com/zheng95z/rgbx)**
- **[DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer)** 
- **[Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)** (the cosmos version of DiffusionRenderer with improved estimation)

Save the outputs as `{i:03d}_depth.png`, `{i:03d}_normal.png`, and `{i:03d}_albedo.png` in a similar directory structure.

### 4. Localize bright light sources

`dataset_creation/find_bright_spots.py` scans the envmap folder and extracts the positions, flux, and size of the dominant light sources in each scene. The result is a JSON file used to augment the VLM prompt in the next step.

```bash
python dataset_creation/find_bright_spots.py
```

Output: `lightmods_light_info.json`

### 5. Generate light descriptions with a VLM

`dataset_creation/eval_vlm_light.py` feeds each (crop, envmap, light_info) tuple to a vision-language model and produces natural-language lighting descriptions. Supports InternVL3, Qwen3-VL, and Gemma-3. You can use the `--debug` flag to run a quick test on a few samples before scaling up to the full dataset.

```bash
# Single-node run
python dataset_creation/eval_vlm_light.py \
    --root_folder      output_lightmods/crop_and_envmap \
    --light_info_json  lightmods_light_info.json \
    --detail_level     low_one_paragraph

# Multi-node: split work across N workers
python dataset_creation/eval_vlm_light.py \
    --root_folder  output_lightmods/crop_and_envmap \
    --detail_level low_one_paragraph \
    --idx 0 --total 4
```

To condense one-paragraph descriptions into a few-word summary, run a second pass:

```bash
python dataset_creation/eval_vlm_light.py \
    --detail_level summary_few_words \
    --summary_json path/to/one_paragraph_results.json
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026unilight,
  title={UniLight: A Unified Representation for Lighting},
  author={Zhang, Zitian and Georgiev, Iliyan and Fischer, Michael and Hold-Geoffroy, Yannick and Lalonde, Jean-Fran{\c{c}}ois and Deschaintre, Valentin},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026}
}
```



## Acknowledgements

This work was partially supported by NSERC grant RGPIN-2020-04799 and partially done during Zhang’s internship at Adobe Research London. Computing resources were provided by Adobe and the Digital Research Alliance of Canada.

This implementation builds upon Hugging Face’s [Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers) libraries.