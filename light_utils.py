import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import ezexr
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any, Tuple, Optional


def rgb2srgb(rgb):
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1 / 2.4) - 0.055)


def reinhard(x, max_point=16):
    # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    # lumi = lumi[..., None]
    # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
    # y_rein = x / (1+x)
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein


def apply_ev_and_tonemap(hdr_envmap: torch.Tensor, ev_value: float, max_point: float = 16) -> torch.Tensor:
    """
    Apply exposure value (EV) adjustment to HDR envmap and convert to LDR.

    Args:
        hdr_envmap: HDR environment map tensor of shape [batch, channels, height, width]
        ev_value: Exposure value to apply (in stops)
        max_point: Maximum point for Reinhard tone mapping

    Returns:
        LDR environment map tensor in sRGB space, range [0, 1]
    """
    # Apply exposure adjustment: multiply by 2^EV
    exposure_scale = 2.0 ** ev_value
    hdr_adjusted = hdr_envmap * exposure_scale

    # Clamp to valid range
    # ldr_linear = reinhard(hdr_adjusted, max_point=max_point).clamp(0, 1)
    ldr_linear = hdr_adjusted.clamp(0, 1)

    # Convert to sRGB
    ldr_srgb = rgb2srgb(ldr_linear)

    return ldr_srgb


def hdr_mapping(env_hdr, log_scale=1000):
    """Map HDR environment maps to LDR and logarithmic representations."""
    env_ldr = rgb2srgb(reinhard(env_hdr, max_point=16).clamp(0, 1))
    env_log = rgb2srgb(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
    return env_ldr, env_log


def encode_envmap(img_tensor: torch.Tensor, log_scale: int = 1000):
    """
    Given an HDR environment map array `img` (HxWx3), compute:
    - Eldr: Reinhard tonemapped LDR image
    - Elog: log-space normalized image
    - Edir: directional unit-vectors for each pixel

    Returns:
    Eldr:  float32, 3xHxW, in [-1,1] 
    Elog:  float32, 3xHxW, in [-1,1]
    Edir:  float32, 3xHxW, unit vectors
    """
    _, H, W = img_tensor.shape
    device = img_tensor.device

    # 0) We don't need to convert the envmap back to [0,1] if normalize was applied, since its range is [0, inf]
    img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=65504.0, neginf=0.0)  # Handle NaNs and Infs

    # 1) Reinhard tonemapping, in the range of [0, 1]
    # 2) Log encoding, in the range of [0, 1]
    Eldr, Elog = hdr_mapping(img_tensor, log_scale=log_scale)

    # 3) Directional encoding, in the range of [-1, 1]
    #    For equirectangular:
    #      u ∈ [0,1) → θ = 2π u
    #      v ∈ [0,1] → φ = π v
    #    Camera coordinate system: x right, y up, z forward
    #    Direction vector:
    #      x =  sin(θ) * sin(φ)
    #      y =  cos(φ)
    #      z =  cos(θ) * sin(φ)

    # Create coordinate grids
    u = torch.linspace(0, 1, W, dtype=torch.float32, device=device)
    v = torch.linspace(0, 1, H, dtype=torch.float32, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    # Convert to spherical coordinates
    theta = 2 * torch.pi * u_grid  # azimuthal angle
    phi = torch.pi * v_grid        # polar angle

    # Convert to Cartesian direction vectors
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(phi)
    z = torch.cos(theta) * torch.sin(phi)

    # Stack to create direction map: (3, H, W)
    Edir = torch.stack([x, y, z], dim=0)

    # Normalize to [-1, 1] if requested
    Eldr = Eldr * 2.0 - 1.0
    Elog = Elog * 2.0 - 1.0
    # Edir is already in [-1, 1]

    return Eldr, Elog, Edir


def preprocess_image(
    modal: str,
    image,
    image_size: Optional[Tuple[int, int]] = (512, 512),
    envmap_size: Optional[Tuple[int, int]] = (512, 512),
    log_scale: int = 1000,
) -> 'torch.Tensor':

    is_envmap = 'envmap' in modal
    is_hdr = isinstance(image, str) and (
        image.lower().endswith('.exr') or image.lower().endswith('.hdr')
    )

    # 1. Load image as float32 numpy array with shape (H, W, 3), [0, inf], for HDR or [0, 1] for LDR.
    if is_hdr:
        img_np = ezexr.imread(image).astype(np.float32)
        img_np = img_np[:, :, :3]  # ensure 3 channels
    elif isinstance(image, str):
        # LDR file path
        img_np = np.array(Image.open(image).convert('RGB')).astype(np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    else:
        # PIL.Image
        img_np = np.array(image.convert('RGB')).astype(np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    # 2. Resize and encode
    if is_envmap:
        h, w = envmap_size
        if img_np.shape[0] != h or img_np.shape[1] != w:
            interp = cv2.INTER_AREA
            img_np = cv2.resize(img_np, (w, h), interpolation=interp)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]

        Eldr, Elog, Edir = encode_envmap(img_tensor, log_scale=log_scale)

        if not is_hdr:
            # LDR envmap: use the loaded image directly as Eldr (no tonemapping), and zero out Elog (-1 matches the RandomDropElog convention).
            Eldr = img_tensor * 2.0 - 1.0
            Elog = torch.full_like(Elog, -1.0)

        return torch.cat([Eldr, Elog, Edir], dim=0)  # [9, H, W]

    else:
        # LDR image (rgb, irradiance, etc.)
        h, w = image_size
        if img_np.shape[0] != h or img_np.shape[1] != w:
            img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]
        img_tensor = img_tensor * 2.0 - 1.0
        return img_tensor


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def encode_intrinsics(combined_intrinsics_tensor: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    """
    Encode the intrinsics from the combined tensor using the VAE.
    """
    # Extract the intrinsics from the combined tensor
    intrinsics = [combined_intrinsics_tensor[:, i, :, :, :] for i in range(combined_intrinsics_tensor.shape[1])]

    # Encode each intrinsic using the VAE
    encoded_intrinsics = []
    for intrinsic in intrinsics:
        latent = vae.encode(intrinsic.to(vae.dtype)).latent_dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        encoded_intrinsics.append(latent.to(weight_dtype))

    return torch.cat(encoded_intrinsics, dim=1)  # Stack along a new dimension


def visualize_cosine_similarity(similarity_matrix, x_titles, y_titles,
                                image_title, save_path,
                                v_min=None, v_max=None,
                                cmap='YlOrRd'
                                ):
    """
    Visualizes a 2D NumPy array representing cosine similarity.

    Args:
        similarity_matrix (np.ndarray): The 2D array of cosine similarity values.
        x_titles (list): A list of strings for the x-axis labels.
        y_titles (list): A list of strings for the y-axis labels.
        image_title (str): The title of the image.
        save_path (str): The path to save the generated image.
        v_min (float, optional): Minimum value for colormap normalization. Defaults to None.
        v_max (float, optional): Maximum value for colormap normalization. Defaults to None.
        cmap (str, optional): Colormap to use. Defaults to 'YlOrRd', 'coolwarm'.
    """
    fig, ax = plt.subplots()

    # Create the colored grid
    cax = ax.matshow(similarity_matrix, cmap=cmap, vmin=v_min, vmax=v_max)

    # Add a color bar
    fig.colorbar(cax)

    # Over-plot each cell with its value
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                    ha='center', va='center', color='black')

    # Put the y-axis on the left (top 0)
    ax.set_yticks(np.arange(len(y_titles)))
    ax.set_yticklabels(y_titles)

    # Put the x-axis on the top (left 0)
    ax.set_xticks(np.arange(len(x_titles)))
    ax.set_xticklabels(x_titles, rotation=90)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Save the figure
    plt.title(image_title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def convert_light_descriptions_to_list(light_descriptions,
                                       light_categories=['Direct Lighting',
                                                         'Indirect Lighting',
                                                         'Visibility'],
                                       mode='summary'):

    selected_categories = light_categories
    match mode:
        case 'three_categories':
            selected_categories = light_categories  # Use all categories
        case 'three_categories_random':
            selected_categories = [light_categories[0]]  # keep the direct lighting, and randomly select from the next two
            selected_categories += random.sample(light_categories[1:], k=random.randint(0, 1))
        case 'direct_only':
            selected_categories = [light_categories[0]]  # Only use direct lighting
        case m if 'summary' in m:
            selected_categories = ['One Paragraph']
        case _:
            selected_categories = ['One Paragraph']
    # Convert the light descriptions dictionary to a list of strings
    light_descriptions_processed = []
    for b in range(len(list(light_descriptions.values())[0])):
        cur_light_descriptions = []
        for category in selected_categories:
            if category in light_descriptions:
                cur_light_descriptions.append(f"{light_descriptions[category][b]}")
        light_descriptions_processed.append(' '.join(cur_light_descriptions))

    return light_descriptions_processed


def tokenize_light_descriptions(tokenizer, light_descriptions, device, mode='summary'):
    task = "You are an embedding model. Encode the scene lighting description for similarity search and image generation conditioning. The embeddings must capture "
    match mode:
        case m if 'summary' in m:
            task += "the position (left, right, top, bottom, front, back, above, down, etc.) and the color of the dominant light sources (very important). And include the overall brightness, color temperature, and mood of the scene."
        case _:
            task += "the position (left, right, top, bottom, front, back, above, down, etc.) and the color of the dominant light sources (very important). And include the overall brightness, color temperature, and mood of the scene."
    if isinstance(light_descriptions, dict):
        # If the values are lists, join them per column into a list of strings, e.g., join light_description['Direct Lighting'][b] with light_description['Indirect Lighting'][b] for all examples in this batch
        light_descriptions = convert_light_descriptions_to_list(light_descriptions, mode=mode)
    elif isinstance(light_descriptions, str):
        # If light_descriptions is a single string, wrap it in a list
        light_descriptions = [light_descriptions]

    light_descriptions = [f'Instruct: {task} Query: {desc}' for desc in light_descriptions]
    tokenized = tokenizer(light_descriptions, padding=True, max_length=1000, truncation=True, return_tensors="pt")  # default max_length=8192
    tokenized = tokenized.to(device=device)
    return tokenized


def find_unused_parameters(model: torch.nn.Module,
                           example_inputs: tuple,
                           loss_fn: callable) -> list[str]:
    """
    Runs a forward+backward on `model` with `example_inputs` and `loss_fn`,
    then returns a list of names of parameters whose .grad is still None.

    Args:
      model: any torch.nn.Module
      example_inputs: tuple of inputs to the forward() of model
      loss_fn: a function loss = loss_fn(output) that returns a scalar Tensor

    Returns:
      unused: list of parameter names that never participated in the backward pass
    """
    model.zero_grad()
    output = model(*example_inputs)
    # assume loss_fn returns a scalar tensor
    loss = loss_fn(output)
    loss.backward()

    unused = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused.append(name)
    return unused


def text_to_image(
    text: str,
    img_size: Tuple[int, int] = (512, 512),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    margin_ratio: float = 0.05,
    line_spacing: int = 4
) -> Image.Image:
    """
    Render `text` into an image using PIL’s built-in font (no external .ttf).

    Because load_default() is a fixed bitmap font (~11px tall),
    we cannot change its size—but we can still wrap and center.

    Args:
      text         String to render (supports newlines).
      img_size     Width, height in pixels.
      bg_color     Background RGB tuple.
      text_color   Text RGB tuple.
      margin_ratio Fractional padding (e.g. 0.05 → 5% on each side).
      line_spacing Extra pixels between lines.

    Returns:
      A PIL Image with the rendered text.
    """
    img_w, img_h = img_size
    margin = int(min(img_w, img_h) * margin_ratio)
    drawable_w = img_w - 2 * margin
    drawable_h = img_h - 2 * margin

    # Load default PIL font (fixed size ~11px)
    font = ImageFont.load_default()

    # Create a dummy draw for measurements
    dummy = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(dummy)

    # Precompute line height: prefer font.getmetrics(), fallback to bbox measurement
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        line_h = ascent + descent + line_spacing
    else:
        # Measure a string that includes ascenders + descenders to get full line height
        bbox = draw.textbbox((0, 0), "Aygjpq", font=font)
        line_h = (bbox[3] - bbox[1]) + line_spacing

    # Wrap each paragraph so no line exceeds drawable_w
    wrapped_lines = []
    for para in text.split("\n"):
        # estimate chars per line via average 'x' width
        bbox = draw.textbbox((0, 0), "x", font=font)
        avg_w = bbox[2] - bbox[0] or 1
        max_chars = drawable_w // avg_w
        wrapped = textwrap.fill(para, width=max_chars)
        wrapped_lines.extend(wrapped.split("\n"))
    if not wrapped_lines:
        wrapped_lines = [""]

    # Compute total text block height
    total_h = line_h * len(wrapped_lines) - line_spacing

    # Draw onto the real image
    img = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(img)
    # Start Y so block is vertically centered
    y = margin + max((drawable_h - total_h) // 2, 0)

    for line in wrapped_lines:
        draw.text((margin, y), line, font=font, fill=text_color)
        y += line_h

    return img


def tensor_to_numpy(img, initial_range=(0, 1)):
    # scale to [0, 1]
    img = img - initial_range[0]
    img = img / (initial_range[1] - initial_range[0])
    if img.dim() == 4:
        img = img.squeeze(0)
    return np.clip(img.permute(1, 2, 0).detach().cpu().numpy(), 0, 1)


def numpy_to_pil(img):
    img = (img * 255.0).astype("uint8")
    if img.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_img = Image.fromarray(img.squeeze())
    else:
        pil_img = Image.fromarray(img)
    return pil_img


def tensor_to_pil(img, initial_range=(0, 1)):
    img = tensor_to_numpy(img, initial_range)
    img = numpy_to_pil(img)
    return img


def tensor_to_pil_list(images, initial_range=(0, 1)):
    images = tensor_to_numpy_list(images, initial_range)
    images = numpy_to_pil_list(images)
    return images


def tensor_to_numpy_list(images, initial_range=(0, 1)):
    # scale to [0, 1]
    images = images - initial_range[0]
    images = images / (initial_range[1] - initial_range[0])
    return np.clip(images.permute(0, 2, 3, 1).cpu().numpy(), 0, 1)


def numpy_to_pil_list(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255.0).astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image, mode="RGB") for image in images]

    return pil_images


def impath_to_numpy(image_name, is_Gamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0
    if is_Gamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


def numpy_to_tensor(img):
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img


def impath_to_tensor(image_name, is_Gamma=False):
    img = impath_to_numpy(image_name, is_Gamma)
    img = numpy_to_tensor(img)
    return img


def minmax_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
