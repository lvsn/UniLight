import ezexr
import numpy as np
from skimage.measure import label
import math
import json
import os
from pathlib import Path
from tqdm import tqdm


def coords_to_spherical(x, y, width, height):
    """
    Converts pixel (x, y) coordinates in an equirectangular map to spherical 
    (theta, phi) coordinates in radians.
    theta = 0 at the center (horizon), positive upward, negative downward.
    phi = 0 at the front, increases counter-clockwise.
    """
    u = x / width
    v = y / height
    phi = (u - 0.5) * 2 * np.pi
    theta = (0.5 - v) * np.pi
    return theta, phi


def describe_position(theta_rad, phi_rad):
    """
    Converts spherical coordinates (in radians) into a human-readable string.
    theta = 0 at horizon, positive upward, negative downward (range: -π/2 to π/2)
    """
    # Vertical description (based on theta)
    # theta ranges from -π/2 (bottom) to π/2 (top), with 0 at horizon
    if np.pi / 4 < theta_rad <= np.pi / 2:
        vertical_desc = "high up"
    elif np.pi / 8 < theta_rad <= np.pi / 4:
        vertical_desc = "up"
    elif -np.pi / 8 <= theta_rad <= np.pi / 8:
        vertical_desc = ""
    elif -np.pi / 4 <= theta_rad < -np.pi / 8:
        vertical_desc = "down"
    else:  # theta_rad < -np.pi / 4
        vertical_desc = "low down"

    # Horizontal description (based on phi)
    phi_deg = np.rad2deg(phi_rad)
    if -22.5 <= phi_deg < 22.5:
        horizontal_desc = "in the front"
    elif 22.5 <= phi_deg < 67.5:
        horizontal_desc = "on the front-right"
    elif 67.5 <= phi_deg < 112.5:
        horizontal_desc = "on the right"
    elif 112.5 <= phi_deg < 157.5:
        horizontal_desc = "on the back-right"
    elif -67.5 <= phi_deg < -22.5:
        horizontal_desc = "on the front-left"
    elif -112.5 <= phi_deg < -67.5:
        horizontal_desc = "on the left"
    elif -157.5 <= phi_deg < -112.5:
        horizontal_desc = "on the back-left"
    else:
        horizontal_desc = "in the back"

    description = f"{vertical_desc.capitalize()}, {horizontal_desc}" if vertical_desc else horizontal_desc.capitalize()
    return description.strip()


def find_major_light_sources(exr_path, brightness_threshold=4, min_area_pixels=20, max_lights_to_report=5, verbose=False):
    """
    Analyzes an EXR file to find a few major light sources, filtered by size and importance.
    Returns a list of light source information dictionaries.
    """
    img = ezexr.imread(exr_path)
    height, width, _ = img.shape

    luminance = np.dot(img[..., :3], [0.2126, 0.7152, 0.0722])

    num_labels = 0
    while num_labels < 1 and brightness_threshold > 0.8:
        bright_mask = luminance > brightness_threshold
        labeled_areas, num_labels = label(bright_mask, return_num=True, connectivity=2)
        
        brightness_threshold /= math.sqrt(2)

    if num_labels == 0:
        print(f"No areas found with brightness above {brightness_threshold}.")

    # --- NEW: Collect and filter potential light sources ---
    all_lights = []
    filtered_lights = []
    for i in range(1, num_labels + 1):
        area_mask = (labeled_areas == i)
        area_size = np.sum(area_mask)

        # Calculate the total energy (flux) of the light source
        total_flux = np.sum(luminance[area_mask])

        # Find the brightest point within this valid area
        max_luminance_in_area = np.max(luminance[area_mask])
        y, x = np.argwhere((luminance == max_luminance_in_area) & area_mask)[0]

        temp_light_info = {
            'x': int(x),
            'y': int(y),
            'max_brightness': float(max_luminance_in_area),
            'area_size': int(area_size),
            'total_flux': float(total_flux)
        }
        if area_size >= min_area_pixels:
            all_lights.append(temp_light_info)
        else:
            filtered_lights.append(temp_light_info)

    # **2. Sort by Importance (total flux) and select the top N**
    # Sorting by total_flux gives a better sense of a light's impact than just peak brightness.
    all_lights.sort(key=lambda item: item['total_flux'], reverse=True)
    filtered_lights.sort(key=lambda item: item['total_flux'], reverse=True)
    if not all_lights and filtered_lights:
        all_lights.append(filtered_lights[0])

    if not all_lights and not filtered_lights:
        print(f"No light sources found in {exr_path}.")
        return []

    top_lights = all_lights[:max_lights_to_report]

    if verbose:
        print(f"Found {len(top_lights)} major light source(s) (out of {len(all_lights)} potential areas):\n")

    # --- Process and enrich light information ---
    light_info_list = []
    for i, light in enumerate(top_lights):
        theta_rad, phi_rad = coords_to_spherical(light['x'], light['y'], width, height)
        theta_deg, phi_deg = np.rad2deg(theta_rad), np.rad2deg(phi_rad)
        description = describe_position(theta_rad, phi_rad)

        light_info = {
            'rank': i,
            'total_flux': light['total_flux'],
            'max_brightness': light['max_brightness'],
            'area_size': light['area_size'],
            'u': light['x'] / width,
            'v': light['y'] / height,
            'pixel_x': light['x'],
            'pixel_y': light['y'],
            'theta_deg': float(theta_deg),
            'phi_deg': float(phi_deg),
            'theta_rad': float(theta_rad),
            'phi_rad': float(phi_rad),
            'position_description': description
        }
        light_info_list.append(light_info)

        if verbose:
            print(f"--- Light Source #{i+1} ---")
            print(f"  💡 Importance (Total Flux): {light['total_flux']:.2f}")
            print(f"  🔆 Brightness (Peak): {light['max_brightness']:.2f}")
            print(f"  📏 Area: {light['area_size']} pixels")
            print(f"  📍 Pixel Coords (x, y): ({light['x']}, {light['y']})")
            print(f"  🌐 Spherical (degrees): theta={theta_deg:.2f}°, phi={phi_deg:.2f}°")
            print(f"  🗺️ Position: {description}\n")

    return light_info_list


def process_envmap_folder(root_folder, output_json_path, brightness_threshold=4, min_area_pixels=30, max_lights_to_report=4):
    """
    Process all envmap files in the folder structure and save light information to JSON.

    Expected folder structure:
    root_folder/
        dataset_name/
            scene_name/
                000_envmap.exr
                001_envmap.exr
                ...

    Args:
        root_folder: Path to the root folder containing dataset folders
        output_json_path: Path where the JSON file will be saved
        brightness_threshold: Initial brightness threshold for light detection
        min_area_pixels: Minimum area size for a valid light source
        max_lights_to_report: Maximum number of light sources to report per image
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"Error: Root folder '{root_folder}' does not exist.")
        return

    # Find all envmap files
    envmap_files = list(root_path.glob("**/*_envmap.exr"))

    if not envmap_files:
        print(f"No envmap files found in '{root_folder}'")
        return

    print(f"Found {len(envmap_files)} envmap files to process...")

    # Initialize the nested dictionary structure
    light_data = {}

    # Process each envmap file
    for exr_path in tqdm(envmap_files, desc="Processing envmaps"):
        # Extract metadata from path
        # Expected: .../dataset_name/scene_name/XXX_envmap.exr
        relative_path = exr_path.relative_to(root_path)
        parts = relative_path.parts

        if len(parts) < 3:
            print(f"Warning: Unexpected path structure for {exr_path}, skipping...")
            continue

        dataset_name = parts[0]
        scene_name = parts[1]
        filename = parts[2]

        # Extract image name (e.g., "000" from "000_envmap.exr")
        image_name = filename.replace("_envmap.exr", "")

        # Analyze the envmap
        light_sources = find_major_light_sources(
            exr_path=str(exr_path),
            brightness_threshold=brightness_threshold,
            min_area_pixels=min_area_pixels,
            max_lights_to_report=max_lights_to_report,
            verbose=False
        )

        # Build nested dictionary structure
        if dataset_name not in light_data:
            light_data[dataset_name] = {}

        if scene_name not in light_data[dataset_name]:
            light_data[dataset_name][scene_name] = {}

        light_data[dataset_name][scene_name][image_name] = light_sources

    # Save to JSON file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(light_data, f, indent=2)

    print(f"\n✅ Processing complete!")
    print(f"📊 Processed {len(envmap_files)} envmap files")
    print(f"💾 Saved light information to: {output_json_path}")

    # Print summary statistics
    total_scenes = sum(len(scenes) for scenes in light_data.values())
    total_images = sum(len(images) for dataset in light_data.values() for images in dataset.values())
    print(f"📁 Datasets: {len(light_data)}")
    print(f"🏠 Scenes: {total_scenes}")
    print(f"🖼️ Images: {total_images}")


if __name__ == "__main__":
    # Example usage
    root_folder = "lightmods/crop_and_envmap"
    output_json = "light_sources_info.json"

    process_envmap_folder(
        root_folder=root_folder,
        output_json_path=output_json,
        brightness_threshold=4,
        min_area_pixels=20,
        max_lights_to_report=4
    )

    # Optional: Test with a single file to verify output
    # exr_file_path = "examples/target_lighting/indoor_000_envmap.exr"
    # lights = find_major_light_sources(
    #     exr_path=exr_file_path,
    #     brightness_threshold=4,
    #     min_area_pixels=30,
    #     max_lights_to_report=4,
    #     verbose=True
    # )
