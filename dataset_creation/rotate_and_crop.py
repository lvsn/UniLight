import os
import glob
import numpy as np
from typing import Tuple
from tqdm import tqdm
import cv2
from envmap import EnvironmentMap, rotation_matrix
import ezexr
from torchvision.transforms import functional as tvf
from hdr_utils import reexpose_image, autoexpose
import json
import multiprocessing as mp
from functools import partial


def get_base_name_from_path(envmap_path: str, folder_type: str) -> str:
    """
    Extract base name from file path based on folder type

    Args:
        envmap_path: Full path to the environment map file
        folder_type: Type of folder ('polyhaven', 'ulaval_indoor', 'ulaval_outdoor', etc.)

    Returns:
        Base name for the environment map
    """
    filename = os.path.basename(envmap_path).split('.')[0]

    if folder_type == 'polyhaven':
        parts = filename.split('_')
        return '_'.join(parts[:-1]) if len(parts) > 1 else filename
    elif folder_type == 'ulaval_indoor':
        # Extract the part after the first underscore for ulaval_indoor
        # IndoorHDRDataset2018_9C4A0052-9b4fa1e4a1 -> 9C4A0052-9b4fa1e4a1
        if '_' in filename:
            return filename.split('_', 1)[1]  # Split on first underscore and take second part
        else:
            return filename
    elif folder_type == 'ulaval_outdoor':
        # Extract the part before " Panorama_hdr" for ulaval_outdoor
        # 9C4A0006 Panorama_hdr -> 9C4A0006
        if ' Panorama_hdr' in filename:
            return filename.split(' Panorama_hdr')[0]
        else:
            # Fallback: use everything before the first space
            return filename.split(' ')[0] if ' ' in filename else filename
    else:
        # Default behavior
        return filename


def rotate_and_crop_envmap_worker(
    args: Tuple[str, str, str, int, float, float, Tuple[int], dict]
) -> str:
    """
    Worker function for multiprocessing - processes a single environment map

    Args:
        args: Tuple containing (envmap_path, output_dir, folder_type, number_of_rotations, 
              camera_vfov, crop_aspect_ratio, output_resolution, meta_data)

    Returns:
        Status message for the processed file
    """
    envmap_path, output_dir, folder_type, number_of_rotations, camera_vfov, crop_aspect_ratio, output_resolution, meta_data = args

    # Get base filename based on folder type
    base_name = get_base_name_from_path(envmap_path, folder_type)

    try:
        rotate_and_crop_envmap(
            envmap_path=envmap_path,
            output_dir=output_dir,
            folder_type=folder_type,
            number_of_rotations=number_of_rotations,
            camera_vfov=camera_vfov,
            crop_aspect_ratio=crop_aspect_ratio,
            output_resolution=output_resolution,
            meta_data=meta_data
        )
        return f"✓ {base_name}"
    except Exception as e:
        return f"✗ {base_name}: {str(e)}"


def rotate_and_crop_envmap(
    envmap_path: str,
    output_dir: str,
    folder_type: str,
    number_of_rotations: int = 3,
    camera_vfov: float = 90,
    crop_aspect_ratio: float = 16/9,
    output_resolution: Tuple[int] = (1920, 1080),
    meta_data: dict = None
):
    """
    Process a single environment map with multiple rotations and crops

    Args:
        envmap_path: Path to input HDR environment map
        output_dir: Directory to save outputs
        folder_type: Type of folder ('polyhaven', 'ulaval_indoor', 'ulaval_outdoor')
        number_of_rotations: Number of rotations to generate
        camera_vfov: Camera vertical field of view in degrees
        crop_aspect_ratio: Aspect ratio for crops
        output_resolution: Resolution for crop outputs
        meta_data: Metadata dictionary (only used for polyhaven)
    """
    # Get base filename based on folder type
    base_name = get_base_name_from_path(envmap_path, folder_type)

    # Load the HDR environment map
    hdr_image = ezexr.imread(envmap_path)
    hdr_image = hdr_image.astype(np.float32)

    # Downsample to 2k to make processing faster
    process_resolution = 2048
    if hdr_image.shape[1] > 2048:
        # Resize to 2k
        hdr_image = cv2.resize(hdr_image, (int(hdr_image.shape[1] * process_resolution / hdr_image.shape[0]), process_resolution), interpolation=cv2.INTER_AREA)

    # Making samples by taking center crops while rolling the reexposed HDR image
    roll_factor = int(hdr_image.shape[1] / number_of_rotations)

    scene_type = None
    # Load metadata for current scene (only for polyhaven)
    if folder_type == 'polyhaven' and meta_data and base_name in meta_data:
        scene_meta = meta_data[base_name]
        scene_type = 'indoor' if 'indoor' in scene_meta['categories'] else 'outdoor'

    # Create output subdirectory for this envmap
    envmap_output_dir = os.path.join(output_dir, base_name) if scene_type is None else os.path.join(output_dir, scene_type, base_name)
    os.makedirs(envmap_output_dir, exist_ok=True)

    # Process each rotation
    for i in range(number_of_rotations):
        # Create EnvironmentMap object
        envmap = EnvironmentMap(hdr_image, "latlong")

        # Create perspective crop
        crop = envmap.project(
            vfov=camera_vfov,
            rotation_matrix=rotation_matrix(azimuth=0, elevation=0, roll=0),
            ar=crop_aspect_ratio,
            resolution=output_resolution,
            projection="perspective",
            mode="normal",
        )

        # Re-expose this crop
        # crop_ldr, reexposure_factor = genLDRimageMedian(
        #     crop,
        #     putMedianIntensityAt=0.45,
        #     returnIntensityMultiplier=True,
        #     gamma=1 / 2.2,
        # )
        crop_ldr, reexposure_factor = autoexpose(crop, exposure_factor=0.35)

        # # Save crop as EXR
        # crop_exr_path = os.path.join(envmap_output_dir, f"{base_name}_crop_{i:02d}.exr")
        # ezexr.imwrite(crop_exr_path, crop)

        crop_ldr = np.clip(crop_ldr ** (1 / 2.2), 0, 1.0)
        crop_ldr = np.nan_to_num(crop_ldr, nan=0.0, posinf=0.0, neginf=0.0)
        # Save crop as PNG
        crop_ldr = (crop_ldr * 255).astype(np.uint8)
        crop_png_path = os.path.join(envmap_output_dir, f"{i:03d}_crop.png")
        cv2.imwrite(crop_png_path, cv2.cvtColor(crop_ldr, cv2.COLOR_RGB2BGR))

        # Save envmap as EXR
        envmap_low_res = cv2.resize(hdr_image, (int(hdr_image.shape[1] * output_resolution[1] / hdr_image.shape[0]), output_resolution[1]), interpolation=cv2.INTER_AREA)
        envmap_low_res = reexpose_image(envmap_low_res, reexposure_factor)
        envmap_low_res = np.nan_to_num(envmap_low_res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        envmap_exr_path = os.path.join(envmap_output_dir, f"{i:03d}_envmap.exr")
        ezexr.imwrite(envmap_exr_path, envmap_low_res, pixel_type='half')

        # Save envmap as PNG (tone mapped)
        envmap_ldr = np.clip(envmap_low_res**(1/2.2), 0, 1)
        envmap_ldr = np.nan_to_num(envmap_ldr, nan=0.0, posinf=0.0, neginf=0.0)
        envmap_ldr = (envmap_ldr * 255).astype(np.uint8)
        envmap_png_path = os.path.join(envmap_output_dir, f"{i:03d}_envmap.png")
        cv2.imwrite(envmap_png_path, cv2.cvtColor(envmap_ldr, cv2.COLOR_RGB2BGR))

        # Roll the image for the next rotation
        hdr_image = np.roll(hdr_image, roll_factor, axis=1)


def process_multiple_folders(
    input_folders: dict,
    output_base_folder: str = "./processed_envmaps",
    meta_data_path: str = "./polyhaven_metadata.json",
    number_of_rotations: int = 3,
    camera_vfov: float = 90,
    crop_aspect_ratio: float = 16/9,
    output_resolution: Tuple[int] = (1920, 1080),
    num_processes: int = None
):
    """
    Process environment maps from multiple folders with different naming conventions

    Args:
        input_folders: Dictionary mapping folder_type to folder_path
                      e.g., {'ulaval_indoor': './ulaval_indoor'}
        output_base_folder: Base folder for all outputs
        meta_data_path: Path to metadata JSON file (only used for polyhaven)
        number_of_rotations: Number of rotations to generate
        camera_vfov: Camera vertical field of view in degrees
        crop_aspect_ratio: Aspect ratio for crops
        output_resolution: Resolution for crop outputs
        num_processes: Number of processes to use (None = auto-detect)
    """
    all_files = []

    # Collect all files from all folders
    for folder_type, input_folder in input_folders.items():
        if not os.path.exists(input_folder):
            print(f"Warning: Folder {input_folder} does not exist, skipping {folder_type}")
            continue

        # Find all EXR files in this folder
        pattern = os.path.join(input_folder, "*.exr")
        exr_files = glob.glob(pattern)

        if not exr_files:
            print(f"No *.exr files found in {input_folder} for {folder_type}")
            continue

        print(f"Found {len(exr_files)} files in {input_folder} ({folder_type})")

        # Create output folder for this type
        output_folder = os.path.join(output_base_folder, folder_type)
        os.makedirs(output_folder, exist_ok=True)

        # Add files to processing list
        for exr_file in exr_files:
            all_files.append((exr_file, output_folder, folder_type))

    if not all_files:
        print("No EXR files found in any of the input folders")
        return

    # # Load failed scenes txt file if it exists
    # failed_scenes_path = "failed_scene_names.txt"
    # if os.path.exists(failed_scenes_path):
    #     with open(failed_scenes_path, 'r') as f:
    #         failed_scenes = [line.strip() for line in f if line.strip()]
    #     print(f"Loaded {len(failed_scenes)} failed scenes from {failed_scenes_path}")
    #     # Filter all_files to only include failed scenes
    #     failed_files = []
    #     for exr_file, output_folder, folder_type in all_files:
    #         base_name = get_base_name_from_path(exr_file, folder_type)
    #         if any(base_name in scene for scene in failed_scenes):
    #             failed_files.append((exr_file, output_folder, folder_type))
    #     all_files = failed_files

    print(f"Total: {len(all_files)} HDR environment maps to process")

    # Load metadata if available (only used for polyhaven)
    meta_data = None
    if meta_data_path and os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)
        print(f"Loaded metadata from {meta_data_path}")

    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(all_files))
    else:
        num_processes = min(num_processes, len(all_files))

    print(f"Using {num_processes} processes for parallel processing")

    # Prepare arguments for worker function
    worker_args = [
        (
            exr_file,
            output_folder,
            folder_type,
            number_of_rotations,
            camera_vfov,
            crop_aspect_ratio,
            output_resolution,
            meta_data
        )
        for exr_file, output_folder, folder_type in all_files
    ]

    # Process files in parallel with progress bar
    with mp.Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(rotate_and_crop_envmap_worker, worker_args),
            total=len(all_files),
            desc="Processing HDR environment maps"
        ))

    # Print summary of results
    successful = sum(1 for r in results if r.startswith("✓"))
    failed = sum(1 for r in results if r.startswith("✗"))

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved in {output_base_folder}")

    # Print failed files if any
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result.startswith("✗"):
                print(f"  {result}")


if __name__ == "__main__":
    # Configuration for multiple folders
    INPUT_FOLDERS = {
        'polyhaven': "./polyhaven",
        'ulaval_indoor': "./ulaval_indoor",
        'ulaval_outdoor': "./ulaval_outdoor",
    }
    # OUTPUT_BASE_FOLDER = "/mnt/localssd/processed_envmaps"
    OUTPUT_BASE_FOLDER = "/mnt/localssd/processed_envmaps_failed"
    META_DATA_PATH = "./polyhaven_metadata.json"

    # Rotation angles (in degrees) - 3 rotations as requested
    # NUMBER_OF_ROTATIONS = 3
    NUMBER_OF_ROTATIONS = 9

    # Camera and crop settings
    CAMERA_VFOV = 90  # degrees
    CROP_ASPECT_RATIO = 1/1  # width/height
    OUTPUT_RESOLUTION = (512, 512)  # width, height

    # Multiprocessing settings
    NUM_PROCESSES = 24  # None = auto-detect based on CPU count
    # NUM_PROCESSES = 4   # Uncomment to set specific number of processes

    # Process all environment maps from multiple folders
    process_multiple_folders(
        input_folders=INPUT_FOLDERS,
        output_base_folder=OUTPUT_BASE_FOLDER,
        meta_data_path=META_DATA_PATH,
        number_of_rotations=NUMBER_OF_ROTATIONS,
        camera_vfov=CAMERA_VFOV,
        crop_aspect_ratio=CROP_ASPECT_RATIO,
        output_resolution=OUTPUT_RESOLUTION,
        num_processes=NUM_PROCESSES
    )
