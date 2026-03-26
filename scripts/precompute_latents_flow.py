# precompute_latents_unified.py
"""
Unified script to precompute latents for images, masks, and sketches using a local VAE 
and Accelerate for multi-GPU processing.

This script finds triplets of (image, mask, sketch) for both CelebA-HQ and FFHQ datasets,
encodes them into latent space, and saves them for fast access during training.
"""

import os
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import argparse
import logging
import shutil
from accelerate import Accelerator
import gc
import threading

# Set NCCL timeout to 1 hour (in seconds) - much longer than default
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '3600'

# Thread-local storage for transforms to avoid potential race conditions
thread_local = threading.local()

# --- Helper Functions ---

def get_thread_local_transforms(resolution):
    """Get transforms for current thread to avoid race conditions."""
    if not hasattr(thread_local, 'transforms'):
        thread_local.transforms = {
            'image': transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            'sketch': transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        }
    return thread_local.transforms

def ensure_rgb(img: Image.Image) -> Image.Image:
    """Ensures an image is in RGB mode."""
    return img.convert("RGB") if img.mode != "RGB" else img

def ensure_grayscale_and_convert_to_rgb_tensor(img: Image.Image, transform: transforms.Compose):
    """Ensures an image is grayscale, applies transforms, and repeats channels to create an RGB-like tensor."""
    if img.mode != "L":
        img = img.convert("L")
    tensor = transform(img)
    return tensor.repeat(3, 1, 1) # Repeat single channel to 3 channels

def get_file_paths_for_celeba(config, split='train'):
    """Get file paths for CelebA-HQ style structure, assuming all triplets exist."""
    data_root = config['data']['data_root']
    text_root = config['data']['text_root']
    image_folder = config['data']['image_folder']
    mask_folder = config['data']['mask_folder']
    sketch_folder = config['data']['sketch_folder']
    mask_suffix = config['data']['mask_suffix']
    sketch_suffix = config['data']['sketch_suffix']
    text_suffix = config['data']['text_suffix']
    
    base_dir = Path(data_root) / split
    image_paths = sorted(list((base_dir / image_folder).glob("*.jpg"))) + \
                  sorted(list((base_dir / image_folder).glob("*.png")))

    valid_triplets = []
    logging.info(f"[CelebA-HQ] Assuming all triplets exist. Found {len(image_paths)} potential images.")

    for img_path in image_paths:
        base_name = img_path.stem
        mask_path = base_dir / mask_folder / f"{base_name}{mask_suffix}"
        sketch_path = base_dir / sketch_folder / f"{base_name}{sketch_suffix}"
        text_path = Path(text_root) / f"{base_name}{text_suffix}"

        # Assume all files exist - no verification
        valid_triplets.append((str(img_path), str(mask_path), str(sketch_path), str(text_path), base_name))
    
    logging.info(f"[CelebA-HQ] Created {len(valid_triplets)} triplets (assuming all exist).")
    return valid_triplets

def get_file_paths_for_ffhq(config):
    """Get file paths for FFHQ style structure, assuming all triplets exist."""
    data_root = config['ffhq_data']['data_root']
    mask_data_root = config['ffhq_data']['mask_data_root']
    sketch_data_root = config['ffhq_data']['sketch_data_root']
    text_root = config['ffhq_data']['text_root']
    mask_suffix = config['data']['mask_suffix']
    sketch_suffix = config['data']['sketch_suffix']
    text_suffix = config['data']['text_suffix']

    image_paths = sorted(list(Path(data_root).rglob("*.jpg"))) + \
                  sorted(list(Path(data_root).rglob("*.png")))

    valid_triplets = []
    logging.info(f"[FFHQ] Assuming all triplets exist. Found {len(image_paths)} potential images.")

    for img_path in image_paths:
        base_name = img_path.stem
        mask_path = Path(mask_data_root) / f"{base_name}{mask_suffix}"
        sketch_path = Path(sketch_data_root) / f"{base_name}{sketch_suffix}"
        text_path = Path(text_root) / f"{base_name}{text_suffix}"

        # Assume all files exist - no verification
        valid_triplets.append((str(img_path), str(mask_path), str(sketch_path), str(text_path), base_name))
            
    logging.info(f"[FFHQ] Created {len(valid_triplets)} triplets (assuming all exist).")
    return valid_triplets

def check_existing_latents(latent_dirs, base_name):
    """Check if all latent files already exist for a given base name."""
    return all((latent_dirs[key] / f"{base_name}.pt").exists() for key in ['image', 'mask', 'sketch'])

def remove_corrupted_latent_files(latent_dirs, base_name):
    """Remove corrupted latent files to allow re-processing."""
    for key in ['image', 'mask', 'sketch']:
        latent_file = latent_dirs[key] / f"{base_name}.pt"
        if latent_file.exists():
            try:
                # Try to load the file to check if it's corrupted
                torch.load(latent_file)
            except:
                # If loading fails, remove the corrupted file
                latent_file.unlink()
                logging.warning(f"Removed corrupted latent file: {latent_file}")

def precompute_latents_for_resolution(accelerator, vae, triplets, output_dir, dataset_name, resolution, vae_scaling_factor, batch_size):
    """Precompute latents for a specific resolution and dataset."""
    
    # Create output directories on the main process
    if accelerator.is_main_process:
        dataset_output_dir = Path(output_dir) / dataset_name
        image_latents_dir = dataset_output_dir / "image_latents"
        mask_latents_dir = dataset_output_dir / "mask_latents"
        sketch_latents_dir = dataset_output_dir / "sketch_latents"
        text_dir = dataset_output_dir / "text"
        
        for directory in [image_latents_dir, mask_latents_dir, sketch_latents_dir, text_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    # Wait for the main process to create directories before other processes proceed
    accelerator.wait_for_everyone()
    
    # Re-assign paths for all processes after they have been created
    dataset_output_dir = Path(output_dir) / dataset_name
    image_latents_dir = dataset_output_dir / "image_latents"
    mask_latents_dir = dataset_output_dir / "mask_latents"
    sketch_latents_dir = dataset_output_dir / "sketch_latents"
    text_dir = dataset_output_dir / "text"
    
    latent_dirs = {
        'image': image_latents_dir,
        'mask': mask_latents_dir, 
        'sketch': sketch_latents_dir
    }
    
    # Filter out triplets that already have all latents computed (on main process)
    if accelerator.is_main_process:
        logging.info(f"Checking existing latents for {dataset_name}...")
        remaining_triplets = []
        for triplet in triplets:
            _, _, _, _, base_name = triplet
            if not check_existing_latents(latent_dirs, base_name):
                remaining_triplets.append(triplet)
            else:
                logging.debug(f"Skipping {base_name} - latents already exist")
        
        logging.info(f"Found {len(remaining_triplets)} out of {len(triplets)} triplets that need processing")
        triplets_to_process = remaining_triplets
    else:
        triplets_to_process = triplets  # Placeholder, will be replaced by split_between_processes
    
    # Wait for main process to finish filtering
    accelerator.wait_for_everyone()
    
    # Now all processes will work on the same filtered list
    all_triplets = triplets_to_process  # This will be the same for all processes after main process filtering
    
    # Use split_between_processes to divide the work
    num_triplets = len(all_triplets)
    
    with accelerator.split_between_processes(range(num_triplets)) as local_indices:
        progress_bar = tqdm(
            total=len(local_indices),
            desc=f"Processing {dataset_name} {resolution}x{resolution}",
            disable=not accelerator.is_local_main_process,
        )

        for idx, i in enumerate(local_indices):
            img_path, mask_path, sketch_path, text_path, base_name = all_triplets[i]
            
            try:
                # Check if all latent files already exist (double-check for race conditions)
                if check_existing_latents(latent_dirs, base_name):
                    progress_bar.update(1)
                    continue
                
                # Remove any corrupted latent files if they exist
                remove_corrupted_latent_files(latent_dirs, base_name)
                
                # Get thread-local transforms
                transforms_dict = get_thread_local_transforms(resolution)
                image_transform = transforms_dict['image']
                sketch_transform = transforms_dict['sketch']
                
                # Load and transform all three images with error handling
                try:
                    image = image_transform(ensure_rgb(Image.open(img_path))).unsqueeze(0)
                    mask = image_transform(ensure_rgb(Image.open(mask_path))).unsqueeze(0)
                    sketch = ensure_grayscale_and_convert_to_rgb_tensor(Image.open(sketch_path), sketch_transform).unsqueeze(0)
                except Exception as img_error:
                    logging.error(f"Error loading images for {base_name}: {img_error}")
                    continue
                
                # Verify tensor shapes are consistent
                if not all(t.shape == image.shape for t in [mask, sketch]):
                    logging.error(f"Shape mismatch for {base_name}: image {image.shape}, mask {mask.shape}, sketch {sketch.shape}")
                    continue

                # Batch them together for a single VAE pass
                batch = torch.cat([image, mask, sketch], dim=0).to(accelerator.device, dtype=vae.dtype)

                # Encode to latents in a single forward pass
                with torch.no_grad():
                    latents = vae.encode(batch).latent_dist.sample() * vae_scaling_factor
                
                # Split the latents back
                image_latents, mask_latents, sketch_latents = latents.chunk(3)
                
                # Validate latent shapes
                if any(latent.shape[0] != 1 for latent in [image_latents, mask_latents, sketch_latents]):
                    logging.error(f"Invalid latent shape for {base_name}")
                    continue
                
                # Save latents with temporary files to prevent corruption
                temp_image_latent = image_latents_dir / f"{base_name}.pt.temp"
                temp_mask_latent = mask_latents_dir / f"{base_name}.pt.temp"
                temp_sketch_latent = sketch_latents_dir / f"{base_name}.pt.temp"
                
                torch.save(image_latents.cpu(), temp_image_latent)
                torch.save(mask_latents.cpu(), temp_mask_latent)
                torch.save(sketch_latents.cpu(), temp_sketch_latent)
                
                # Rename temp files to final names (atomic operation)
                (image_latents_dir / f"{base_name}.pt").unlink(missing_ok=True)
                temp_image_latent.rename(image_latents_dir / f"{base_name}.pt")
                
                (mask_latents_dir / f"{base_name}.pt").unlink(missing_ok=True)
                temp_mask_latent.rename(mask_latents_dir / f"{base_name}.pt")
                
                (sketch_latents_dir / f"{base_name}.pt").unlink(missing_ok=True)
                temp_sketch_latent.rename(sketch_latents_dir / f"{base_name}.pt")
                
                # Copy text file (only one process needs to do this, but it's idempotent)
                dest_text_path = text_dir / f"{base_name}.txt"
                if not dest_text_path.exists():
                    shutil.copy2(text_path, dest_text_path)
                
                # Clear memory periodically to prevent memory issues
                if idx % 50 == 0:  # More frequent cleanup
                    del image, mask, sketch, batch, latents, image_latents, mask_latents, sketch_latents
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing {base_name} on process {accelerator.process_index}: {e}")
                # Remove any partial files if error occurred during processing
                for key in ['image', 'mask', 'sketch']:
                    temp_file = latent_dirs[key] / f"{base_name}.pt.temp"
                    if temp_file.exists():
                        temp_file.unlink()
                # Clear memory on error
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            progress_bar.update(1)
        progress_bar.close()

def main():
    parser = argparse.ArgumentParser(description="Precompute latents for images, masks, and sketches with Accelerate.")
    parser.add_argument("--config_path", type=str, default="config_256_unified.yml", help="Path to the training configuration YAML file.")
    parser.add_argument("--output_base_dir", type=str, default="latents", help="Base directory for output latents.")
    parser.add_argument("--resolutions", type=int, nargs='+', default=[256], help="List of resolutions to process.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU for VAE encoding.")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Setup logging, but only on the main process
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"Starting latent precomputation on {accelerator.num_processes} GPUs.")
    
    # Load config on all processes
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load VAE from local path
    # Use bfloat16 for performance if supported
    weight_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if accelerator.is_main_process:
        logging.info("Loading FLUX.1 VAE from local path './VAE'...")
    vae = AutoencoderKL.from_pretrained("./VAE").to(device=accelerator.device, dtype=weight_dtype).eval()
    vae_scaling_factor = vae.config.get("scaling_factor", 0.3611) # From FLUX config
    
    # --- Discover file paths on the main process (assuming all triplets exist) ---
    celeb_triplets = []
    ffhq_triplets = []
    if accelerator.is_main_process:
        logging.info("Processing CelebA-HQ dataset...")
        celeb_triplets = get_file_paths_for_celeba(config, split='train')
        
        logging.info("Processing FFHQ dataset...")
        ffhq_triplets = get_file_paths_for_ffhq(config)
    
    # Wait for main process to finish collecting triplets
    accelerator.wait_for_everyone()
    
    # --- Precompute latents for all resolutions ---
    for resolution in args.resolutions:
        output_dir = f"{args.output_base_dir}_{resolution}"
        if accelerator.is_main_process:
            logging.info(f"--- Starting resolution {resolution}x{resolution} ---")
            logging.info(f"Output directory will be: {output_dir}")
        
        # Process CelebA-HQ
        precompute_latents_for_resolution(
            accelerator, vae, celeb_triplets, output_dir, "CelebA-HQ", resolution, vae_scaling_factor, args.batch_size
        )
        
        # Process FFHQ
        precompute_latents_for_resolution(
            accelerator, vae, ffhq_triplets, output_dir, "FFHQ", resolution, vae_scaling_factor, args.batch_size
        )
    
    if accelerator.is_main_process:
        logging.info("✅ Latent precomputation completed successfully!")

if __name__ == "__main__":
    main()