# sample_final.py
"""
Sample images using the Unified Dual-Stream DiT model.
This script supports generating images from either a mask or a sketch,
specified by a command-line argument.
"""

import argparse
import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

# --- Import the unified DiT model and consolidated utility functions ---
from models.diffusion.model_dual_stream_unified import UnifiedDualStreamDiT

# --- Consolidated Utility Functions ---
def ensure_rgb(img: Image.Image) -> Image.Image:
    """Ensures an image is in RGB mode."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def ensure_grayscale(img: Image.Image) -> Image.Image:
    """Ensures an image is in grayscale mode."""
    if img.mode != "L":
        return img.convert("L")
    return img

# --- Preprocessing Functions ---
def preprocess_mask_for_vae(mask_path: Path, target_resolution: int) -> torch.Tensor:
    """Load and preprocess a mask image for VAE encoding."""
    try:
        mask_image = Image.open(mask_path)
        mask_image = ensure_rgb(mask_image)
        mask_image = mask_image.resize((target_resolution, target_resolution), resample=Image.Resampling.LANCZOS)
        mask_tensor = transforms.ToTensor()(mask_image)  # [C, H, W] in [0.0, 1.0]
        mask_tensor = (mask_tensor - 0.5) * 2.0  # Normalize to [-1, 1]
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, C, H, W]
        return mask_tensor
    except Exception as e:
        logging.error(f"Error processing mask {mask_path}: {e}")
        raise

def preprocess_sketch_for_vae(sketch_path: Path, target_resolution: int) -> torch.Tensor:
    """Load and preprocess a sketch image for VAE encoding."""
    try:
        sketch_image = Image.open(sketch_path)
        sketch_image = ensure_grayscale(sketch_image)
        sketch_image = sketch_image.resize((target_resolution, target_resolution), resample=Image.Resampling.LANCZOS)
        sketch_tensor = transforms.ToTensor()(sketch_image)  # [1, H, W] in [0.0, 1.0]
        sketch_tensor = (sketch_tensor - 0.5) * 2.0  # Normalize to [-1, 1]
        # Repeat single channel to 3 channels to match VAE input
        sketch_tensor = sketch_tensor.repeat(3, 1, 1)  # [3, H, W]
        sketch_tensor = sketch_tensor.unsqueeze(0)  # [1, 3, H, W]
        return sketch_tensor
    except Exception as e:
        logging.error(f"Error processing sketch {sketch_path}: {e}")
        raise

# --- Main Sampling Function ---
def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # --- Device and Precision Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32
        args.mixed_precision = "no"

    logger.info(f"Using device: {device}")
    logger.info(f"Using mixed precision: {args.mixed_precision} ({weight_dtype})")

    # --- Load Configuration ---
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {args.config_path}")
    except Exception as e:
        logger.error(f"Failed to load config from {args.config_path}: {e}")
        raise

    # --- Load Base Models (VAE, Scheduler, Tokenizers, Text Encoders) ---
    model_revision = config['model'].get('revision')
    logger.info("Loading FLUX.1 VAE from local path './VAE'...")
    vae = AutoencoderKL.from_pretrained("./VAE")  # Load from local path
    vae = vae.to(device, dtype=weight_dtype).eval()
    vae_scaling_factor = vae.config.get("scaling_factor", 0.3611)  # FLUX VAE scaling factor

    # Local SD2 Path
    LOCAL_MODEL_PATH = "stable-diffusion-2-1-base"

    logger.info("Loading Scheduler...")
    try:
        scheduler_template = DPMSolverMultistepScheduler.from_pretrained(
            config['model']['base_model_id'], subfolder="scheduler", revision=model_revision
        )
    except Exception as e:
        logger.warning(f"Failed to load scheduler from HF: {e}. Falling back to local path: '{LOCAL_MODEL_PATH}'")
        scheduler_template = DPMSolverMultistepScheduler.from_pretrained(
            LOCAL_MODEL_PATH, subfolder="scheduler"
        )

    logger.info("Loading CLIP Tokenizer and Text Encoder...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['base_model_id'], subfolder="tokenizer", revision=model_revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            config['model']['base_model_id'], subfolder="text_encoder", revision=model_revision
        )
        text_encoder = text_encoder.to(device, dtype=weight_dtype).eval()
    except Exception as e:
        logger.warning(f"Failed to load from HF: {e}. Falling back to local path: '{LOCAL_MODEL_PATH}'...")
        try:
            tokenizer = CLIPTokenizer.from_pretrained(LOCAL_MODEL_PATH, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(LOCAL_MODEL_PATH, subfolder="text_encoder")
            text_encoder = text_encoder.to(device, dtype=weight_dtype).eval()
        except OSError as local_e:
            logger.error(f"FATAL: Could not load model from '{LOCAL_MODEL_PATH}'. Ensure the path is correct and contains 'tokenizer' and 'text_encoder' subfolders. Error: {local_e}")
            raise

    # --- Calculate Latent Resolution ---
    resolution = config['model'].get('resolution', 256)
    vae_scale_factor_int = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_resolution = resolution // vae_scale_factor_int
    logger.info(f"Image resolution: {resolution}x{resolution}")
    logger.info(f"Latent resolution: {latent_resolution}x{latent_resolution}")

    # --- Load DiT Model ---
    logger.info("Initializing Unified Dual-Stream DiT Model...")
    dit_config = config.get('dit', {})

    # --- Read attention configuration from config ---
    attention_config = dit_config.get('attention', {'type': 'full'})
    attention_type = attention_config.get('type', 'full')
    sparse_attn_config = attention_config.get('sparse_config') if attention_type == 'sparse' else None

    if attention_type == 'sparse':
        logger.info(f"Initializing with SPARSE attention. Config: {sparse_attn_config}")
    else:
        logger.info("Initializing with FULL attention.")

    dit_model = UnifiedDualStreamDiT(
        input_size=latent_resolution,
        patch_size=dit_config.get('patch_size', 2),
        in_channels=dit_config.get('in_channels', 8),
        hidden_size=dit_config.get('hidden_size', 1152),
        depth=dit_config.get('depth', 28),
        num_heads=dit_config.get('num_heads', 16),
        mlp_ratio=dit_config.get('mlp_ratio', 4.0),
        text_embed_dim=text_encoder.config.hidden_size,
        learn_sigma=dit_config.get('learn_sigma', False),
        # Pass attention config to model
        attention_type=attention_type,
        sparse_attn_config=sparse_attn_config
    )
    logger.info(f"Unified DiT Model initialized with in_channels={dit_model.in_channels}")

    # --- Load DiT Weights ---
    logger.info(f"Loading DiT weights from {args.weights_path}...")
    try:
        state_dict = load_file(args.weights_path)
        dit_model.load_state_dict(state_dict, strict=True)
        logger.info("DiT weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load DiT weights from {args.weights_path}: {e}")
        raise

    dit_model = dit_model.to(device, dtype=weight_dtype).eval()
    logger.info("DiT model moved to device and set to eval mode.")

    # --- Prepare Inference Context ---
    autocast_dtype = weight_dtype if args.mixed_precision != "no" else torch.float32
    inference_context = torch.autocast(str(device).split(":")[0], dtype=autocast_dtype, enabled=(args.mixed_precision != "no"))

    # --- Create Output Directory ---
    output_root_dir = Path(args.output_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_root_dir}")

    # --- Prepare Prompts and CFG ---
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info(f"Classifier-Free Guidance enabled: {do_classifier_free_guidance} (Scale: {args.guidance_scale})")
    
    with torch.no_grad():
        # --- Positive Prompt ---
        # CLIP embeddings
        clip_text_inputs = tokenizer(
            [args.prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        clip_prompt_text_encoder_output = text_encoder(clip_text_inputs.input_ids.to(device), output_hidden_states=True)
        # Use the penultimate layer hidden states as recommended for DiT
        last_hidden_state_prompt = clip_prompt_text_encoder_output.hidden_states[-2]
        prompt_pooled_embeds = clip_prompt_text_encoder_output.pooler_output.to(dtype=weight_dtype) # Alternative pooling
        # prompt_pooled_embeds = last_hidden_state_prompt.mean(dim=1).to(dtype=weight_dtype) # Mean pooling
        prompt_clip_seq_embeds = last_hidden_state_prompt.to(dtype=weight_dtype)
        
        # --- Negative Prompt ---
        # CLIP embeddings
        clip_uncond_inputs = tokenizer(
            [args.negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        clip_uncond_text_encoder_output = text_encoder(clip_uncond_inputs.input_ids.to(device), output_hidden_states=True)
        last_hidden_state_uncond = clip_uncond_text_encoder_output.hidden_states[-2]
        uncond_pooled_embeds = clip_uncond_text_encoder_output.pooler_output.to(dtype=weight_dtype) # Alternative pooling
        # uncond_pooled_embeds = last_hidden_state_uncond.mean(dim=1).to(dtype=weight_dtype) # Mean pooling
        uncond_clip_seq_embeds = last_hidden_state_uncond.to(dtype=weight_dtype)

        # --- Concatenate for CFG ---
        if do_classifier_free_guidance:
            text_pooled_embeddings = torch.cat([uncond_pooled_embeds, prompt_pooled_embeds])
            clip_seq_embeddings = torch.cat([uncond_clip_seq_embeds, prompt_clip_seq_embeds])
        else:
            text_pooled_embeddings = prompt_pooled_embeds
            clip_seq_embeddings = prompt_clip_seq_embeds

    # --- Preprocess and Encode Conditioning Image based on Modality ---
    conditioning_path = Path(args.conditioning_path)
    modality_str = args.modality.lower()
    
    if modality_str == "mask":
        logger.info(f"Preprocessing mask from {conditioning_path}...")
        conditioning_tensor = preprocess_mask_for_vae(conditioning_path, resolution)
        modality_tensor = torch.tensor(0) # 0 for mask
    elif modality_str == "sketch":
        logger.info(f"Preprocessing sketch from {conditioning_path}...")
        conditioning_tensor = preprocess_sketch_for_vae(conditioning_path, resolution)
        modality_tensor = torch.tensor(1) # 1 for sketch
    else:
        logger.error(f"Invalid modality specified: {args.modality}. Must be 'mask' or 'sketch'.")
        raise ValueError(f"Invalid modality: {args.modality}")

    conditioning_tensor = conditioning_tensor.to(device, dtype=weight_dtype)
    with torch.no_grad():
        conditioning_latents = vae.encode(conditioning_tensor).latent_dist.sample() * vae_scaling_factor
        logger.debug(f"Encoded conditioning latents shape: {conditioning_latents.shape}")

    # --- Main Sampling Loop ---
    logger.info(f"Starting generation of {args.num_samples} samples using '{modality_str}' conditioning...")
    for i in tqdm(range(args.num_samples), desc="Generating Samples"):
        # --- Seed Management ---
        if args.seed is not None:
            local_seed = args.seed + i
        else:
            local_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(local_seed)

        conditioning_basename = conditioning_path.stem
        prompt_slug = "".join(filter(str.isalnum, args.prompt))[:30]
        output_filename = f"{conditioning_basename}_{modality_str}_{prompt_slug}_seed{local_seed}.png"
        output_path = output_root_dir / output_filename
        logger.info(f"Generating sample {i+1}/{args.num_samples} (ID: {conditioning_basename}, Seed: {local_seed})")

        try:
            # --- Initialize Latents (Image Noise) ---
            latent_channels = dit_model.in_channels // 2 # For FLUX: 32 // 2 = 16 channels for image latents
            latents_shape = (1, latent_channels, latent_resolution, latent_resolution)
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=weight_dtype)
            
            # --- Create a fresh scheduler instance for each sample ---
            scheduler = type(scheduler_template).from_config(scheduler_template.config)
            scheduler.set_timesteps(args.num_inference_steps, device=device)
            latents = latents * scheduler.init_noise_sigma
            
            # --- Denoising Loop ---
            with inference_context, torch.no_grad():
                for t in tqdm(scheduler.timesteps, desc=f"Denoising Sample {i+1}", leave=False):
                    # Prepare input for the model
                    latent_model_input_base = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input_scaled = scheduler.scale_model_input(latent_model_input_base, t)
                    t_tensor = t.expand(latent_model_input_scaled.shape[0]).to(device)
                    
                    batch_size_for_model = latent_model_input_scaled.shape[0]
                    conditioning_latents_for_concat = conditioning_latents.repeat(batch_size_for_model, 1, 1, 1)

                    # Concatenate image and conditioning latents
                    concatenated_latents = torch.cat([latent_model_input_scaled, conditioning_latents_for_concat], dim=1)

                    # --- Prepare Modality Tensor for CFG ---
                    # Duplicate the modality tensor to match the batch size for CFG
                    modality_tensor_batched = modality_tensor.repeat(batch_size_for_model).to(device)

                    # Predict the noise using the unified model
                    noise_pred = dit_model(
                        x=concatenated_latents,
                        t=t_tensor,
                        text_pooled_embeddings=text_pooled_embeddings,
                        modality=modality_tensor_batched, # Pass the modality flag
                        clip_seq_embeddings=clip_seq_embeddings
                    )
                    
                    # Handle sigma prediction if model was trained with learn_sigma
                    if hasattr(dit_model, 'learn_sigma') and dit_model.learn_sigma:
                        noise_pred, _ = torch.chunk(noise_pred, 2, dim=1)
                    
                    predicted_image_channels = latents.shape[1]
                    noise_pred = noise_pred[:, :predicted_image_channels, :, :]

                    # Perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Scheduler step
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            # --- Decode Latents to Image ---
            latents_for_decoding = 1 / vae_scaling_factor * latents
            image_decoded = vae.decode(latents_for_decoding.to(vae.dtype)).sample

            # --- Post-process and Save Image ---
            image = (image_decoded / 2 + 0.5).clamp(0, 1)
            generated_image_pil = transforms.ToPILImage()(image[0].cpu().float())
            generated_image_pil.save(output_path)
            logger.info(f"Saved sample {i+1} to {output_path}")

        except Exception as e:
            logger.error(f"Error generating sample for seed {local_seed}: {e}", exc_info=True)
            continue

    logger.info(f"Finished generating all samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using the Unified Dual-Stream DiT model with either mask or sketch conditioning.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve newlines in help
    )
    # Required arguments
    parser.add_argument("--config_path", type=str, required=True, help="Path to the training config YAML file.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the DiT model weights file (.safetensors).")
    # --- NEW: Unified Modality Arguments ---
    parser.add_argument("--modality", type=str, choices=["mask", "sketch"], required=True, help="Type of conditioning to use.")
    parser.add_argument("--conditioning_path", type=str, required=True, help="Path to the conditioning image file (mask or sketch).")
    # --- END OF NEW ARGUMENTS ---
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt for generation.")
    
    # Output and sampling parameters
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Directory for output images.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    
    # Optional arguments
    parser.add_argument("--seed", type=int, default=None, help="Base seed for generation. If None, a random seed is used.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="fp16", help="Use mixed precision.")
    parser.add_argument("--negative_prompt", type=str, default="", help="The negative prompt.")
    
    args = parser.parse_args()
    main(args)

"""
DEFAULT COMMANDS FOR MASK AND SKETCH CONDITIONING:

# Example command for mask conditioning with FLUX VAE:
python sample_final.py \\
  --config_path \"config_256_unified.yml\" \\
  --weights_path \"dit-unified-flux-vae-256/dit_model_weights_final.safetensors\" \\
  --modality \"mask\" \\
  --conditioning_path \"/mnt/e/Multi-Modal Face Generation/Celeb_Dataset/Celeb_Final/test/masks/5.png\" \\
  --prompt \"A photo of a young woman with blonde wavy hair, wearing lipstick and smiling.\" \\
  --output_dir \"generated_samples_mask\" \\
  --num_samples 4 \\
  --guidance_scale 7.5 \\
  --num_inference_steps 50 \\
  --seed 42 \\
  --mixed_precision \"bf16\"

# Example command for sketch conditioning with FLUX VAE:
python sample_final.py \\
  --config_path \"config_256_unified.yml\" \\
  --weights_path \"dit-unified-flux-vae-256/dit_model_weights_final.safetensors\" \\
  --modality \"sketch\" \\
  --conditioning_path \"/mnt/e/Multi-Modal Face Generation/Celeb_Dataset/Celeb_Final/test/sketches/5.png\" \\
  --prompt \"A photo of a young woman with blonde wavy hair, wearing lipstick and smiling.\" \\
  --output_dir \"generated_samples_sketch\" \\
  --num_samples 4 \\
  --guidance_scale 7.5 \\
  --num_inference_steps 50 \\
  --seed 42 \\
  --mixed_precision \"bf16\"

# Example command for 512px model with mask conditioning:
python sample_final.py \\
  --config_path \"config_512_unified.yml\" \\
  --weights_path \"dit-unified-flux-vae-512/dit_model_weights_final.safetensors\" \\
  --modality \"mask\" \\
  --conditioning_path \"/mnt/e/Multi-Modal Face Generation/Celeb_Dataset/Celeb_Final/test/masks/5.png\" \\
  --prompt \"A photo of a young woman with blonde wavy hair, wearing lipstick and smiling.\" \\
  --output_dir \"generated_samples_512_mask\" \\
  --num_samples 4 \\
  --guidance_scale 7.5 \\
  --num_inference_steps 50 \\
  --seed 42 \\
  --mixed_precision \"bf16\"
"""