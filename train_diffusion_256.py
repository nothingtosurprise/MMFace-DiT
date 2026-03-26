# train_256_unified.py
# Training script for Unified Dual-Stream DiT with either mask or sketch conditioning using pre-computed latents or raw images.
# Modified to use CLIP for both global conditioning (via c_embedder) and token-by-token sequential conditioning in dual-stream architecture.
# Includes fixes for EMA model saving and incorporates CORRECT SNR weighting.
# Adapted for unified training from scratch at 256x256 resolution using ConcatDataset for CelebA-HQ and FFHQ.

import argparse
import logging
import math
import os
import shutil
import yaml
from pathlib import Path
import random

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

# Diffusers imports
import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from safetensors.torch import save_file, load_file

# Import the unified model and datasets
from models.diffusion.model_dual_stream_unified import UnifiedDualStreamDiT
from utils.utils_final import PrecomputedUnifiedLatentDataset, UnifiedImageDataset

# --- Version Check ---
check_min_version("0.21.0")
logger = get_logger(__name__, log_level="INFO")

# --- Helper Functions ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Unified Dual-Stream DiT model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/diffusion/config_256_unified.yml",
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume training from. If not provided, will try to find the latest checkpoint in the output directory."
    )
    args = parser.parse_args()
    return args

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    """Helper function to get the full repo name."""
    # Placeholder
    return model_id

def log_validation(): # Placeholder for future validation logic
    """Placeholder for validation logic."""
    logger.info("Running validation... (placeholder)")
    pass

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand for batch broadcasting
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()

    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    checkpoint_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        return None
    
    # Sort by step number to get the latest checkpoint
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]), reverse=True)
    return str(checkpoint_dirs[0])

def load_ema_checkpoint(accelerator, dit_model, ema_model, checkpoint_path, use_ema):
    """Load EMA checkpoint and all other states for resuming training."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the accelerator state (optimizer, scheduler, etc.)
    try:
        accelerator.load_state(checkpoint_path)
        logger.info(f"Successfully loaded accelerator state from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error loading accelerator state: {e}")
        raise e
    
    # If using EMA, load the EMA weights
    if use_ema:
        try:
            ema_weights_path = Path(checkpoint_path) / "dit_model_weights_ema.safetensors"
            if ema_weights_path.exists():
                # Load the EMA weights
                ema_state_dict = load_file(ema_weights_path)
                logger.info(f"Loading EMA weights from {ema_weights_path}")
                
                # Get the unwrapped model
                unwrapped_dit = accelerator.unwrap_model(dit_model)
                
                # Load the EMA weights into the model
                unwrapped_dit.load_state_dict(ema_state_dict, strict=True)
                
                # Update the EMA model shadow parameters to match
                # This ensures that the EMA model starts with the correct averaged weights
                ema_model.model = unwrapped_dit
                
                # Load the entire EMA state if available (includes shadow parameters)
                ema_state_path = Path(checkpoint_path) / "ema_state.pt"
                if ema_state_path.exists():
                    try:
                        ema_state_checkpoint = torch.load(ema_state_path, map_location="cpu")
                        ema_model.load_state_dict(ema_state_checkpoint)
                        # Move the EMA model to the correct device after loading
                        ema_model.to(accelerator.device)
                        logger.info("Loaded full EMA state including shadow parameters")
                    except Exception as e:
                        logger.warning(f"Could not load full EMA state: {e}, falling back to manual initialization")
                        # If we can't load the full state, manually set shadow params to match loaded weights
                        ema_model.load_state_dict(ema_state_dict)
                        # Make sure to move the EMA model to the correct device even in fallback
                        ema_model.to(accelerator.device)
                else:
                    # If no full EMA state exists, manually initialize shadow parameters
                    # Store current EMA weights in shadow params
                    ema_model.store(unwrapped_dit.parameters())
                    # Then restore the original weights so training continues normally
                    ema_model.restore(unwrapped_dit.parameters())
                
                logger.info("Loaded EMA weights for resuming training")
            else:
                logger.warning(f"EMA weights file not found at {ema_weights_path}, using regular model weights")
        except Exception as e:
            logger.error(f"Error loading EMA checkpoint: {e}")
            raise e
    else:
        logger.info("EMA not used, skipping EMA weight loading")
    
    # Extract global_step from checkpoint directory name
    checkpoint_step = int(Path(checkpoint_path).name.split("-")[1])
    return checkpoint_step

# --- Main Training Function ---
def main():
    args = parse_args()
    config_path = args.config_path

    # --- 1. Load Configuration ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Use print for initial logging before accelerator is initialized
        print(f"Configuration loaded successfully from {config_path}")
    except Exception as e:
        print(f"Failed to load configuration file {config_path}: {e}")
        raise

    # --- 2. Basic Setup ---
    output_dir = Path(config['training']['output_dir'])
    logging_dir = Path(output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        log_with=config['logging']['report_to'],
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
        
    # Now we can safely use the logger
    logger.info(f"Configuration loaded successfully from {config_path}")

    # Set seed
    set_seed(config['training']['seed'] + accelerator.process_index)
    logger.info(f"Set seed to {config['training']['seed']} + process index {accelerator.process_index}")

    # Create output directory
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")

    # --- 3. Load Pretrained Components ---
    model_revision = config['model'].get('revision')
    logger.info("Loading CLIP Tokenizer and Text Encoder...")
    from transformers import CLIPTokenizer, CLIPTextModel
    
    LOCAL_MODEL_PATH = "/mnt/e/Multi-Modal Face Generation/0_MMDiT_Advanced/14.Flow_Flux/stable-diffusion-2-1-base"
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['base_model_id'], subfolder="tokenizer", revision=model_revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            config['model']['base_model_id'], subfolder="text_encoder", revision=model_revision
        )
    except Exception as e:
        logger.warning(f"Failed to load from HF: {e}. Falling back to local path: {LOCAL_MODEL_PATH}")
        tokenizer = CLIPTokenizer.from_pretrained(LOCAL_MODEL_PATH, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(LOCAL_MODEL_PATH, subfolder="text_encoder")

    # Determine if we are using precomputed latents or raw images based on config
    use_precomputed_latents = config['conditioning']['method'] == "concat_latent"
    
    # Only load VAE if we're not using precomputed latents
    if not use_precomputed_latents:
        logger.info("Loading VAE...")
        # Assuming AutoencoderKL is imported from diffusers
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            config['model']['vae_id'], revision=model_revision
        )
        vae_scaling_factor = vae.config.scaling_factor
        logger.info(f"VAE scaling factor: {vae_scaling_factor}")
    else:
        logger.info("Using precomputed latents. Skipping VAE initialization to save VRAM.")
        vae = None
        vae_scaling_factor = None

    # --- 4. Initialize Unified Dual-Stream DiT Model ---
    logger.info("Initializing Unified Dual-Stream DiT Model...")
    # Determine if we are using precomputed latents or raw images based on config
    use_precomputed_latents = config['conditioning']['method'] == "concat_latent"
    resolution = config['model']['resolution']
    latent_resolution = resolution // 8 # Assuming VAE downsamples by 8

    # DiT Configuration
    dit_config = config.get('dit', {})
    attention_config = dit_config.get('attention', {})
    attention_type = attention_config.get('type', 'full')
    sparse_attn_config = attention_config.get('sparse_config', None) if attention_type == 'sparse' else None

    # Initialize the unified model
    dit_model = UnifiedDualStreamDiT(
        input_size=latent_resolution,
        patch_size=dit_config.get('patch_size', 2),
        in_channels=dit_config.get('in_channels', 8), # Should be 8 for concat (4 img + 4 conditioning)
        hidden_size=dit_config.get('hidden_size', 1152),
        depth=dit_config.get('depth', 28),
        num_heads=dit_config.get('num_heads', 16),
        mlp_ratio=dit_config.get('mlp_ratio', 4.0),
        text_embed_dim=text_encoder.config.hidden_size,
        learn_sigma=dit_config.get('learn_sigma', False),
        caption_dropout_prob=config['training'].get('text_dropout_prob', 0.1),
        # Pass attention config to model
        attention_type=attention_type,
        sparse_attn_config=sparse_attn_config
    )
    logger.info(f"Initialized Unified Dual-Stream DiT Model with config: {dit_config}")

    # Freeze VAE and Text Encoders
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    dit_model.requires_grad_(True) # DiT is trainable

    # --- 5. Check Trainable Parameters ---
    if accelerator.is_main_process:
        params_to_optimize_check = [p for p in dit_model.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in params_to_optimize_check)
        total_params = sum(p.numel() for p in dit_model.parameters())
        logger.info(f"Total DiT parameters: {total_params / 1e6:.2f} M")
        logger.info(f"Trainable DiT parameters: {num_trainable_params / 1e6:.2f} M ({num_trainable_params/total_params*100:.3f}%)")
        if num_trainable_params == 0:
            logger.error("FATAL: No trainable parameters found in DiT. Training cannot proceed.")
            return

    # --- 6. Hardware Optimizations ---
    if config['training'].get('gradient_checkpointing', False):
        dit_model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for DiT.")
    if config['training'].get('allow_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Enabled TF32 for matmul.")

    # --- 7. Setup Optimizer ---
    if config['training'].get('use_8bit_adam', False):
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.error("use_8bit_adam=True requires bitsandbytes. Install with: pip install bitsandbytes")
            raise
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer.")

    params_to_optimize = list(filter(lambda p: p.requires_grad, dit_model.parameters()))
    if not params_to_optimize:
        logger.error("FATAL: Optimizer received no parameters to optimize. Check DiT setup.")
        return

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['adam_weight_decay'],
        eps=config['training']['adam_epsilon'],
    )

    # --- 8. Create Dataset and DataLoader ---
    logger.info("Creating dataset and dataloader...")
    if use_precomputed_latents:
        train_dataset = PrecomputedUnifiedLatentDataset(
            latents_root=config['data']['latents_root'],
            tokenizer=tokenizer,
            text_dropout_prob=config['training'].get('text_dropout_prob', 0.0),
            mask_dropout_prob=config['training'].get('mask_dropout_prob', 0.0),
            sketch_dropout_prob=config['training'].get('sketch_dropout_prob', 0.0),
        )
        logger.info(f"Loaded pre-computed unified latent dataset with {len(train_dataset)} samples.")
    else: # Use raw images - Combine CelebA-HQ and FFHQ
        try:
            # --- Load CelebA-HQ Dataset ---
            logger.info("Loading CelebA-HQ dataset...")
            train_dataset_celeb = UnifiedImageDataset(
                data_root=config['data']['data_root'],
                text_root=config['data']['text_root'],
                tokenizer=tokenizer,
                resolution=config['data']['resolution'],
                image_folder=config['data'].get('image_folder', 'images'),
                mask_folder=config['data'].get('mask_folder', 'masks'),
                sketch_folder=config['data'].get('sketch_folder', 'sketches'),
                mask_suffix=config['data'].get('mask_suffix', '.png'),
                sketch_suffix=config['data'].get('sketch_suffix', '.png'),
                text_suffix=config['data'].get('text_suffix', '.txt'),
                text_dropout_prob=config['training'].get('text_dropout_prob', 0.0),
                mask_dropout_prob=config['training'].get('mask_dropout_prob', 0.0),
                sketch_dropout_prob=config['training'].get('sketch_dropout_prob', 0.0),
                dataset_type='celeba',
                # mask_data_root and sketch_data_root are not used for CelebA-HQ
            )
            logger.info(f"Loaded CelebA-HQ dataset with {len(train_dataset_celeb)} samples.")

            # --- Load FFHQ Dataset ---
            logger.info("Loading FFHQ dataset...")
            train_dataset_ffhq = UnifiedImageDataset(
                data_root=config['ffhq_data']['data_root'],
                text_root=config['ffhq_data']['text_root'],
                tokenizer=tokenizer,
                resolution=config['ffhq_data']['resolution'],
                # image_folder, mask_folder, sketch_folder are not used for FFHQ
                mask_suffix=config['data'].get('mask_suffix', '.png'), # Use global config suffix
                sketch_suffix=config['data'].get('sketch_suffix', '.png'), # Use global config suffix
                text_suffix=config['data'].get('text_suffix', '.txt'), # Use global config suffix
                text_dropout_prob=config['training'].get('text_dropout_prob', 0.0),
                mask_dropout_prob=config['training'].get('mask_dropout_prob', 0.0),
                sketch_dropout_prob=config['training'].get('sketch_dropout_prob', 0.0),
                dataset_type='ffhq',
                mask_data_root=config['ffhq_data'].get('mask_data_root'),
                sketch_data_root=config['ffhq_data'].get('sketch_data_root'),
            )
            logger.info(f"Loaded FFHQ dataset with {len(train_dataset_ffhq)} samples.")

            # --- Combine Datasets ---
            train_dataset = torch.utils.data.ConcatDataset([train_dataset_celeb, train_dataset_ffhq])
            logger.info(f"Combined dataset created. Total samples: {len(train_dataset)} "
                        f"(CelebA-HQ: {len(train_dataset_celeb)}, FFHQ: {len(train_dataset_ffhq)})")
        except FileNotFoundError as e:
            logger.error(f"Dataset creation failed: {e}. Check dataset paths in config.")
            return
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}", exc_info=True)
            raise

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['training'].get('train_batch_size', 1),
        num_workers=config['training'].get('dataloader_num_workers', 4),
        pin_memory=True,
    )

    # --- 9. Setup Noise Scheduler ---
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(config['model']['base_model_id'], subfolder="scheduler")
    except Exception as e:
        logger.warning(f"Failed to load scheduler from HF. Falling back to local path.")
        LOCAL_MODEL_PATH = "/mnt/e/Multi-Modal Face Generation/0_MMDiT_Advanced/14.Flow_Flux/stable-diffusion-2-1-base"
        noise_scheduler = DDPMScheduler.from_pretrained(LOCAL_MODEL_PATH, subfolder="scheduler")
    
    # --- 10. Prepare Everything with Accelerator (except lr_scheduler which needs to be created first) ---
    logger.info("Preparing model, optimizer, and dataloader with Accelerator...")
    dit_model, optimizer, train_dataloader = accelerator.prepare(
        dit_model, optimizer, train_dataloader
    )
    
    # Calculate the number of update steps per epoch after the dataloader has been prepared/sharded
    # After accelerator.prepare, train_dataloader is sharded for each process
    num_update_steps_per_epoch = len(train_dataloader)
    
    # Calculate the total number of training steps based on the sharded dataloader
    if config['training'].get('max_train_steps') is None:
        max_train_steps = config['training']['num_train_epochs'] * num_update_steps_per_epoch
        num_train_epochs = config['training']['num_train_epochs']
    else:
        max_train_steps = config['training']['max_train_steps']
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        
    # Setup LR scheduler after calculating max_train_steps
    lr_scheduler = get_scheduler(
        config['training'].get('lr_scheduler', 'constant'),
        optimizer=optimizer,
        num_warmup_steps=config['training'].get('lr_warmup_steps', 0) * config['training']['gradient_accumulation_steps'],
        num_training_steps=max_train_steps * config['training']['gradient_accumulation_steps'],
    )
    
    # Now prepare the lr_scheduler separately
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Move VAE and Text Encoder to device and cast to weight_dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Ensure Text Encoders are on the correct device and dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not use_precomputed_latents and vae is not None:
        # Only move VAE to device if we're not using precomputed latents
        vae.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"Ensured VAE is on device {accelerator.device} with dtype {weight_dtype}")
    logger.info(f"Ensured Text Encoders are on device {accelerator.device} with dtype {weight_dtype}")

    # --- 11. Initialize EMA Model ---
    use_ema = config['training'].get('use_ema', False)
    ema_model = None
    if use_ema:
        # Assuming EMAModel is available, e.g., from diffusers
        from diffusers.training_utils import EMAModel
        ema_model = EMAModel(
            accelerator.unwrap_model(dit_model).parameters(),
            decay=config['training'].get('ema_decay', 0.9999),
            model_cls=UnifiedDualStreamDiT,
            model_config=dit_model.config if hasattr(dit_model, 'config') else None,
        )
        ema_model.to(accelerator.device)
        logger.info(f"Initialized EMA model with decay {config['training'].get('ema_decay')} and moved to {accelerator.device}")

    # --- 12. Setup Logging and Tracking ---
    if accelerator.is_main_process:
        try:
            tracker_config = dict(config)
            project_name = Path(output_dir).name
            accelerator.init_trackers(project_name, config=tracker_config)
            logger.info(f"Initialized trackers ({config['logging'].get('report_to')}) with project name '{project_name}'.")
        except Exception as e:
            logger.warning(f"Failed to initialize trackers: {e}")

    # Calculate total batch size after accelerator is prepared
    total_batch_size = config['training']['train_batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']

    # --- 13. Resume from Checkpoint (if applicable) ---
    # Check if resuming from a checkpoint
    first_epoch = 0
    global_step = 0
    initial_global_step = 0
    resume_step = 0
    
    # Check for existing checkpoints
    checkpoint_to_resume = None
    if args.resume_from_checkpoint:
        checkpoint_to_resume = args.resume_from_checkpoint
        logger.info(f"Resuming from explicitly specified checkpoint: {checkpoint_to_resume}")
    else:
        latest_checkpoint_dir = get_latest_checkpoint(output_dir)
        if latest_checkpoint_dir is not None:
            checkpoint_to_resume = latest_checkpoint_dir
            logger.info(f"Found existing checkpoint: {latest_checkpoint_dir}")
    
    if checkpoint_to_resume is not None:
        # Load the checkpoint
        resume_step = load_ema_checkpoint(accelerator, dit_model, ema_model, checkpoint_to_resume, use_ema)
        global_step = resume_step
        initial_global_step = resume_step
        first_epoch = global_step // num_update_steps_per_epoch
        
        logger.info(f"Resuming from epoch {first_epoch}, global step {global_step}")
        logger.info(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
    else:
        logger.info("No existing checkpoint found. Starting training from scratch.")

    # Calculate SNR (Signal-to-Noise Ratio) for weighting
    snr_gamma = config['training'].get('snr_gamma', None)
    if snr_gamma is not None and snr_gamma > 0:
        logger.info(f"Using min-SNR weighting with gamma={snr_gamma}")
        snr_weighting = True
    else:
        snr_weighting = False
        logger.info("Not using min-SNR weighting.")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config['training']['train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    if initial_global_step > 0:
        logger.info(f"  Resuming from step {initial_global_step}")
        logger.info(f"  Will first complete epoch {first_epoch}")

    # Loss Smoothing (for logging)
    loss_ema = None
    loss_ema_decay = 0.99

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # --- 14. Main Training Loop ---
    for epoch in range(first_epoch, num_train_epochs):
        dit_model.train()
        train_loss_accum = 0.0
        epoch_ema_loss_sum = 0.0
        steps_in_epoch = 0

        # Calculate how many steps to skip if resuming mid-epoch
        resume_step_in_epoch = 0
        if epoch == first_epoch and resume_step > 0:
            resume_step_in_epoch = resume_step % num_update_steps_per_epoch
            logger.info(f"Resuming epoch {epoch + 1} from step {resume_step_in_epoch}")

        epoch_progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=False
        )

        for step, batch in enumerate(epoch_progress_bar):
            # Skip steps if resuming mid-epoch
            if epoch == first_epoch and step < resume_step_in_epoch:
                # Update the progress bar to reflect skipped steps
                if step % config['training']['gradient_accumulation_steps'] == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(dit_model):
                try:
                    if use_precomputed_latents:
                        # Get batch data (pre-computed latents)
                        image_latents = batch["image_latent"].to(dtype=weight_dtype)
                        conditioning_latents = batch["conditioning_latent"].to(dtype=weight_dtype)
                        modality_flags = batch["modality"].to(device=accelerator.device) # 0 for mask, 1 for sketch
                        # Handle CLIP input_ids
                        clip_input_ids = batch["clip_input_ids"] if "clip_input_ids" in batch else batch["input_ids"]
                    else:
                        # Get batch data (raw images)
                        images = batch["image"].to(dtype=weight_dtype)
                        conditioning_images = batch["conditioning_image"].to(dtype=weight_dtype) # Already normalized by dataset
                        modality_flags = batch["modality"].to(device=accelerator.device) # 0 for mask, 1 for sketch
                        # Handle CLIP input_ids
                        clip_input_ids = batch["clip_input_ids"] if "clip_input_ids" in batch else batch["input_ids"]

                    # --- Encode Text Embeddings ---
                    with torch.no_grad():
                        # Text Embedding Handling using CLIP
                        clip_text_encoder_output = text_encoder(
                            clip_input_ids.to(accelerator.device),
                            output_hidden_states=True
                        )
                        # Use CLIP pooled embeddings for global conditioning (adaLN-Zero)
                        # The model's internal c_embedder will handle projection and dropout
                        pooled_text_embeddings = clip_text_encoder_output.pooler_output
                        # Use CLIP sequence embeddings for the text stream
                        clip_sequence_text_embeddings = clip_text_encoder_output.hidden_states[-2].to(dtype=weight_dtype)

                    # --- Encode Images to Latents (if not using precomputed) ---
                    if not use_precomputed_latents and vae is not None:
                        with torch.no_grad():
                            latents = vae.encode(images).latent_dist.sample() * vae_scaling_factor
                            conditioning_latents = vae.encode(conditioning_images).latent_dist.sample() * vae_scaling_factor
                            latents = latents.to(device=accelerator.device, dtype=weight_dtype)
                            conditioning_latents = conditioning_latents.to(device=accelerator.device, dtype=weight_dtype)
                    else:
                        latents = image_latents

                    # --- Noise Addition and Concatenation ---
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_image_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Concatenate image latents and conditioning latents along the channel dimension
                    concatenated_latents = torch.cat([noisy_image_latents, conditioning_latents], dim=1)
                    concatenated_latents = concatenated_latents.to(dtype=noisy_image_latents.dtype)

                    # --- DiT Model Prediction ---
                    # The model's internal c_embedder will process pooled_text_embeddings
                    model_pred = accelerator.unwrap_model(dit_model)(
                        x=concatenated_latents, # Pass concatenated tensor
                        t=timesteps,
                        text_pooled_embeddings=pooled_text_embeddings.to(dtype=concatenated_latents.dtype),
                        modality=modality_flags, # Pass the modality tensor (0 for mask, 1 for sketch)
                        clip_seq_embeddings=clip_sequence_text_embeddings
                    )

                    # --- Compute Loss with CORRECT SNR weighting ---
                    # 1. Determine Target (for standard diffusion models, this is usually noise)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        msg = f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                        logger.error(msg)
                        raise ValueError(msg)

                    # 2. Compute MSE loss per sample in the batch (no reduction yet)
                    mse_loss_per_batch_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    mse_loss_per_batch_element = mse_loss_per_batch_element.mean(dim=[1, 2, 3]) # Shape: [B]

                    if snr_weighting:
                        # 3. Compute SNR weights (CORRECT implementation)
                        snr = compute_snr(noise_scheduler, timesteps)
                        # The correct formula: min(SNR, gamma) / SNR
                        # Adding a small epsilon for numerical stability
                        snr_weights = torch.minimum(snr, torch.full_like(snr, float(snr_gamma))) / (snr + 1e-8)

                        # 4. Weight the MSE loss for each batch element
                        weighted_loss_per_batch_element = mse_loss_per_batch_element * snr_weights

                        # 5. Final loss is the mean over the batch
                        loss = weighted_loss_per_batch_element.mean()

                        if accelerator.is_main_process and global_step % 100 == 0:
                            avg_snr_weight = snr_weights.mean().item()
                            logger.debug(f"Step {global_step}: Avg SNR Weight = {avg_snr_weight:.4f}")
                    else:
                        # Standard MSE loss (no weighting)
                        loss = mse_loss_per_batch_element.mean()
                        if accelerator.is_main_process and global_step % 100 == 0:
                            logger.debug(f"Step {global_step}: SNR Weighting Disabled. Using standard MSE.")

                    train_loss_accum += loss.detach().item()

                    # --- Backward Pass and Optimizer Step ---
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if params_to_optimize:
                            accelerator.clip_grad_norm_(params_to_optimize, config['training'].get('max_grad_norm', 1.0))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=config['training'].get('optimizer_set_to_none', True))

                    # --- Update EMA ---
                    if use_ema:
                        unwrapped_dit = accelerator.unwrap_model(dit_model)
                        try:
                            ema_model.step(unwrapped_dit.parameters())
                        except RuntimeError as e:
                            if "Expected all tensors to be on the same device" in str(e):
                                logger.error("Device mismatch error in EMA step. Debugging parameter devices:")
                                for i, (name, param) in enumerate(unwrapped_dit.named_parameters()):
                                    logger.error(f"  DiT Param {i} ({name}): {param.device}")
                            # Note: ema_model.shadow_params might not be directly accessible this way
                            raise e

                except Exception as e:
                    logger.error(f"Error in training step: {e}", exc_info=True)
                    raise e

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # Update EMA loss for logging
                if loss_ema is None:
                    loss_ema = train_loss_accum
                else:
                    loss_ema = loss_ema * loss_ema_decay + train_loss_accum * (1 - loss_ema_decay)
                accelerator.log({"train_loss": loss_ema}, step=global_step)
                # Accumulate for epoch-level averaging
                epoch_ema_loss_sum += loss_ema
                steps_in_epoch += 1
                train_loss_accum = 0.0

                # --- Save Checkpoint ---
                if global_step % config['logging']['checkpointing_steps'] == 0:
                    if accelerator.is_main_process:
                        save_path_dir = Path(output_dir, f"checkpoint-{global_step}")
                        save_path_dir.mkdir(parents=True, exist_ok=True)
                        
                        # --- CORRECTED EMA SAVING ---
                        if use_ema:
                            try:
                                # 1. Get the unwrapped model that is being trained.
                                unwrapped_dit = accelerator.unwrap_model(dit_model)
                                # 2. Store the original, non-averaged parameters from the current training step.
                                ema_model.store(unwrapped_dit.parameters())
                                # 3. Copy the averaged (EMA) weights into the model.
                                ema_model.copy_to(unwrapped_dit.parameters())
                                # 4. Save the stateDict of the model. It now holds the EMA weights.
                                ema_weights_name = "dit_model_weights_ema.safetensors"
                                ema_weights_path = save_path_dir / ema_weights_name
                                save_file(unwrapped_dit.state_dict(), ema_weights_path)
                                logger.info(f"Saved EMA weights to {ema_weights_path}")
                                
                                # 5. Save the full EMA state (including shadow parameters)
                                ema_state_path = save_path_dir / "ema_state.pt"
                                torch.save(ema_model.state_dict(), ema_state_path)
                                logger.info(f"Saved full EMA state to {ema_state_path}")
                                
                                # 6. Restore the original non-averaged weights back to the model.
                                ema_model.restore(unwrapped_dit.parameters())
                                logger.info("Restored original model weights after saving EMA checkpoint.")
                            except Exception as e:
                                logger.error(f"Error saving EMA checkpoint at step {global_step}: {e}", exc_info=True)
                        
                        # Save accelerator state (optimizer, scheduler, etc.)
                        try:
                            accelerator.save_state(str(save_path_dir))
                            logger.info(f"Saved accelerator state to {save_path_dir}")
                        except Exception as e:
                            logger.error(f"Error saving checkpoint at step {global_step}: {e}", exc_info=True)

            logs = {"step_loss": loss_ema, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            epoch_progress_bar.set_postfix(**logs)
            
            # Break inner loop if max steps reached
            if global_step >= max_train_steps:
                break

        # Break outer loop if max steps reached
        if global_step >= max_train_steps:
            break

        # Log average epoch loss and save to file
        if accelerator.is_main_process and steps_in_epoch > 0:
            avg_epoch_ema_loss = epoch_ema_loss_sum / steps_in_epoch
            logger.info(f"Epoch {epoch + 1}/{num_train_epochs} Summary| Average EMA Loss: {avg_epoch_ema_loss:.5f}")
            
            # Save to loss.txt
            loss_file_path = Path(output_dir) / "loss.txt"
            with open(loss_file_path, "a") as f:
                f.write(f"Epoch {epoch + 1}: {avg_epoch_ema_loss:.5f}\n")

        # --- Run Validation (Placeholder) ---
        if accelerator.is_main_process:
            if global_step % config['logging']['validation_steps'] == 0:
                log_validation()

    # --- Save Final Model ---
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        dit_model_final = accelerator.unwrap_model(dit_model)
        try:
            final_save_path_dir = Path(output_dir)
            final_save_path_dir.mkdir(parents=True, exist_ok=True)
            final_weights_name = "dit_model_weights_final.safetensors"
            final_weights_path = final_save_path_dir / final_weights_name
            save_file(dit_model_final.state_dict(), final_weights_path)
            logger.info(f"Saved final DiT weights to {final_weights_path}")

            # --- CORRECTED FINAL EMA SAVING ---
            if use_ema:
                try:
                    # Get the unwrapped model that was being trained.
                    unwrapped_dit = accelerator.unwrap_model(dit_model)
                    # Apply the same store -> copy_to -> save logic as with checkpoints.
                    # Note: For the final model, restoring the original weights isn't strictly
                    # necessary since training is over, but it's good practice.
                    # 1. Store the final non-averaged parameters.
                    ema_model.store(unwrapped_dit.parameters())
                    # 2. Copy the final averaged (EMA) weights into the model.
                    ema_model.copy_to(unwrapped_dit.parameters())
                    # 3. Save the stateDict of the model. It now holds the final EMA weights.
                    ema_final_save_path = Path(final_save_path_dir, "ema_model_final")
                    ema_final_save_path.mkdir(parents=True, exist_ok=True)
                    ema_final_weights_name = "dit_model_weights_ema_final.safetensors"
                    ema_final_weights_path = ema_final_save_path / ema_final_weights_name
                    save_file(unwrapped_dit.state_dict(), ema_final_weights_path)
                    logger.info(f"Saved final EMA weights to {ema_final_weights_path}")
                    
                    # 4. Save the full final EMA state
                    ema_final_state_path = ema_final_save_path / "ema_state_final.pt"
                    torch.save(ema_model.state_dict(), ema_final_state_path)
                    logger.info(f"Saved final EMA state to {ema_final_state_path}")
                    
                    # 5. Restore the original non-averaged weights back to the model (optional here).
                    ema_model.restore(unwrapped_dit.parameters())
                    logger.info("Restored original model weights after saving final EMA checkpoint.")
                    
                    # Save EMA model config if needed
                    # ema_model.save_pretrained(ema_final_save_path) # If EMAModel has this method
                except Exception as e:
                    logger.error(f"Error saving final EMA model weights: {e}", exc_info=True)

            # Save model card or other metadata if needed
            # save_model_card(...) 

        except Exception as e:
            logger.error(f"Error saving final model weights: {e}", exc_info=True)

    accelerator.end_training()
    logger.info("Training finished successfully.")

# --- Entry Point ---
if __name__ == "__main__":
    main()