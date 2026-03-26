# train_512_unified_optimized.py
# 512px training with NaN protection and forced learning rate
import argparse
import logging
import math
import yaml
from pathlib import Path
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from safetensors.torch import save_file, load_file
from models.flow.model_dual_stream_unified import UnifiedDualStreamDiT
from utils.utils_final import PrecomputedUnifiedLatentDataset

check_min_version("0.21.0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train 512px with NaN Protection")
    parser.add_argument("--config_path", type=str, default="config_512_unified_rfm_nan_fix.yml")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--force_lr", type=float, default=None, help="Force specific learning rate, overriding checkpoint")
    return parser.parse_args()

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint by step number."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoint_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]), reverse=True)
    return str(checkpoint_dirs[0])

def load_checkpoint_with_model_only(model, checkpoint_path, device="cpu"):
    """Load only model weights, skip optimizer/scheduler."""
    try:
        global_step = int(Path(checkpoint_path).name.split("-")[1])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}")

    logger.info(f"Loading MODEL WEIGHTS ONLY from '{checkpoint_path}' at step {global_step}")

    # Load EMA weights if available, otherwise regular weights
    ema_weights_path = Path(checkpoint_path) / "dit_model_weights_ema.safetensors"
    model_weights_path = Path(checkpoint_path) / "model.safetensors"

    if ema_weights_path.exists():
        weights_path = ema_weights_path
        logger.info("Loading from EMA weights")
    elif model_weights_path.exists():
        weights_path = model_weights_path
        logger.info("Loading from model weights")
    else:
        raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    logger.info("✓ Model weights loaded successfully")
    return global_step

def load_ema_checkpoint(accelerator, ema_model, checkpoint_path):
    """Load full training state including EMA."""
    try:
        global_step = int(Path(checkpoint_path).name.split("-")[1])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}")

    logger.info(f"Loading checkpoint from '{checkpoint_path}' at step {global_step}")
    accelerator.load_state(checkpoint_path)
    logger.info("✓ Loaded optimizer and scheduler state")

    if ema_model is not None:
        ema_weights_path = Path(checkpoint_path) / "dit_model_weights_ema.safetensors"
        ema_state_path = Path(checkpoint_path) / "ema_state.pt"

        if ema_weights_path.exists() and ema_state_path.exists():
            ema_state_dict = torch.load(ema_state_path, map_location="cpu")
            ema_model.load_state_dict(ema_state_dict)
            ema_model.to(accelerator.device)

            unwrapped_model = accelerator.unwrap_model(accelerator._models[0])
            ema_weights = load_file(ema_weights_path, device="cpu")
            unwrapped_model.load_state_dict(ema_weights)
            logger.info("✓ Loaded EMA weights and state")
        else:
            logger.warning("EMA files not found, skipping EMA restoration")
    return global_step

def save_checkpoint(accelerator, dit_model, ema_model, output_dir, global_step, use_ema):
    """Save checkpoint with proper EMA handling."""
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(save_path))

    if use_ema and ema_model is not None:
        unwrapped_model = accelerator.unwrap_model(dit_model)
        ema_model.store(unwrapped_model.parameters())
        ema_model.copy_to(unwrapped_model.parameters())
        save_file(unwrapped_model.state_dict(), save_path / "dit_model_weights_ema.safetensors")
        torch.save(ema_model.state_dict(), save_path / "ema_state.pt")
        ema_model.restore(unwrapped_model.parameters())

    return save_path

def check_for_nan(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        return f"NaN detected in {name}"
    if torch.isinf(tensor).any():
        return f"Inf detected in {name}"
    return None

def main():
    args = parse_args()

    # === 1. Load Configuration ===
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['training']['output_dir'])
    logging_dir = output_dir / "logs"

    # === 2. Initialize Accelerator ===
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        log_with=config['logging']['report_to'],
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir)),
        kwargs_handlers=[ddp_kwargs],
        cpu=False,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    set_seed(config['training']['seed'] + accelerator.process_index)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

    # === 3. Load Components ===
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained(
        config['model']['base_model_id'],
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config['model']['base_model_id'],
        subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # === 4. Initialize Model ===
    resolution = config['model']['resolution']
    latent_resolution = resolution // 8
    dit_config = config.get('dit', {})
    dit_model = UnifiedDualStreamDiT(
        input_size=latent_resolution,
        patch_size=dit_config.get('patch_size', 2),
        in_channels=dit_config.get('in_channels', 32),
        hidden_size=dit_config.get('hidden_size', 1152),
        depth=dit_config.get('depth', 28),
        num_heads=dit_config.get('num_heads', 16),
        mlp_ratio=dit_config.get('mlp_ratio', 4.0),
        text_embed_dim=text_encoder.config.hidden_size,
        learn_sigma=dit_config.get('learn_sigma', False),
        caption_dropout_prob=config['training'].get('text_dropout_prob', 0.1),
        attention_type=dit_config.get('attention', {}).get('type', 'full'),
        sparse_attn_config=dit_config.get('attention', {}).get('sparse_config', None)
    )
    dit_model.requires_grad_(True)

    if config['training'].get('gradient_checkpointing', False):
        dit_model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")

    if config['training'].get('allow_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ TF32 enabled")

    torch.backends.cudnn.benchmark = True

    # === 5. Create Dataset ===
    train_dataset = PrecomputedUnifiedLatentDataset(
        latents_root=config['data']['latents_root'],
        tokenizer=tokenizer,
        text_dropout_prob=config['training'].get('text_dropout_prob', 0.0),
        mask_dropout_prob=config['training'].get('mask_dropout_prob', 0.0),
        sketch_dropout_prob=config['training'].get('sketch_dropout_prob', 0.0),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['training'].get('train_batch_size', 1),
        num_workers=config['training'].get('dataloader_num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    logger.info(f"✓ Dataset loaded: {len(train_dataset)} samples")

    # === 6. Determine Training Scenario ===
    checkpoint_to_resume = args.resume_from_checkpoint or config['training'].get('resume_from_checkpoint')
    if checkpoint_to_resume == "latest":
        checkpoint_to_resume = get_latest_checkpoint(output_dir)

    force_new_lr = args.force_lr or config['training'].get('force_lr', None)

    # === 7. Calculate Training Steps ===
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['training']['gradient_accumulation_steps'])
    num_train_epochs = config['training']['num_train_epochs']
    max_train_steps = config['training'].get('max_train_steps') or (num_train_epochs * num_update_steps_per_epoch)

    # === 8. Initialize EMA ===
    use_ema = config['training'].get('use_ema', False)
    ema_model = None
    if use_ema:
        from diffusers.training_utils import EMAModel
        ema_model = EMAModel(dit_model.parameters(), decay=config['training'].get('ema_decay', 0.9999))

    # === 9. Setup Optimizer ===
    optimizer_class = torch.optim.AdamW
    if config['training'].get('use_8bit_adam', False):
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("✓ Using 8-bit AdamW")
        except ImportError:
            logger.warning("bitsandbytes not found, using standard AdamW")

    # Determine learning rate (force_lr takes priority)
    if force_new_lr is not None:
        learning_rate = force_new_lr
        logger.info(f"🔧 FORCING learning rate to: {learning_rate}")
    else:
        learning_rate = config['training']['learning_rate']

    optimizer_kwargs = {
        'lr': learning_rate,
        'betas': (config['training']['adam_beta1'], config['training']['adam_beta2']),
        'weight_decay': config['training']['adam_weight_decay'],
        'eps': config['training']['adam_epsilon'],
    }
    if not config['training'].get('use_8bit_adam', False):
        optimizer_kwargs['fused'] = True

    optimizer = optimizer_class(dit_model.parameters(), **optimizer_kwargs)

    lr_scheduler = get_scheduler(
        config['training'].get('lr_scheduler', 'constant'),
        optimizer=optimizer,
        num_warmup_steps=config['training'].get('lr_warmup_steps', 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # === 10. Load Checkpoint Based on Scenario ===
    global_step = 0
    first_epoch = 0

    if checkpoint_to_resume and force_new_lr is not None:
        logger.info("\n" + "="*70)
        logger.info("🔧 RESUMING WITH FORCED LEARNING RATE")
        logger.info("="*70)
        logger.info(f"Checkpoint: {checkpoint_to_resume}")
        logger.info(f"Forced LR: {force_new_lr}")
        logger.info("="*70)

        # Load ONLY model weights, skip optimizer/scheduler
        start_step = load_checkpoint_with_model_only(dit_model, checkpoint_to_resume, device="cpu")

        # Now prepare with new optimizer/scheduler
        dit_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            dit_model, optimizer, train_dataloader, lr_scheduler
        )

        if use_ema:
            ema_model.to(accelerator.device)
            # Initialize EMA from current model weights
            unwrapped_model = accelerator.unwrap_model(dit_model)
            ema_model.load_state_dict(unwrapped_model.state_dict())
            logger.info("✓ EMA re-initialized from current model")

        global_step = start_step
        first_epoch = global_step // num_update_steps_per_epoch
        logger.info(f"✓ Model loaded from step {start_step}")
        logger.info(f"✓ New optimizer created with LR={learning_rate}")
        logger.info(f"✓ Continuing from epoch {first_epoch + 1}, step {global_step}")

    elif checkpoint_to_resume:
        logger.info("\n" + "="*70)
        logger.info("🔄 RESUMING NORMAL TRAINING")
        logger.info("="*70)
        dit_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            dit_model, optimizer, train_dataloader, lr_scheduler
        )

        if use_ema:
            ema_model.to(accelerator.device)

        global_step = load_ema_checkpoint(accelerator, ema_model, checkpoint_to_resume)
        first_epoch = global_step // num_update_steps_per_epoch
        logger.info(f"✓ Resumed from epoch {first_epoch + 1}, step {global_step}")

    else:
        logger.info("\n" + "="*70)
        logger.info("🆕 TRAINING FROM SCRATCH")
        logger.info("="*70)
        dit_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            dit_model, optimizer, train_dataloader, lr_scheduler
        )

        if use_ema:
            ema_model.to(accelerator.device)

    # === 11. Setup Dtype ===
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # === 12. Initialize Tracking ===
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(Path(output_dir).name, config=config)
        except:
            logger.warning("Failed to initialize trackers")

        loss_log_path = output_dir / "training_loss.txt"
        nan_log_path = output_dir / "nan_detections.txt"
        with open(loss_log_path, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Starting from step: {global_step}\n")
            f.write("-" * 70 + "\n")

    total_batch_size = config['training']['train_batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']

    logger.info("\n" + "="*70)
    logger.info("📊 TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"  Dataset size: {len(train_dataset):,} samples")
    logger.info(f"  Starting epoch: {first_epoch + 1}")
    logger.info(f"  Starting step: {global_step}")
    logger.info(f"  Total epochs: {num_train_epochs}")
    logger.info(f"  Total steps: {max_train_steps:,}")
    logger.info(f"  Batch size per device: {config['training']['train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {total_batch_size}")
    logger.info(f"  Learning rate: {learning_rate} {'(FORCED)' if force_new_lr else ''}")
    logger.info(f"  LR Scheduler: {config['training'].get('lr_scheduler', 'constant')}")
    logger.info(f"  Max grad norm: {config['training']['max_grad_norm']}")
    logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"  EMA enabled: {use_ema}")
    logger.info("="*70 + "\n")

    # === 13. Training Loop with NaN Protection ===
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    log_interval = 50
    loss_accumulator = 0.0
    loss_count = 0
    nan_count = 0
    consecutive_nans = 0

    for epoch in range(first_epoch, num_train_epochs):
        dit_model.train()
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        resume_step = 0
        if epoch == first_epoch and global_step > 0:
            resume_step = global_step % num_update_steps_per_epoch

        epoch_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=False
        )

        for step, batch in epoch_bar:
            if epoch == first_epoch and step < resume_step * config['training']['gradient_accumulation_steps']:
                continue

            with accelerator.accumulate(dit_model):
                # === Data Preparation ===
                image_latents = batch["image_latent"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                conditioning_latents = batch["conditioning_latent"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                clip_input_ids = batch["clip_input_ids"].to(device=accelerator.device, non_blocking=True)
                modality_flags = batch["modality"].to(device=accelerator.device, non_blocking=True)

                # NaN Check: Input data
                nan_error = check_for_nan(image_latents, "image_latents")
                if nan_error:
                    logger.error(f"⚠️ {nan_error} - Skipping batch")
                    continue

                nan_error = check_for_nan(conditioning_latents, "conditioning_latents")
                if nan_error:
                    logger.error(f"⚠️ {nan_error} - Skipping batch")
                    continue

                # === Text Encoding ===
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=weight_dtype):
                    encoder_output = text_encoder(clip_input_ids, output_hidden_states=True)
                    pooled_text_embeddings = encoder_output.pooler_output
                    clip_sequence_text_embeddings = encoder_output.hidden_states[-2]

                # NaN Check: Text embeddings
                nan_error = check_for_nan(pooled_text_embeddings, "pooled_text_embeddings")
                if nan_error:
                    logger.error(f"⚠️ {nan_error} - Skipping batch")
                    continue

                # === Rectified Flow Matching ===
                x0 = torch.randn_like(image_latents)
                x1 = image_latents
                # Clamp to prevent extreme values
                x1 = torch.clamp(x1, -10, 10)

                t = torch.rand((x0.shape[0],), device=accelerator.device, dtype=weight_dtype)
                t_expanded = t.view(-1, 1, 1, 1)
                z_t = x0.add(x1.sub(x0).mul(t_expanded))
                v_target = x1 - x0

                # NaN Check: Flow targets
                nan_error = check_for_nan(v_target, "v_target")
                if nan_error:
                    logger.error(f"⚠️ {nan_error} - Skipping batch")
                    continue

                concatenated_input = torch.cat([z_t, conditioning_latents], dim=1)

                # === Model Prediction ===
                try:
                    model_pred = dit_model(
                        x=concatenated_input,
                        t=t,
                        text_pooled_embeddings=pooled_text_embeddings,
                        modality=modality_flags,
                        clip_seq_embeddings=clip_sequence_text_embeddings
                    )
                except RuntimeError as e:
                    logger.error(f"⚠️ Model forward pass failed: {e}")
                    continue

                # NaN Check: Model output
                nan_error = check_for_nan(model_pred, "model_pred")
                if nan_error:
                    logger.error(f"⚠️ {nan_error} at step {global_step}")
                    nan_count += 1
                    consecutive_nans += 1
                    if accelerator.is_main_process:
                        with open(output_dir / "nan_detections.txt", 'a') as f:
                            f.write(f"Step {global_step}: {nan_error}\n")
                            f.write(f"  LR: {lr_scheduler.get_last_lr()[0]}\n")
                            f.write(f"  Consecutive NaNs: {consecutive_nans}\n")
                    if consecutive_nans >= 5:
                        logger.error("⚠️ 5 consecutive NaNs detected! Stopping training.")
                        logger.error("Suggestions:")
                        logger.error("1. Lower learning rate further (try 5e-7)")
                        logger.error("2. Check if checkpoint is corrupted")
                        logger.error("3. Verify data preprocessing")
                        raise ValueError("Training unstable: too many consecutive NaNs")
                    continue

                # Reset consecutive NaN counter on success
                consecutive_nans = 0

                # Compute loss with stability checks
                loss = F.mse_loss(model_pred.float(), v_target.float(), reduction="mean")

                # NaN Check: Loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"⚠️ NaN/Inf loss at step {global_step}")
                    nan_count += 1
                    consecutive_nans += 1
                    continue

                # Clamp loss for extreme values
                if loss.item() > 100.0:
                    logger.warning(f"⚠️ Very high loss: {loss.item():.4f} - clamping to 100.0")
                    loss = torch.clamp(loss, max=100.0)

                # === Backward Pass with Gradient Checking ===
                accelerator.backward(loss)

                # Check gradients for NaN
                if accelerator.sync_gradients:
                    has_nan_grad = False
                    for name, param in dit_model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.error(f"⚠️ NaN/Inf gradient in {name}")
                                has_nan_grad = True
                                break
                    if has_nan_grad:
                        optimizer.zero_grad(set_to_none=True)
                        consecutive_nans += 1
                        continue

                    # Clip gradients
                    grad_norm = accelerator.clip_grad_norm_(dit_model.parameters(), config['training']['max_grad_norm'])

                    # Check if grad norm is reasonable
                    if grad_norm > 10.0:
                        logger.warning(f"⚠️ High gradient norm: {grad_norm:.4f}")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # === Post-Step Operations ===
            if accelerator.sync_gradients:
                global_step += 1
                epoch_steps += 1

                if use_ema and ema_model is not None and global_step % 10 == 0:
                    ema_model.step(dit_model.parameters())

                loss_value = loss.detach().item()
                epoch_loss_sum += loss_value
                loss_accumulator += loss_value
                loss_count += 1

                if global_step % log_interval == 0:
                    avg_loss = loss_accumulator / loss_count
                    current_lr = lr_scheduler.get_last_lr()[0]
                    try:
                        accelerator.log({
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                            "nan_count": nan_count
                        }, step=global_step)
                    except:
                        pass

                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'nans': nan_count
                    })

                    epoch_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'avg': f'{epoch_loss_sum/epoch_steps:.4f}'
                    })

                    loss_accumulator = 0.0
                    loss_count = 0

                progress_bar.update(1)

                # === Save Checkpoint ===
                if global_step % config['logging']['checkpointing_steps'] == 0:
                    if accelerator.is_main_process:
                        save_path = save_checkpoint(
                            accelerator, dit_model, ema_model,
                            output_dir, global_step, use_ema
                        )
                        logger.info(f"\n💾 Checkpoint saved: {save_path.name}")

            if global_step >= max_train_steps:
                break

        # === End of Epoch ===
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            epoch_duration = time.time() - epoch_start_time
            samples_per_sec = len(train_dataset) / epoch_duration
            if accelerator.is_main_process:
                logger.info(f"\n{'='*70}")
                logger.info(f"📈 EPOCH {epoch + 1}/{num_train_epochs} | Loss: {avg_epoch_loss:.6f} | "
                           f"Time: {epoch_duration/60:.1f}m | Speed: {samples_per_sec:.0f} samples/s | "
                           f"NaNs: {nan_count}")
                logger.info(f"{'='*70}\n")
                loss_log_path = output_dir / "training_loss.txt"
                with open(loss_log_path, 'a') as f:
                    f.write(f"Epoch {epoch + 1:3d} | Loss: {avg_epoch_loss:.6f} | "
                           f"Time: {epoch_duration/60:.1f}m | Step: {global_step} | NaNs: {nan_count}\n")

        if global_step >= max_train_steps:
            break

    # === 14. Save Final Model ===
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("\n" + "="*70)
        logger.info("💾 SAVING FINAL MODEL")
        logger.info("="*70)
        unwrapped_model = accelerator.unwrap_model(dit_model)

        if use_ema and ema_model is not None:
            ema_model.copy_to(unwrapped_model.parameters())
            logger.info("✓ Applied EMA weights")

        final_save_path = output_dir / "dit_model_weights_final.safetensors"
        save_file(unwrapped_model.state_dict(), final_save_path)
        logger.info(f"✓ Saved: {final_save_path}")

        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {num_train_epochs} | Steps: {global_step}\n")
        logger.info("="*70 + "\n")

    accelerator.end_training()
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()