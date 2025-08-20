import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils import make_image_grid
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import logging
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import random
import gc
import math
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltrasoundDataset(Dataset):
    def __init__(self, data_path, transforms=None, use_classes=None, augment_factor=8):
        self.data_path = Path(data_path)
        self.transforms = transforms
        self.use_classes = use_classes or ["CN_ON", "CN_OY"]  # Only clean images for DDPM training
        self.augment_factor = augment_factor  # Multiply dataset size by this factor

        self.image_files = []
        self._load_image_files()

    def _load_image_files(self):
        """Load image files based on specified classes"""
        for image_file in self.data_path.glob("*.bmp"):
            filename = image_file.name
            # Check if file belongs to specified classes
            for class_name in self.use_classes:
                if filename.startswith(class_name):
                    self.image_files.append(image_file)
                    break

        logger.info(f"Loaded {len(self.image_files)} base images from classes: {self.use_classes}")
        logger.info(f"Dataset will be augmented by {self.augment_factor}x to {len(self.image_files) * self.augment_factor} total samples")

    def __len__(self):
        return len(self.image_files) * self.augment_factor

    def __getitem__(self, idx):
        # Get base image index
        base_idx = idx % len(self.image_files)
        image_path = self.image_files[base_idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply augmentation before transforms
        image = self._apply_augmentation(image, idx)

        if self.transforms:
            image = self.transforms(image)

        return {"images": image}

    def _apply_augmentation(self, image: Image.Image, idx: int) -> Image.Image:
        """Apply data augmentation specific to ultrasound images"""
        # Use idx to ensure reproducible augmentations
        random.seed(idx)
        np.random.seed(idx % 10000)

        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=0)

        # Random brightness adjustment (0.8 to 1.2)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # Random contrast adjustment (0.8 to 1.2)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # Random scaling and cropping (0.9 to 1.1)
        if random.random() > 0.4:
            width, height = image.size
            scale = random.uniform(0.9, 1.1)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)

            # Crop or pad to original size
            if scale > 1.0:
                # Crop from center
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                image = image.crop((left, top, left + width, top + height))
            else:
                # Pad to original size
                new_img = Image.new('L', (width, height), 0)
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                new_img.paste(image, (paste_x, paste_y))
                image = new_img

        # Small random translation
        if random.random() > 0.5:
            width, height = image.size
            max_shift = min(width, height) // 20  # Max 5% shift
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)

            # Create new image with translation
            new_img = Image.new('L', (width, height), 0)
            new_img.paste(image, (dx, dy))
            image = new_img

        # Add slight Gaussian noise
        if random.random() > 0.6:
            img_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, 2, img_array.shape)  # Small noise
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))

        return image

def get_transforms(image_size=512):
    """Get image transforms for training"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

def evaluate(config, epoch, pipeline, accelerator):
    """Generate sample images for evaluation with memory management"""
    try:
        # Use smaller batch size for evaluation to save memory
        eval_batch_size = min(config.eval_batch_size, 8)

        with torch.no_grad():
            images = pipeline(
                batch_size=eval_batch_size,
                generator=torch.manual_seed(config.seed),
                num_inference_steps=100,  # Reduce steps for faster evaluation
            ).images

        # Make a grid out of the images
        grid_size = min(int(math.sqrt(eval_batch_size)), 4)
        image_grid = make_image_grid(images[:grid_size*grid_size], rows=grid_size, cols=grid_size)

        # Save the images
        test_dir = Path(config.output_dir) / "samples"
        test_dir.mkdir(exist_ok=True)
        image_grid.save(test_dir / f"sample-{epoch:04d}.png")

        # Clean up
        del images
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}. Skipping evaluation for epoch {epoch}")

def main():
    parser = argparse.ArgumentParser(description="Train DDPM on ultrasound images")
    parser.add_argument("--train_data_dir", type=str,
                       default="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/data/train_CN_CY_ALL",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./ddpm-ultrasound-model",
                       help="Output directory for model and samples")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--save_image_epochs", type=int, default=20,
                       help="Save sample images every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=50,
                       help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--augment_factor", type=int, default=8,
                       help="Data augmentation multiplier")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true",
                       help="Pin memory for faster data transfer")
    parser.add_argument("--use_ema", action="store_true",
                       help="Use Exponential Moving Average")
    parser.add_argument("--checkpointing_steps", type=int, default=2000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")

    args = parser.parse_args()

    config = args

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Create dataset and dataloader
    train_transforms = get_transforms(config.resolution)
    train_dataset = UltrasoundDataset(
        config.train_data_dir,
        transforms=train_transforms,
        use_classes=["CN_ON", "CN_OY"],  # Only clean images for DDPM training
        augment_factor=config.augment_factor
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )

    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Dataloader batches per epoch: {len(train_dataloader)}")

    # Create model - optimized for 512x512 and A100
    model = UNet2DModel(
        sample_size=config.resolution,
        in_channels=1,  # Grayscale images
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512, 1024, 1024),  # Larger model for 512px
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # Attention at 64x64
            "AttnDownBlock2D",  # Attention at 32x32
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",   # Attention at 32x32
            "AttnUpBlock2D",   # Attention at 64x64
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=8,  # Optimized for A100
        norm_num_groups=32,
        class_embed_type=None,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Optimizer with better settings for large models
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # Learning rate scheduler
    num_training_steps = config.num_epochs * len(train_dataloader) // config.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize EMA if requested
    ema_model = None
    if config.use_ema:
        from diffusers.training_utils import EMAModel
        ema_model = EMAModel(model.parameters(), decay=0.9999, update_after_step=0)
        ema_model.to(accelerator.device)

    logger.info(f"Training steps per epoch: {len(train_dataloader)}")
    logger.info(f"Total training steps: {num_training_steps}")

    global_step = 0
    best_loss = float('inf')

    # Now you train the model
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device, dtype=clean_images.dtype)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                if config.use_ema and ema_model is not None:
                    ema_model.step(model.parameters())

            # Logging
            total_loss += loss.detach().item()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

            # Save checkpoint periodically with sampling
            if global_step % config.checkpointing_steps == 0 and accelerator.is_main_process:
                checkpoint_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                accelerator.save_state(checkpoint_path)
                logger.info(f"Saved checkpoint at step {global_step}")

                # Generate samples when checkpoint is saved
                logger.info(f"Generating samples at step {global_step}...")
                if config.use_ema and ema_model is not None:
                    # For EMA model, copy weights to original model for checkpoint sampling
                    ema_model.copy_to(model.parameters())
                    checkpoint_unet = accelerator.unwrap_model(model)
                else:
                    checkpoint_unet = accelerator.unwrap_model(model)

                checkpoint_pipeline = DDPMPipeline(
                    unet=checkpoint_unet,
                    scheduler=noise_scheduler
                )

                # Generate samples with current checkpoint
                try:
                    eval_batch_size = min(config.eval_batch_size, 8)

                    with torch.no_grad():
                        checkpoint_images = checkpoint_pipeline(
                            batch_size=eval_batch_size,
                            generator=torch.manual_seed(config.seed + global_step),  # Different seed for variety
                            num_inference_steps=100,
                        ).images

                    # Make a grid out of the images
                    grid_size = min(int(math.sqrt(eval_batch_size)), 4)
                    checkpoint_grid = make_image_grid(checkpoint_images[:grid_size*grid_size], rows=grid_size, cols=grid_size)

                    # Save the checkpoint samples
                    checkpoint_samples_dir = Path(config.output_dir) / "checkpoint_samples"
                    checkpoint_samples_dir.mkdir(exist_ok=True)
                    checkpoint_grid.save(checkpoint_samples_dir / f"checkpoint-{global_step}-samples.png")
                    logger.info(f"Saved checkpoint samples: checkpoint-{global_step}-samples.png")

                    # Clean up
                    del checkpoint_images, checkpoint_grid

                except Exception as e:
                    logger.warning(f"Failed to generate checkpoint samples at step {global_step}: {e}")

                # Cleanup pipeline
                del checkpoint_pipeline
                torch.cuda.empty_cache()
                gc.collect()

            # Memory cleanup
            if step % 100 == 0:
                torch.cuda.empty_cache()

        # Calculate average loss for epoch
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss and accelerator.is_main_process:
            best_loss = avg_loss
            if config.use_ema and ema_model is not None:
                # For EMA model, we need to copy weights back to original model
                ema_model.copy_to(model.parameters())
                best_unet = accelerator.unwrap_model(model)
            else:
                best_unet = accelerator.unwrap_model(model)

            pipeline = DDPMPipeline(
                unet=best_unet,
                scheduler=noise_scheduler
            )
            best_path = Path(config.output_dir) / "best_model"
            pipeline.save_pretrained(best_path)
            logger.info(f"New best model saved with loss: {best_loss:.6f}")

        # After each epoch, optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if config.use_ema and ema_model is not None:
                # For EMA model, copy weights to original model for evaluation
                ema_model.copy_to(model.parameters())
                eval_unet = accelerator.unwrap_model(model)
            else:
                eval_unet = accelerator.unwrap_model(model)

            pipeline = DDPMPipeline(
                unet=eval_unet,
                scheduler=noise_scheduler
            )

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, accelerator)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                save_path = Path(config.output_dir) / f"epoch_{epoch+1}"
                pipeline.save_pretrained(save_path)
                logger.info(f"Model saved at epoch {epoch+1}")

            # Cleanup
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Training completed! Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()