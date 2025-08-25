import argparse
import os
from pathlib import Path
import random
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

logger = get_logger(__name__, log_level="INFO")

class EMAModel:
    """Enhanced EMA implementation with better decay scheduling"""
    def __init__(self, parameters, decay=0.9999, use_ema_warmup=True, inv_gamma=1.0, power=2/3, min_decay=0.0):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.decay = decay
        self.optimization_step = 0
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_decay = min_decay
        
    def get_decay(self, optimization_step):
        """Compute decay value with improved scheduling"""
        if self.use_ema_warmup:
            step = max(1, optimization_step)
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
            return max(min(value, self.decay), self.min_decay)
        else:
            return self.decay
    
    def step(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        self.optimization_step += 1
        
        decay = self.get_decay(self.optimization_step)
        
        for s_param, param in zip(self.shadow_params, parameters):
            s_param.sub_((1.0 - decay) * (s_param - param))
    
    def copy_to(self, parameters):
        """Copy EMA parameters to model"""
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)
    
    def store(self, parameters):
        """Store current model parameters"""
        self.temp_stored_params = [param.clone().detach() for param in parameters if param.requires_grad]
    
    def restore(self, parameters):
        """Restore previously stored model parameters"""
        parameters = [p for p in parameters if p.requires_grad]
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)
    
    @property
    def cur_decay_value(self):
        return self.get_decay(self.optimization_step)


class PerceptualLoss(torch.nn.Module):
    """Simple perceptual loss using feature differences"""
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, pred, target):
        # Multi-scale loss
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        # Add gradient penalty for smoother results
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        grad_loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return mse + 0.1 * l1 + 0.05 * (grad_loss_x + grad_loss_y)


class NoiseSchedulerV(torch.nn.Module):
    """V-parameterization noise scheduler for better training dynamics"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        
        # Use cosine schedule for better quality
        betas = self.cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008, beta_start=0.0001, beta_end=0.02):
        """Cosine schedule for better quality"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    
    def add_noise(self, original_samples, noise, timesteps):
        # Move timesteps to CPU to match scheduler tensors
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        # Move back to device
        sqrt_alpha_prod = sqrt_alpha_prod.to(original_samples.device)
            
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # Move back to device
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(original_samples.device)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_v_prediction(self, original_samples, noise, timesteps):
        """V-parameterization target"""
        # Move timesteps to CPU to match scheduler tensors
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        # Move back to device
        sqrt_alpha_prod = sqrt_alpha_prod.to(original_samples.device)
            
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # Move back to device
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(original_samples.device)
            
        v_pred = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original_samples
        return v_pred

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced DDPM training script for high-quality 512x512 images")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/ubuntu/Desktop/JY/PAADI/diffusers/Dataset_BUSI_with_GT",
        help="Path to dataset directory containing benign, malignant, normal folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-busi-512-enhanced",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images"
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=4, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=4, 
        help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=200,
        help="Number of training epochs - increased for better quality"
    )
    parser.add_argument(
        "--save_images_epochs", 
        type=int, 
        default=10, 
        help="How often to save images during training."
    )
    parser.add_argument(
        "--save_model_epochs", 
        type=int, 
        default=25, 
        help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate - reduced for better stability.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-4, 
        help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma", 
        type=float, 
        default=1.0, 
        help="The inverse gamma value for the EMA decay."
    )
    parser.add_argument(
        "--ema_power", 
        type=float, 
        default=2/3, 
        help="The power value for the EMA decay."
    )
    parser.add_argument(
        "--ema_max_decay", 
        type=float, 
        default=0.9999, 
        help="The maximum decay magnitude for EMA."
    )
    parser.add_argument(
        "--use_v_prediction",
        action="store_true",
        default=True,
        help="Whether to use v-parameterization for better training dynamics."
    )
    parser.add_argument(
        "--use_perceptual_loss",
        action="store_true", 
        default=True,
        help="Whether to use perceptual loss for better quality."
    )
    parser.add_argument(
        "--progressive_training",
        action="store_true",
        default=False,
        help="Whether to use progressive training starting from lower resolution."
    )
    parser.add_argument(
        "--advanced_augmentation",
        action="store_true",
        default=True,
        help="Whether to use advanced data augmentation techniques."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps during inference."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for better generation quality."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_tensorboard",
        action="store_true",
        help="Enable TensorBoard logging for training metrics and images"
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs"
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_path is None:
        raise ValueError("Need a dataset folder.")

    return args

def load_local_dataset(dataset_path, split="train"):
    """Load images from local directory structure with better organization"""
    dataset_dict = {"image": [], "category": []}
    
    for category in ["benign", "malignant", "normal"]:
        category_path = Path(dataset_path) / category
        if category_path.exists():
            image_files = [f for f in category_path.glob("*.png") if not f.name.endswith("_mask.png")]
            for img_path in image_files:
                dataset_dict["image"].append(str(img_path))
                dataset_dict["category"].append(category)
    
    return dataset_dict


def apply_advanced_augmentation(image, strength=0.3):
    """Apply advanced data augmentation techniques"""
    if random.random() < strength:
        # Random brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    if random.random() < strength:
        # Random contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    if random.random() < strength * 0.5:
        # Light gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
    
    if random.random() < strength * 0.3:
        # Random rotation (small angles)
        angle = random.uniform(-5, 5)
        image = image.rotate(angle, fillcolor=0)
    
    return image

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize TensorBoard writer
    writer = None
    if args.enable_tensorboard and accelerator.is_main_process:
        tensorboard_dir = os.path.join(args.output_dir, args.tensorboard_log_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir {tensorboard_dir}")

    # Load dataset
    dataset_dict = load_local_dataset(args.dataset_path)
    dataset = {
        "image": dataset_dict["image"]
    }
    
    logger.info(f"Dataset contains {len(dataset['image'])} images")

    def preprocess_image(image, resolution, apply_augmentations=True, advanced_aug=False):
        """Enhanced preprocessing function with better augmentation"""
        # Resize image with better resampling
        image = image.resize((resolution, resolution), Image.LANCZOS)
        
        # Apply advanced augmentation if enabled
        if apply_augmentations and advanced_aug:
            image = apply_advanced_augmentation(image, strength=0.25)
        
        # Random horizontal flip for augmentation
        if apply_augmentations and random.random() > 0.5:
            image = ImageOps.mirror(image)
            
        # Random vertical flip (less common but useful for medical images)
        if apply_augmentations and random.random() > 0.8:
            image = ImageOps.flip(image)
        
        # Convert to numpy array with better normalization
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Apply histogram equalization-like effect for better contrast
        if apply_augmentations and random.random() > 0.7:
            image_array = np.clip((image_array - image_array.mean()) / (image_array.std() + 1e-8) * 0.1 + image_array.mean(), 0, 1)
        
        # Normalize to [-1, 1] with slightly better scaling
        image_array = (image_array - 0.5) / 0.5
        
        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add channel dim for grayscale
        
        return image_tensor


    # Create enhanced dataset from dict
    train_dataset = []
    for img_path in dataset["image"]:
        try:
            image = Image.open(img_path).convert("L")  # Convert to grayscale
            transformed_image = preprocess_image(
                image, 
                args.resolution, 
                apply_augmentations=True, 
                advanced_aug=args.advanced_augmentation
            )
            train_dataset.append({"input": transformed_image})
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(train_dataset)} images")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )

    # Create enhanced model optimized for 512x512 resolution
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=1,  # Grayscale input
        out_channels=1,  # Grayscale output
        layers_per_block=2,  # Optimal for 512x512
        # Optimized channels for 512x512: 512→256→128→64→32→16→8 feature map sizes
        block_out_channels=(128, 256, 512, 512, 768, 1024),  # 512x512 optimized
        down_block_types=(
            "DownBlock2D",      # 512→256
            "DownBlock2D",      # 256→128  
            "DownBlock2D",      # 128→64
            "AttnDownBlock2D",  # 64→32 (start attention at reasonable resolution)
            "AttnDownBlock2D",  # 32→16
            "AttnDownBlock2D",  # 16→8 (deepest level with attention)
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 8→16
            "AttnUpBlock2D",    # 16→32
            "AttnUpBlock2D",    # 32→64
            "UpBlock2D",        # 64→128
            "UpBlock2D",        # 128→256
            "UpBlock2D",        # 256→512
        ),
        attention_head_dim=16,  # Increased for 512x512
    )

    # Create enhanced noise scheduler or v-prediction scheduler
    if args.use_v_prediction:
        noise_scheduler = NoiseSchedulerV(num_train_timesteps=1000)
        logger.info("Using V-parameterization for better training dynamics")
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",  # Better schedule
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon"
        )

    # Initialize enhanced optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the training dataset
    train_dataloader = accelerator.prepare(train_dataloader)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
        num_cycles=3 if args.lr_scheduler == "cosine_with_restarts" else 1,
    )

    # Initialize perceptual loss if enabled
    perceptual_loss = None
    if args.use_perceptual_loss:
        perceptual_loss = PerceptualLoss()
        logger.info("Using perceptual loss for enhanced quality")

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    if perceptual_loss:
        perceptual_loss = accelerator.prepare(perceptual_loss)

    # Initialize enhanced EMA
    if args.use_ema:
        ema_model = EMAModel(
            accelerator.unwrap_model(model).parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            min_decay=0.9990,
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            
            # Sample a random timestep for each image with bias towards difficult timesteps
            if random.random() < 0.2:  # 20% chance to sample from difficult timesteps
                timesteps = torch.randint(
                    noise_scheduler.num_train_timesteps // 2, 
                    noise_scheduler.num_train_timesteps, 
                    (bsz,), device=clean_images.device
                ).long()
            else:
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual or v-prediction
                model_output = model(noisy_images, timesteps).sample
                
                # Calculate loss based on prediction type
                if args.use_v_prediction and hasattr(noise_scheduler, 'get_v_prediction'):
                    target = noise_scheduler.get_v_prediction(clean_images, noise, timesteps)
                    loss = F.mse_loss(model_output, target)
                else:
                    loss = F.mse_loss(model_output, noise)
                
                # Add perceptual loss if enabled
                if args.use_perceptual_loss and perceptual_loss:
                    # For perceptual loss, we need to reconstruct the predicted clean image
                    if args.use_v_prediction:
                        # Move timesteps to CPU to match scheduler tensors
                        timesteps_cpu = timesteps.cpu()
                        sqrt_alpha_prod = noise_scheduler.sqrt_alphas_cumprod[timesteps_cpu].flatten()
                        sqrt_one_minus_alpha_prod = noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps_cpu].flatten()
                        while len(sqrt_alpha_prod.shape) < len(clean_images.shape):
                            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        # Move back to device
                        sqrt_alpha_prod = sqrt_alpha_prod.to(clean_images.device)
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(clean_images.device)
                        
                        pred_original = sqrt_alpha_prod * clean_images + sqrt_one_minus_alpha_prod * model_output
                    else:
                        # Standard epsilon prediction
                        # Move timesteps to CPU to match scheduler tensors
                        timesteps_cpu = timesteps.cpu()
                        alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps_cpu].flatten()
                        while len(alpha_prod_t.shape) < len(clean_images.shape):
                            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
                        # Move back to device
                        alpha_prod_t = alpha_prod_t.to(clean_images.device)
                        pred_original = (noisy_images - ((1 - alpha_prod_t) ** 0.5) * model_output) / (alpha_prod_t ** 0.5)
                    
                    perceptual_loss_val = perceptual_loss(pred_original, clean_images)
                    loss = loss + 0.1 * perceptual_loss_val

                accelerator.backward(loss)

                # Enhanced gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(accelerator.unwrap_model(model).parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            
            # Log metrics to TensorBoard
            if writer is not None and accelerator.is_main_process:
                writer.add_scalar("train/loss", loss.detach().item(), global_step)
                writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
                if args.use_ema:
                    writer.add_scalar("train/ema_decay", ema_model.cur_decay_value, global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                if args.use_ema:
                    ema_model.store(accelerator.unwrap_model(model).parameters())
                    ema_model.copy_to(accelerator.unwrap_model(model).parameters())

                # Use enhanced sampling with DPM-Solver++ for better quality
                try:
                    # Try using DPM-Solver++ for faster and better sampling
                    dpm_scheduler = DPMSolverMultistepScheduler.from_config(
                        noise_scheduler.config if hasattr(noise_scheduler, 'config') else {
                            "num_train_timesteps": 1000,
                            "beta_start": 0.00085,
                            "beta_end": 0.012,
                            "beta_schedule": "scaled_linear",
                        }
                    )
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(model),
                        scheduler=dpm_scheduler,
                    )
                except:
                    # Fallback to standard DDPM scheduler
                    if args.use_v_prediction:
                        inference_scheduler = DDPMScheduler(
                            num_train_timesteps=1000,
                            beta_schedule="scaled_linear",
                            beta_start=0.00085,
                            beta_end=0.012,
                        )
                    else:
                        inference_scheduler = noise_scheduler
                    
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(model),
                        scheduler=inference_scheduler,
                    )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference with enhanced settings
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.num_inference_steps,
                ).images

                # denormalize the images and save to disk
                image_save_dir = Path(args.output_dir) / f"samples_epoch_{epoch:04d}"
                image_save_dir.mkdir(exist_ok=True)
                
                for i, image in enumerate(images):
                    image.save(image_save_dir / f"image_{i:04d}.png")
                
                # Log images to TensorBoard
                if writer is not None:
                    # Convert PIL images to tensors for TensorBoard
                    image_tensors = []
                    for image in images[:8]:  # Log only first 8 images to save space
                        # Convert PIL to RGB and then to tensor
                        img_array = np.array(image.convert('RGB'))
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                        image_tensors.append(img_tensor)
                    
                    if len(image_tensors) > 0:
                        # Stack images and log as grid
                        images_tensor = torch.stack(image_tensors)
                        writer.add_images(f"generated_samples/epoch_{epoch}", images_tensor, epoch, dataformats='NCHW')

                if args.use_ema:
                    ema_model.restore(accelerator.unwrap_model(model).parameters())

        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.store(accelerator.unwrap_model(model).parameters())
                    ema_model.copy_to(accelerator.unwrap_model(model).parameters())

                # Save with enhanced scheduler configuration
                if args.use_v_prediction:
                    save_scheduler = DDPMScheduler(
                        num_train_timesteps=1000,
                        beta_schedule="scaled_linear",
                        beta_start=0.00085,
                        beta_end=0.012,
                        prediction_type="v_prediction"
                    )
                else:
                    save_scheduler = noise_scheduler

                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=save_scheduler,
                )
                pipeline.save_pretrained(args.output_dir)
                
                # Save additional metadata
                metadata = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "resolution": args.resolution,
                    "use_v_prediction": args.use_v_prediction,
                    "use_perceptual_loss": args.use_perceptual_loss,
                    "advanced_augmentation": args.advanced_augmentation,
                }
                import json
                with open(Path(args.output_dir) / "training_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                if args.use_ema:
                    ema_model.restore(accelerator.unwrap_model(model).parameters())

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed")
    
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)