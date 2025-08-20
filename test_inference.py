#!/usr/bin/env python3
"""
DDPM Test Inference Script
Test inference using trained 512x512 ultrasound DDPM model
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from torchvision import transforms
import math
import gc
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import diffusers components
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.utils import make_image_grid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltrasoundDDPMInference:
    def __init__(self, model_path, device='cuda', image_size=512):
        """
        Initialize DDPM inference pipeline
        
        Args:
            model_path: Path to trained DDPM model directory
            device: Device to use (cuda/cpu)
            image_size: Image resolution (should match training)
        """
        self.device = device
        self.image_size = image_size
        self.model_path = Path(model_path)
        
        # Load trained model
        self.pipeline = self.load_model()
        
        # Setup image transforms
        self.transforms = self.get_transforms()
        
    def load_model(self):
        """Load the trained DDPM pipeline"""
        try:
            logger.info(f"Loading DDPM model from {self.model_path}")
            
            # Load the best model if available, otherwise use the latest epoch
            best_model_path = self.model_path / "best_model"
            if best_model_path.exists():
                logger.info("Loading best model...")
                logger.info("DEBUG: Calling DDPMPipeline.from_pretrained...")
                pipeline = DDPMPipeline.from_pretrained(best_model_path)
                logger.info("DEBUG: DDPMPipeline.from_pretrained finished.")
            else:
                # Find latest epoch model
                epoch_dirs = [d for d in self.model_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
                if epoch_dirs:
                    latest_epoch = max(epoch_dirs, key=lambda x: int(x.name.split("_")[1]))
                    logger.info(f"Loading latest epoch model: {latest_epoch}")
                    logger.info("DEBUG: Calling DDPMPipeline.from_pretrained...")
                    pipeline = DDPMPipeline.from_pretrained(latest_epoch)
                    logger.info("DEBUG: DDPMPipeline.from_pretrained finished.")
                else:
                    # Load from main directory
                    logger.info("Loading from main model directory...")
                    logger.info("DEBUG: Calling DDPMPipeline.from_pretrained...")
                    pipeline = DDPMPipeline.from_pretrained(self.model_path)
                    logger.info("DEBUG: DDPMPipeline.from_pretrained finished.")

            # Move to device
            logger.info("DEBUG: Moving pipeline to device...")
            pipeline = pipeline.to(self.device)
            logger.info("DEBUG: Pipeline moved to device.")
            pipeline.unet.eval()
            
            logger.info(f"Successfully loaded DDPM model on {self.device}")
            logger.info(f"Model input channels: {pipeline.unet.config.in_channels}")
            logger.info(f"Model output channels: {pipeline.unet.config.out_channels}")
            logger.info(f"Sample size: {pipeline.unet.config.sample_size}")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def load_test_images(self, test_dir, max_images=None):
        """
        Load test images from directory
        
        Args:
            test_dir: Directory containing test images
            max_images: Maximum number of images to load (None for all)
        """
        test_dir = Path(test_dir)
        image_files = sorted(list(test_dir.glob("*.bmp")))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} test images in {test_dir}")
        
        images = []
        image_paths = []
        
        for img_path in image_files:
            try:
                # Load and convert to grayscale
                img = Image.open(img_path).convert('L')
                
                # Apply transforms
                img_tensor = self.transforms(img)
                
                images.append(img_tensor)
                image_paths.append(img_path)
                
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        if images:
            # Stack into batch
            images_batch = torch.stack(images)
            logger.info(f"Loaded {len(images)} images with shape: {images_batch.shape}")
            return images_batch, image_paths
        else:
            logger.error("No images loaded successfully")
            return None, None
    
    def generate_unconditional_samples(self, num_samples=8, num_inference_steps=100, 
                                     generator_seed=None):
        """
        Generate unconditional samples from the trained model
        
        Args:
            num_samples: Number of samples to generate
            num_inference_steps: Number of denoising steps
            generator_seed: Random seed for reproducibility
        """
        logger.info(f"Generating {num_samples} unconditional samples...")
        
        generator = None
        if generator_seed is not None:
            generator = torch.manual_seed(generator_seed)
        
        try:
            with torch.no_grad():
                # Generate samples
                generated = self.pipeline(
                    batch_size=num_samples,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                )
                
                return generated.images
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def encode_images(self, images_batch):
        """
        Encode images to latent space (if using VAE)
        For direct pixel space models, this just returns the images
        """
        return images_batch.to(self.device)
    
    def reconstruct_images(self, images_batch, num_inference_steps=50, noise_strength=0.8):
        """
        Reconstruct images by adding noise and denoising
        
        Args:
            images_batch: Batch of input images
            num_inference_steps: Number of denoising steps
            noise_strength: Amount of noise to add (0.0 to 1.0)
        """
        logger.info(f"Reconstructing {images_batch.shape[0]} images...")
        
        batch_size = images_batch.shape[0]
        device = self.device
        
        # Move images to device
        images = images_batch.to(device)
        
        # Set up scheduler
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipeline.scheduler.timesteps
        
        # Add noise to input images
        noise_timestep = int(noise_strength * len(timesteps))
        if noise_timestep >= len(timesteps):
            noise_timestep = len(timesteps) - 1
        
        noise_t = timesteps[noise_timestep]
        noise = torch.randn_like(images)
        
        # Add noise according to scheduler
        noisy_images = self.pipeline.scheduler.add_noise(images, noise, noise_t)
        
        # Start denoising from the noisy images
        current_images = noisy_images
        
        try:
            with torch.no_grad():
                # Denoising loop
                for i, t in enumerate(tqdm(timesteps[noise_timestep:], desc="Reconstructing")):
                    # Predict noise
                    noise_pred = self.pipeline.unet(current_images, t, return_dict=False)[0]
                    
                    # Scheduler step
                    current_images = self.pipeline.scheduler.step(
                        noise_pred, t, current_images, return_dict=False
                    )[0]
                    
                    # Memory cleanup every 10 steps
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Convert back to PIL images
            reconstructed = self.tensor_to_pil(current_images)
            return reconstructed
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise
    
    def interpolate_images(self, img1_batch, img2_batch, num_steps=10, num_inference_steps=50):
        """
        Interpolate between two sets of images in latent space
        
        Args:
            img1_batch: First set of images
            img2_batch: Second set of images  
            num_steps: Number of interpolation steps
            num_inference_steps: Number of denoising steps
        """
        logger.info(f"Interpolating between {img1_batch.shape[0]} image pairs...")
        
        # Encode images
        latent1 = self.encode_images(img1_batch)
        latent2 = self.encode_images(img2_batch)
        
        interpolations = []
        
        # Generate interpolation weights
        alphas = torch.linspace(0, 1, num_steps)
        
        for alpha in tqdm(alphas, desc="Interpolating"):
            # Linear interpolation in latent space
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            
            # Add noise and denoise to get realistic images
            noise = torch.randn_like(interpolated_latent) * 0.1
            noisy_latent = interpolated_latent + noise
            
            # Quick denoising (fewer steps for interpolation)
            reconstructed = self.reconstruct_from_latent(noisy_latent, num_inference_steps//2)
            interpolations.extend(reconstructed)
        
        return interpolations
    
    def reconstruct_from_latent(self, latent, num_inference_steps=25):
        """Reconstruct image from latent representation"""
        try:
            with torch.no_grad():
                # Simple denoising from given latent
                self.pipeline.scheduler.set_timesteps(num_inference_steps)
                timesteps = self.pipeline.scheduler.timesteps
                
                current = latent
                
                for t in timesteps:
                    noise_pred = self.pipeline.unet(current, t, return_dict=False)[0]
                    current = self.pipeline.scheduler.step(noise_pred, t, current, return_dict=False)[0]
                
                return self.tensor_to_pil(current)
                
        except Exception as e:
            logger.error(f"Latent reconstruction failed: {e}")
            return []
    
    def tensor_to_pil(self, tensor_batch):
        """Convert tensor batch to PIL images"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor_batch = (tensor_batch + 1.0) / 2.0
        tensor_batch = torch.clamp(tensor_batch, 0, 1)
        
        # Convert to numpy
        numpy_batch = tensor_batch.cpu().numpy()
        
        pil_images = []
        for i in range(numpy_batch.shape[0]):
            # Convert to uint8
            img_array = (numpy_batch[i].squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array, mode='L')
            pil_images.append(pil_img)
        
        return pil_images
    
    def calculate_metrics(self, original_images, reconstructed_images):
        """
        Calculate image quality metrics between original and reconstructed images
        
        Args:
            original_images: List of PIL images (originals)
            reconstructed_images: List of PIL images (reconstructed)
            
        Returns:
            List of dictionaries containing metrics for each image pair
        """
        metrics_list = []
        
        for i, (orig_img, recon_img) in enumerate(zip(original_images, reconstructed_images)):
            # Convert PIL images to numpy arrays
            orig_array = np.array(orig_img).astype(np.float32) / 255.0
            recon_array = np.array(recon_img).astype(np.float32) / 255.0
            
            # Calculate metrics
            mse = np.mean((orig_array - recon_array) ** 2)
            
            # PSNR calculation
            if mse == 0:
                psnr_value = float('inf')
            else:
                psnr_value = psnr(orig_array, recon_array, data_range=1.0)
            
            # SSIM calculation
            ssim_value = ssim(orig_array, recon_array, data_range=1.0)
            
            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(orig_array - recon_array))
            
            # Standard deviation of the difference
            diff_std = np.std(orig_array - recon_array)
            
            metrics = {
                'image_id': i,
                'mse': float(mse),
                'psnr': float(psnr_value),
                'ssim': float(ssim_value),
                'mae': float(mae),
                'diff_std': float(diff_std)
            }
            
            metrics_list.append(metrics)
            
        return metrics_list
    
    def save_metrics_to_csv(self, metrics_list, output_path):
        """Save metrics to CSV file"""
        if not metrics_list:
            logger.warning("No metrics to save")
            return
            
        # Calculate summary statistics
        summary_metrics = {
            'image_id': 'AVERAGE',
            'mse': np.mean([m['mse'] for m in metrics_list]),
            'psnr': np.mean([m['psnr'] for m in metrics_list if m['psnr'] != float('inf')]),
            'ssim': np.mean([m['ssim'] for m in metrics_list]),
            'mae': np.mean([m['mae'] for m in metrics_list]),
            'diff_std': np.mean([m['diff_std'] for m in metrics_list])
        }
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_id', 'mse', 'psnr', 'ssim', 'mae', 'diff_std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write individual metrics
            for metrics in metrics_list:
                writer.writerow(metrics)
            
            # Write summary row
            writer.writerow({})  # Empty row for separation
            writer.writerow(summary_metrics)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def save_results(self, images, output_dir, prefix="result", save_grid=True):
        """Save generated/reconstructed images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save individual images
        for i, img in enumerate(images):
            img_path = output_dir / f"{prefix}_{i:03d}.png"
            img.save(img_path)
        
        logger.info(f"Saved {len(images)} images to {output_dir}")
        
        # Save grid if requested
        if save_grid and len(images) > 1:
            grid_size = min(int(math.sqrt(len(images))), 8)
            if grid_size > 1:
                try:
                    grid = make_image_grid(images[:grid_size*grid_size], 
                                        rows=grid_size, cols=grid_size)
                    grid_path = output_dir / f"{prefix}_grid.png"
                    grid.save(grid_path)
                    logger.info(f"Saved grid to {grid_path}")
                except Exception as e:
                    logger.warning(f"Failed to create grid: {e}")

def main():
    parser = argparse.ArgumentParser(description="DDPM Test Inference")
    parser.add_argument("--model_path", type=str, 
                       default="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100",
                       help="Path to trained DDPM model")
    parser.add_argument("--test_dir", type=str,
                       default="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/data/test_CN",
                       help="Directory containing test images")
    parser.add_argument("--output_dir", type=str,
                       default="./inference_results",
                       help="Directory to save results")
    parser.add_argument("--max_test_images", type=int, default=10,
                       help="Maximum number of test images to process")
    parser.add_argument("--num_generate", type=int, default=8,
                       help="Number of unconditional samples to generate")
    parser.add_argument("--num_inference_steps", type=int, default=100,
                       help="Number of inference steps")
    parser.add_argument("--noise_strength", type=float, default=0.8,
                       help="Noise strength for reconstruction (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["generate", "reconstruct", "all"],
                       help="Inference mode")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize inference pipeline
    logger.info("Initializing DDPM inference pipeline...")
    inference = UltrasoundDDPMInference(args.model_path, args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Mode 1: Unconditional Generation
        if args.mode in ["generate", "all"]:
            logger.info("=== Unconditional Generation ===")
            generated_images = inference.generate_unconditional_samples(
                num_samples=args.num_generate,
                num_inference_steps=args.num_inference_steps,
                generator_seed=args.seed
            )
            
            # Save generated images
            inference.save_results(generated_images, output_dir / "generated", 
                                 prefix="generated", save_grid=True)
        
        # Mode 2: Image Reconstruction
        if args.mode in ["reconstruct", "all"]:
            logger.info("=== Image Reconstruction ===")
            
            # Load test images
            test_images, test_paths = inference.load_test_images(
                args.test_dir, max_images=args.max_test_images
            )
            
            if test_images is not None:
                # Save original images for comparison
                original_pil = inference.tensor_to_pil(test_images)
                inference.save_results(original_pil, output_dir / "originals",
                                     prefix="original", save_grid=True)
                
                # Reconstruct images
                reconstructed_images = inference.reconstruct_images(
                    test_images,
                    num_inference_steps=args.num_inference_steps,
                    noise_strength=args.noise_strength
                )
                
                # Save reconstructed images
                inference.save_results(reconstructed_images, output_dir / "reconstructed",
                                     prefix="reconstructed", save_grid=True)
                
                # Calculate and save metrics
                logger.info("Calculating image quality metrics...")
                metrics_list = inference.calculate_metrics(original_pil, reconstructed_images)
                
                # Save metrics to CSV
                metrics_csv_path = output_dir / "reconstruction_metrics.csv"
                inference.save_metrics_to_csv(metrics_list, metrics_csv_path)
                
                # Log summary metrics
                avg_psnr = np.mean([m['psnr'] for m in metrics_list if m['psnr'] != float('inf')])
                avg_ssim = np.mean([m['ssim'] for m in metrics_list])
                avg_mse = np.mean([m['mse'] for m in metrics_list])
                
                logger.info(f"Average PSNR: {avg_psnr:.4f}")
                logger.info(f"Average SSIM: {avg_ssim:.4f}")
                logger.info(f"Average MSE: {avg_mse:.6f}")
                
                # Create comparison grid
                comparison_images = []
                for orig, recon in zip(original_pil[:8], reconstructed_images[:8]):
                    comparison_images.extend([orig, recon])
                
                inference.save_results(comparison_images, output_dir / "comparison",
                                     prefix="comparison", save_grid=True)
                
                logger.info("Reconstruction completed!")
            else:
                logger.error("No test images loaded, skipping reconstruction")
        
        logger.info(f"All inference completed! Results saved to {output_dir}")
        
        # Print summary
        logger.info("=== Summary ===")
        if args.mode in ["generate", "all"]:
            logger.info(f"Generated {args.num_generate} unconditional samples")
        if args.mode in ["reconstruct", "all"]:
            num_reconstructed = min(args.max_test_images, len(test_paths) if 'test_paths' in locals() else 0)
            logger.info(f"Reconstructed {num_reconstructed} test images")
            if 'metrics_list' in locals() and metrics_list:
                logger.info(f"Metrics saved to: {metrics_csv_path}")
                logger.info(f"Average reconstruction quality:")
                logger.info(f"  - PSNR: {avg_psnr:.4f} dB")
                logger.info(f"  - SSIM: {avg_ssim:.4f}")
                logger.info(f"  - MSE: {avg_mse:.6f}")
        logger.info(f"Inference steps: {args.num_inference_steps}")
        logger.info(f"Results directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()