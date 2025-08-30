import argparse
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import json
import glob
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel


class ImageMetrics:
    """Advanced image quality metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Try to load LPIPS
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
            print("✅ LPIPS loaded successfully")
        except ImportError:
            print("⚠️ LPIPS not available. Install with: pip install lpips")
            self.lpips_available = False
    
    def calculate_fid_simple(self, real_images, fake_images):
        """Simplified FID calculation using basic statistics"""
        def extract_features(images):
            features = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = torch.from_numpy(np.array(img)).float() / 255.0
                    if len(img.shape) == 2:
                        img = img.unsqueeze(0)
                
                flat = img.flatten()
                mean = torch.mean(flat)
                std = torch.std(flat)
                
                # Edge features
                if len(img.shape) == 3 and img.shape[0] == 1:
                    img_2d = img.squeeze(0)
                else:
                    img_2d = img
                    
                edge_x = torch.abs(img_2d[:, :-1] - img_2d[:, 1:])
                edge_y = torch.abs(img_2d[:-1, :] - img_2d[1:, :])
                edge_mean = torch.mean(torch.cat([edge_x.flatten(), edge_y.flatten()]))
                
                features.append([mean.item(), std.item(), edge_mean.item()])
                
            return torch.tensor(features)
        
        real_features = extract_features(real_images)
        fake_features = extract_features(fake_images)
        
        real_mean = torch.mean(real_features, dim=0)
        fake_mean = torch.mean(fake_features, dim=0)
        
        fid_score = torch.norm(real_mean - fake_mean).item()
        return fid_score
    
    def calculate_lpips(self, real_images, fake_images):
        """Calculate LPIPS perceptual distance"""
        if not self.lpips_available:
            return -1
            
        with torch.no_grad():
            real_tensors = []
            fake_tensors = []
            
            for real_img, fake_img in zip(real_images, fake_images):
                # Convert PIL to tensor
                if isinstance(real_img, Image.Image):
                    real_tensor = torch.from_numpy(np.array(real_img)).float() / 255.0
                    if len(real_tensor.shape) == 2:
                        real_tensor = real_tensor.unsqueeze(0)
                    real_tensor = real_tensor.repeat(3, 1, 1) if real_tensor.shape[0] == 1 else real_tensor
                    real_tensors.append(real_tensor)
                
                if isinstance(fake_img, Image.Image):
                    fake_tensor = torch.from_numpy(np.array(fake_img)).float() / 255.0
                    if len(fake_tensor.shape) == 2:
                        fake_tensor = fake_tensor.unsqueeze(0)
                    fake_tensor = fake_tensor.repeat(3, 1, 1) if fake_tensor.shape[0] == 1 else fake_tensor
                    fake_tensors.append(fake_tensor)
            
            real_batch = torch.stack(real_tensors).to(self.device)
            fake_batch = torch.stack(fake_tensors).to(self.device)
            
            # Normalize to [-1, 1]
            real_batch = real_batch * 2.0 - 1.0
            fake_batch = fake_batch * 2.0 - 1.0
            
            lpips_scores = self.lpips_model(real_batch, fake_batch)
            return float(lpips_scores.mean())
    
    def calculate_ssim_psnr(self, real_images, fake_images):
        """Calculate SSIM and PSNR"""
        ssim_scores = []
        psnr_scores = []
        
        for real_img, fake_img in zip(real_images, fake_images):
            # Convert PIL to numpy
            if isinstance(real_img, Image.Image):
                real_np = np.array(real_img).astype(np.float32) / 255.0
            else:
                real_np = real_img
                
            if isinstance(fake_img, Image.Image):
                fake_np = np.array(fake_img).astype(np.float32) / 255.0
            else:
                fake_np = fake_img
            
            # Calculate SSIM
            ssim_score = ssim(real_np, fake_np, data_range=1.0)
            ssim_scores.append(ssim_score)
            
            # Calculate PSNR
            psnr_score = psnr(real_np, fake_np, data_range=1.0)
            psnr_scores.append(psnr_score)
        
        return np.mean(ssim_scores), np.mean(psnr_scores)


def load_reference_images(reference_dir, max_images=100):
    """Load reference images from directory"""
    if not reference_dir or not Path(reference_dir).exists():
        return {}
    
    reference_images = {}
    classes = ['benign', 'malignant', 'normal']
    
    for class_name in classes:
        class_dir = Path(reference_dir) / class_name
        if class_dir.exists():
            class_images = []
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            
            for ext in image_extensions:
                for img_path in class_dir.glob(ext):
                    if not img_path.name.endswith('_mask.png'):
                        try:
                            img = Image.open(img_path).convert('L')
                            class_images.append(img)
                            if len(class_images) >= max_images // len(classes):
                                break
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                if len(class_images) >= max_images // len(classes):
                    break
            
            reference_images[class_name] = class_images
            print(f"Loaded {len(class_images)} {class_name} reference images")
    
    return reference_images


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model configuration from args
    args = checkpoint.get('args', {})
    
    # Create model with same configuration as training
    model = UNet2DModel(
        sample_size=args.get('resolution', 512),
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512, 768, 1024),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D", 
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=16,
        norm_eps=1e-6,
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
    )
    
    # Create pipeline
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    
    return pipeline, checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)


def evaluate_checkpoint(checkpoint_path, reference_images_by_class, device, num_samples=16, num_inference_steps=50):
    """Evaluate a single checkpoint against all classes"""
    print(f"\n🔄 Evaluating checkpoint: {Path(checkpoint_path).name}")
    
    try:
        # Load model from checkpoint
        pipeline, epoch, global_step = load_checkpoint_model(checkpoint_path, device)
        
        # Generate images
        generator = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            generated_images = pipeline(
                batch_size=num_samples,
                generator=generator,
                num_inference_steps=num_inference_steps,
            ).images
        
        # Calculate metrics for each class and overall
        all_metrics = {}
        
        if reference_images_by_class:
            image_metrics = ImageMetrics(device=device)
            
            # Calculate metrics for each class
            for class_name, ref_images in reference_images_by_class.items():
                if ref_images:
                    # Take subset of reference images matching generated count
                    samples_per_class = min(len(ref_images), len(generated_images))
                    ref_subset = ref_images[:samples_per_class]
                    gen_subset = generated_images[:samples_per_class]
                    
                    # Calculate metrics for this class
                    fid = image_metrics.calculate_fid_simple(ref_subset, gen_subset)
                    ssim_score, psnr_score = image_metrics.calculate_ssim_psnr(ref_subset, gen_subset)
                    lpips_score = image_metrics.calculate_lpips(ref_subset, gen_subset)
                    
                    class_metrics = {
                        'fid': fid,
                        'ssim': ssim_score,
                        'psnr': psnr_score,
                        'lpips': lpips_score if lpips_score > 0 else None
                    }
                    
                    all_metrics[f'{class_name}'] = class_metrics
                    
                    print(f"   📊 {class_name.upper()} metrics:")
                    print(f"      FID: {fid:.4f}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f}")
                    if lpips_score > 0:
                        print(f"      LPIPS: {lpips_score:.4f}")
            
            # Calculate overall average metrics
            if all_metrics:
                overall_metrics = {}
                for metric_name in ['fid', 'ssim', 'psnr', 'lpips']:
                    values = []
                    for class_metrics in all_metrics.values():
                        if class_metrics.get(metric_name) is not None:
                            values.append(class_metrics[metric_name])
                    
                    if values:
                        overall_metrics[metric_name] = np.mean(values)
                    else:
                        overall_metrics[metric_name] = None
                
                all_metrics['overall'] = overall_metrics
                
                print(f"   🎯 OVERALL AVERAGE:")
                print(f"      FID: {overall_metrics['fid']:.4f}, SSIM: {overall_metrics['ssim']:.4f}, PSNR: {overall_metrics['psnr']:.2f}")
                if overall_metrics['lpips'] is not None:
                    print(f"      LPIPS: {overall_metrics['lpips']:.4f}")
        
        result = {
            'checkpoint_path': checkpoint_path,
            'epoch': epoch,
            'global_step': global_step,
            'metrics': all_metrics,
            'generated_images': generated_images
        }
        
        print(f"✅ Checkpoint evaluated - Epoch: {epoch}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error evaluating checkpoint {checkpoint_path}: {e}")
        return None


def find_best_checkpoint(results):
    """Find the best checkpoint based on combined metrics across all classes"""
    if not results or not results[0]['metrics']:
        # If no metrics, return the latest checkpoint
        return max(results, key=lambda x: x['epoch'])
    
    # Calculate composite score for each checkpoint
    best_checkpoint = None
    best_score = float('-inf')
    
    for result in results:
        metrics = result['metrics']
        
        # Use overall average metrics for scoring
        overall_metrics = metrics.get('overall', {})
        
        if not overall_metrics:
            # If no overall metrics, calculate from individual classes
            individual_scores = []
            for class_name in ['benign', 'malignant', 'normal']:
                class_metrics = metrics.get(class_name, {})
                if class_metrics:
                    class_score = calculate_composite_score(class_metrics)
                    individual_scores.append(class_score)
            
            if individual_scores:
                score = np.mean(individual_scores)
            else:
                score = 0
        else:
            # Use overall metrics for scoring
            score = calculate_composite_score(overall_metrics)
        
        result['composite_score'] = score
        
        if score > best_score:
            best_score = score
            best_checkpoint = result
    
    return best_checkpoint


def calculate_composite_score(metrics):
    """Calculate composite score from metrics"""
    score = 0
    
    # FID (lower is better) - invert and normalize
    if metrics.get('fid') is not None:
        fid_score = -metrics['fid']  # Invert so higher is better
        score += fid_score * 0.3
    
    # SSIM (higher is better)
    if metrics.get('ssim') is not None:
        score += metrics['ssim'] * 100 * 0.3  # Scale up SSIM
    
    # PSNR (higher is better)
    if metrics.get('psnr') is not None:
        score += metrics['psnr'] * 0.2
    
    # LPIPS (lower is better) - invert
    if metrics.get('lpips') is not None:
        lpips_score = -metrics['lpips'] * 100  # Invert and scale
        score += lpips_score * 0.2
    
    return score


def create_evaluation_summary(results, best_checkpoint, output_dir):
    """Create evaluation summary with class-specific results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*100)
    print("📊 CHECKPOINT EVALUATION SUMMARY BY CLASS")
    print("="*100)
    
    # Sort results by epoch for display
    results_sorted = sorted(results, key=lambda x: x['epoch'])
    
    # Print header
    print(f"{'Epoch':<8} {'Overall':<50} {'Benign':<50} {'Malignant':<50} {'Normal':<50} {'Score':<8}")
    print(f"{'':^8} {'FID   SSIM  PSNR  LPIPS':<50} {'FID   SSIM  PSNR  LPIPS':<50} {'FID   SSIM  PSNR  LPIPS':<50} {'FID   SSIM  PSNR  LPIPS':<50}")
    print("-" * 250)
    
    for result in results_sorted:
        epoch = result['epoch']
        metrics = result['metrics']
        score = result.get('composite_score', 0)
        
        marker = "🏆" if result == best_checkpoint else "  "
        
        # Format overall metrics
        overall = metrics.get('overall', {})
        overall_str = format_metrics_line(overall)
        
        # Format class-specific metrics
        benign_str = format_metrics_line(metrics.get('benign', {}))
        malignant_str = format_metrics_line(metrics.get('malignant', {}))
        normal_str = format_metrics_line(metrics.get('normal', {}))
        
        print(f"{marker} {epoch:<6} {overall_str:<50} {benign_str:<50} {malignant_str:<50} {normal_str:<50} {score:>6.2f}")
    
    print("\n" + "="*100)
    if best_checkpoint:
        print(f"🏆 BEST CHECKPOINT: {Path(best_checkpoint['checkpoint_path']).name}")
        print(f"   Epoch: {best_checkpoint['epoch']}")
        print(f"   Global Step: {best_checkpoint['global_step']}")
        print(f"   Composite Score: {best_checkpoint.get('composite_score', 0):.2f}")
        print(f"   Path: {best_checkpoint['checkpoint_path']}")
        
        if best_checkpoint['metrics']:
            print("\n   📊 DETAILED METRICS:")
            
            # Print overall metrics
            overall = best_checkpoint['metrics'].get('overall', {})
            if overall:
                print(f"   🎯 OVERALL AVERAGE:")
                print(f"      FID: {overall.get('fid', 0):.4f}")
                print(f"      SSIM: {overall.get('ssim', 0):.4f}")
                print(f"      PSNR: {overall.get('psnr', 0):.2f}")
                if overall.get('lpips') is not None:
                    print(f"      LPIPS: {overall.get('lpips', 0):.4f}")
            
            # Print class-specific metrics
            for class_name in ['benign', 'malignant', 'normal']:
                class_metrics = best_checkpoint['metrics'].get(class_name, {})
                if class_metrics:
                    print(f"   📈 {class_name.upper()}:")
                    print(f"      FID: {class_metrics.get('fid', 0):.4f}")
                    print(f"      SSIM: {class_metrics.get('ssim', 0):.4f}")
                    print(f"      PSNR: {class_metrics.get('psnr', 0):.2f}")
                    if class_metrics.get('lpips') is not None:
                        print(f"      LPIPS: {class_metrics.get('lpips', 0):.4f}")
                    
    print("="*100)


def format_metrics_line(metrics):
    """Format metrics for table display"""
    if not metrics:
        return "N/A   N/A   N/A   N/A  "
    
    fid_str = f"{metrics.get('fid', 0):.3f}" if metrics.get('fid') is not None else "N/A"
    ssim_str = f"{metrics.get('ssim', 0):.3f}" if metrics.get('ssim') is not None else "N/A"
    psnr_str = f"{metrics.get('psnr', 0):.1f}" if metrics.get('psnr') is not None else "N/A"
    lpips_str = f"{metrics.get('lpips', 0):.3f}" if metrics.get('lpips') is not None else "N/A"
    
    return f"{fid_str:>5} {ssim_str:>5} {psnr_str:>4} {lpips_str:>5}"


def save_evaluation_results(results, best_checkpoint, output_dir):
    """Save detailed evaluation results to JSON"""
    output_dir = Path(output_dir)
    results_sorted = sorted(results, key=lambda x: x['epoch'])
    
    # Save results to JSON
    summary_data = {
        'evaluation_timestamp': str(np.datetime64('now')),
        'best_checkpoint': {
            'path': best_checkpoint['checkpoint_path'],
            'epoch': best_checkpoint['epoch'],
            'global_step': best_checkpoint['global_step'],
            'metrics': best_checkpoint['metrics'],
            'composite_score': best_checkpoint.get('composite_score', 0)
        } if best_checkpoint else None,
        'all_results': [
            {
                'checkpoint_path': r['checkpoint_path'],
                'epoch': r['epoch'],
                'global_step': r['global_step'],
                'metrics': r['metrics'],
                'composite_score': r.get('composite_score', 0)
            }
            for r in results_sorted
        ]
    }
    
    summary_path = output_dir / "checkpoint_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"📄 Detailed results saved to: {summary_path}")


def save_best_checkpoint_images(best_checkpoint, output_dir):
    """Save images from the best checkpoint"""
    output_dir = Path(output_dir)
    best_images_dir = output_dir / "best_checkpoint_images"
    best_images_dir.mkdir(exist_ok=True)
    
    images = best_checkpoint['generated_images']
    
    # Save individual images
    for i, image in enumerate(images):
        save_path = best_images_dir / f"best_generated_{i:04d}.png"
        image.save(save_path)
    
    # Create grid
    if len(images) > 1:
        grid_size = int(np.ceil(np.sqrt(len(images))))
        img_width, img_height = images[0].size
        
        grid_img = Image.new('RGB', (grid_size * img_width, grid_size * img_height), (255, 255, 255))
        
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            if img.mode == 'L':
                img = img.convert('RGB')
            grid_img.paste(img, (col * img_width, row * img_height))
        
        grid_path = best_images_dir / "best_checkpoint_grid.png"
        grid_img.save(grid_path)
        print(f"🖼️ Best checkpoint images saved to: {best_images_dir}")
        print(f"🔲 Grid image: {grid_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test/Inference script for trained DDPM model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/Desktop/JY/PAADI/diffusers/ddpm-busi-512",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints for evaluation"
    )
    parser.add_argument(
        "--reference_data_dir",
        type=str,
        default=None,
        help="Directory containing reference images for metrics calculation"
    )
    parser.add_argument(
        "--evaluate_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints and find the best one"
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Only calculate metrics, don't save individual images"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Evaluate all checkpoints mode
    if args.evaluate_all_checkpoints and args.checkpoint_dir:
        print("🔍 Evaluating all checkpoints mode")
        
        # Load reference images if provided
        reference_images = load_reference_images(args.reference_data_dir)
        
        # Find all checkpoints
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {checkpoint_dir}")
            return
        
        # Sort checkpoints by epoch
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Found {len(checkpoint_files)} checkpoints")
        
        # Evaluate all checkpoints
        results = []
        for checkpoint_path in checkpoint_files:
            result = evaluate_checkpoint(
                checkpoint_path, 
                reference_images, 
                device,
                num_samples=args.num_samples,
                num_inference_steps=args.num_inference_steps
            )
            if result:
                results.append(result)
        
        if not results:
            print("❌ No checkpoints could be evaluated")
            return
        
        # Find best checkpoint based on metrics
        best_checkpoint = find_best_checkpoint(results)
        
        # Create summary
        create_evaluation_summary(results, best_checkpoint, args.output_dir)
        
        # Save detailed results to JSON
        save_evaluation_results(results, best_checkpoint, args.output_dir)
        
        # Save images from best checkpoint if not metrics_only
        if not args.metrics_only and best_checkpoint:
            save_best_checkpoint_images(best_checkpoint, args.output_dir)
        
        return
    
    # Standard single model inference
    try:
        pipeline = DDPMPipeline.from_pretrained(args.model_path)
        pipeline = pipeline.to(device)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model has been trained and saved properly.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    print(f"Generating {args.num_samples} samples...")
    print(f"Using {args.num_inference_steps} inference steps")
    
    # Generate images
    with torch.no_grad():
        images = pipeline(
            batch_size=args.num_samples,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        ).images
    
    # Save generated images
    for i, image in enumerate(images):
        save_path = output_dir / f"generated_sample_{i:04d}.png"
        image.save(save_path)
        print(f"Saved: {save_path}")
    
    print(f"All {len(images)} images saved to {output_dir}")
    
    # Create a grid of images for easy viewing
    if len(images) > 1:
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(images))))
        img_width, img_height = images[0].size
        
        # Create grid - convert grayscale to RGB for proper display
        grid_img = Image.new('RGB', (grid_size * img_width, grid_size * img_height), (255, 255, 255))
        
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            # Convert grayscale to RGB if needed
            if img.mode == 'L':
                img = img.convert('RGB')
            grid_img.paste(img, (col * img_width, row * img_height))
        
        grid_path = output_dir / "generated_grid.png"
        grid_img.save(grid_path)
        print(f"Grid image saved: {grid_path}")

if __name__ == "__main__":
    main()