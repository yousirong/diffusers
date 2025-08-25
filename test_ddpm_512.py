import argparse
import torch
from PIL import Image
from pathlib import Path
import numpy as np

from diffusers import DDPMPipeline

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
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load the trained pipeline
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