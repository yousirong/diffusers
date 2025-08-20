#!/usr/bin/env python3
"""
Debug Model Script - Check model files without loading
"""

import os
import torch
from pathlib import Path
import logging
import sys

# Use GPU 2 only
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_structure():
    """Debug model structure without loading"""
    logger.info("=== Model Structure Debug ===")
    
    base_path = Path("/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100")
    
    logger.info(f"Base path: {base_path}")
    logger.info(f"Base path exists: {base_path.exists()}")
    
    if not base_path.exists():
        logger.error("Base path does not exist!")
        return False
    
    # Check directory structure
    logger.info("Directory structure:")
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            logger.info(f"  📁 {item.name}/")
            # List first few files in subdirectory
            sub_items = list(item.iterdir())[:5]
            for sub_item in sub_items:
                logger.info(f"    - {sub_item.name}")
            if len(sub_items) < len(list(item.iterdir())):
                remaining = len(list(item.iterdir())) - len(sub_items)
                logger.info(f"    ... and {remaining} more files")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            logger.info(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    # Check specific important files
    important_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.bin"
    ]
    
    logger.info("\nChecking important files:")
    for file_path in important_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        size = ""
        if exists:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        logger.info(f"  {file_path}: {'✅' if exists else '❌'}{size}")
    
    # Check best model
    best_model_path = base_path / "best_model"
    logger.info(f"\nBest model directory: {best_model_path.exists()}")
    
    if best_model_path.exists():
        logger.info("Best model structure:")
        for item in sorted(best_model_path.iterdir()):
            if item.is_dir():
                logger.info(f"  📁 {item.name}/")
                sub_items = list(item.iterdir())[:3]
                for sub_item in sub_items:
                    size_mb = sub_item.stat().st_size / (1024 * 1024) if sub_item.is_file() else 0
                    logger.info(f"    - {sub_item.name}" + (f" ({size_mb:.1f} MB)" if size_mb > 0 else ""))
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    # Check epoch directories
    epoch_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    if epoch_dirs:
        logger.info(f"\nFound {len(epoch_dirs)} epoch directories:")
        for epoch_dir in sorted(epoch_dirs)[-3:]:  # Show last 3
            logger.info(f"  📁 {epoch_dir.name}")
    
    return True

def check_config_files():
    """Check configuration files for corruption"""
    logger.info("=== Configuration Files Check ===")
    
    base_path = Path("/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100")
    best_model_path = base_path / "best_model"
    
    config_files = []
    
    # Check both base and best model configs
    for path in [base_path, best_model_path]:
        if path.exists():
            config_files.extend([
                path / "model_index.json",
                path / "scheduler/scheduler_config.json", 
                path / "unet/config.json"
            ])
    
    for config_file in config_files:
        if config_file.exists():
            try:
                import json
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"✅ {config_file.relative_to(base_path)}: Valid JSON")
                
                # Show key info for UNet config
                if config_file.name == "config.json" and "unet" in str(config_file):
                    logger.info(f"    Sample size: {config_data.get('sample_size', 'Unknown')}")
                    logger.info(f"    In channels: {config_data.get('in_channels', 'Unknown')}")
                    logger.info(f"    Out channels: {config_data.get('out_channels', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"❌ {config_file.relative_to(base_path)}: Invalid - {e}")
        else:
            logger.warning(f"⚠️  {config_file.relative_to(base_path)}: Not found")

def main():
    logger.info("=== DDPM Model Debug ===")
    
    try:
        # Check basic info
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
        
        # Debug model structure
        if not debug_model_structure():
            return False
        
        # Check config files
        check_config_files()
        
        logger.info("✅ Debug completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)