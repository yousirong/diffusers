#!/usr/bin/env python3
"""
Debug script to isolate the bus error issue
"""

import os
import sys
import torch
from pathlib import Path

print("Starting debug script...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Testing basic imports...")
    from diffusers import DDPMPipeline, DDPMScheduler
    print("✓ Diffusers import successful")
    
    print("Testing model path...")
    model_path = Path("/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100")
    print(f"Model path exists: {model_path.exists()}")
    
    if model_path.exists():
        print("Listing model directory contents...")
        for item in model_path.iterdir():
            print(f"  {item.name}")
    
    print("Testing best model path...")
    best_model_path = model_path / "best_model"
    print(f"Best model path exists: {best_model_path.exists()}")
    
    if best_model_path.exists():
        print("Listing best model directory contents...")
        for item in best_model_path.iterdir():
            print(f"  {item.name}")
    
    print("Attempting to load pipeline...")
    try:
        pipeline = DDPMPipeline.from_pretrained(best_model_path)
        print("✓ Pipeline loaded successfully")
        
        print("Testing pipeline move to CPU...")
        pipeline = pipeline.to('cpu')
        print("✓ Pipeline moved to CPU successfully")
        
    except Exception as e:
        print(f"✗ Pipeline loading failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ Error during debug: {e}")
    import traceback
    traceback.print_exc()

print("Debug script completed.")