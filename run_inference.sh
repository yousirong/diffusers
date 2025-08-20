#!/bin/bash

# DDPM Test Inference Script
# Usage: bash run_inference.sh

set -e

echo "Starting DDPM Test Inference..."
export CUDA_VISIBLE_DEVICES=2

# Configuration
MODEL_PATH="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100"
TEST_DIR="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/data/test_CN"
OUTPUT_DIR="./inference_results_$(date +%Y%m%d_%H%M%S)"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Please train the model first using: bash train_512_a100.sh"
    exit 1
fi

# Check if test data exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory not found: $TEST_DIR"
    echo "Please check the test data path"
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo "Test data path: $TEST_DIR"
echo "Output directory: $OUTPUT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run inference
python test_inference.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_test_images 10 \
    --num_generate 8 \
    --num_inference_steps 100 \
    --noise_strength 0.8 \
    --seed 42 \
    --device cuda \
    --mode all

echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"

# Show results structure
echo -e "\nResults structure:"
find "$OUTPUT_DIR" -name "*.png" | head -20