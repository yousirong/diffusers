#!/bin/bash

# A100 80GB GPU에 최적화된 512x512 초음파 DDPM 훈련 스크립트
# 사용법: bash train_512_a100.sh

# GPU 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

# 훈련 실행
python train_ultrasound_ddpm.py \
    --train_data_dir "/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/data/train_CN_CY_ALL" \
    --output_dir "./ddpm-ultrasound-512-a100" \
    --resolution 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --num_epochs 200 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --save_image_epochs 10 \
    --save_model_epochs 20 \
    --mixed_precision "bf16" \
    --augment_factor 8 \
    --num_workers 8 \
    --pin_memory \
    --use_ema \
    --checkpointing_steps 2000 \
    --max_grad_norm 1.0 \
    --seed 42

echo "훈련 완료! 모델이 ./ddpm-ultrasound-512-a100 에 저장되었습니다."