```bash
# Training with TensorBoard logging
python train_ddpm_512.py \
  --dataset_path "/home/ubuntu/Desktop/JY/PAADI/diffusers/Dataset_BUSI_with_GT" \
  --output_dir "ddpm-busi-512-enhanced" \
  --num_epochs 200 \
  --train_batch_size 6 \
  --save_checkpoint_epochs 5 \
  --enable_data_curation \
  --min_aesthetic_score 0.15 \
  --samples_per_class 500 \
  --enable_advanced_metrics \
  --metrics_frequency 5 \
  --learning_rate 1e-4 \
  --advanced_augmentation \
  --mixed_precision fp16 \
  --enable_tensorboard \
  --tensorboard_log_dir "tensorboard_logs"

# View training progress in TensorBoard
tensorboard --logdir ./diffusers/ddpm-busi-512-enhanced/tensorboard_logs --bind_all  --port 6006
# Testing
  python test_ddpm_512.py \
    --evaluate_all_checkpoints \
    --checkpoint_dir ./ddpm-busi-512-enhanced/checkpoints \
    --reference_data_dir ./Dataset_BUSI_with_GT \
    --num_samples 24 \
    --output_dir class_evaluation_results

```