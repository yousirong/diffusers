```bash
# Training with TensorBoard logging
python train_ddpm_512.py \
  --dataset_path "/home/ubuntu/Desktop/JY/PAADI/diffusers/Dataset_BUSI_with_GT" \
  --output_dir "ddpm-busi-512-enhanced" \
  --num_epochs 200 \
  --train_batch_size 4 \
  --learning_rate 2e-5 \
  --use_v_prediction \
  --use_perceptual_loss \
  --advanced_augmentation \
  --mixed_precision fp16 \
  --enable_tensorboard \
  --tensorboard_log_dir "tensorboard_logs"

# View training progress in TensorBoard
tensorboard --logdir ddpm-busi-512-enhanced/tensorboard_logs

# Testing
python test_ddpm_512.py --num_samples 10 --num_inference_steps 100 --output_dir my_test_results
```