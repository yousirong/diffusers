import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

VERSION_RADIUS = {
    "V3": [(220, 85)],
    "V4": [(130, 50)],
    "V5": [(90, 30)],
    "V6": [(60, 20)],
    "V7": [(45, 15)]
}

def get_version(filename: str) -> str:
    match = re.search(r'V[3-7]', filename)
    return match.group(0) if match else "Unknown"

def generate_donut_noise_pair(img_path, image_size=512):
    img = Image.open(img_path).convert('L').resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    h, w = img_array.shape
    version = get_version(os.path.basename(img_path))
    radius_pairs = VERSION_RADIUS.get(version, [(int(min(h, w) * 0.3), int(min(h, w) * 0.1))])
    outer_r, inner_r = radius_pairs[0]
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = (dist > inner_r) & (dist <= outer_r)

    # 도넛 노이즈 이미지 생성
    noise = np.random.normal(0, 0.3, size=(h, w))  # 노이즈 강도 조정
    noisy_img = img_array.copy()
    noisy_img[mask] += noise[mask]
    noisy_img = np.clip(noisy_img, 0, 1)  # 값 범위 제한

    return noisy_img.astype(np.float32), img_array.astype(np.float32)

class DonutMaskedNoiseDataset(Dataset):
    def __init__(self, image_dir, image_size=512):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_files[idx])
        noisy_img, clean_img = generate_donut_noise_pair(path, self.image_size)

        # 정규화: [0,1] -> [-1,1]
        noisy_tensor = torch.tensor(noisy_img * 2.0 - 1.0).unsqueeze(0).repeat(3, 1, 1)
        clean_tensor = torch.tensor(clean_img * 2.0 - 1.0).unsqueeze(0).repeat(3, 1, 1)

        # 파일명에서 버전 정보도 반환 (디버깅용)
        version = get_version(self.image_files[idx])

        return noisy_tensor, clean_tensor, version

# 설정
image_dir = "data/train"
output_dir = "ddpm_checkpoints"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 1
lr = 1e-4
image_size = 512

print(f"Using device: {device}")
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU mode")

# 모델 정의 - 도넛 노이즈 제거용 UNet
model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512, 1024),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    attention_head_dim=8,
).to(device)

# DDPM 스케줄러 (더 적은 단계로 조정)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,  # 도넛 노이즈 제거용으로 단계 축소
    beta_schedule="linear",
    prediction_type="epsilon"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

# Mixed precision을 위한 scaler
scaler = GradScaler()

# 데이터 로딩
dataset = DonutMaskedNoiseDataset(image_dir, image_size)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

print(f"Dataset size: {len(dataset)}")
print(f"Batches per epoch: {len(dataloader)}")

# 버전별 분포 확인
version_counts = {}
for _, _, version in dataset:
    version_counts[version] = version_counts.get(version, 0) + 1
print(f"Version distribution: {version_counts}")

# 학습 루프
model.train()
best_loss = float('inf')

for epoch in range(epochs):
    print(f"🌟 Epoch {epoch+1}/{epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")

    epoch_loss = 0.0
    version_losses = {}
    progress_bar = tqdm(dataloader, desc=f"Training")

    for batch_idx, (noisy, clean, versions) in enumerate(progress_bar):
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

        # DDPM 노이즈 추가 (도넛 노이즈 위에 추가)
        diffusion_noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (clean.shape[0],), device=device
        ).long()

        # Mixed precision으로 forward pass
        with autocast():
            # 깨끗한 이미지에 diffusion 노이즈 추가
            noisy_diffusion = noise_scheduler.add_noise(clean, diffusion_noise, timesteps)

            # 모델이 diffusion 노이즈를 예측하도록 훈련
            pred_noise = model(noisy_diffusion, timesteps).sample
            loss = torch.nn.functional.mse_loss(pred_noise, diffusion_noise)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # 버전별 손실 추적
        for version in versions:
            if version not in version_losses:
                version_losses[version] = []
            version_losses[version].append(loss.item())

        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        # 메모리 정리
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average loss: {avg_loss:.6f}")

    # 버전별 평균 손실 출력
    for version, losses in version_losses.items():
        avg_version_loss = sum(losses) / len(losses)
        print(f"  {version} avg loss: {avg_version_loss:.6f} ({len(losses)} samples)")

    # Learning rate 업데이트
    scheduler.step()

    # 도넛 노이즈 제거 테스트 (매 5 에포크마다)
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            print("Testing donut noise removal...")

            # 테스트용 도넛 노이즈 이미지 생성
            test_noisy, test_clean, test_version = dataset[0]
            test_noisy = test_noisy.unsqueeze(0).to(device)

            # DDPM 디노이징 과정
            denoised = test_noisy.clone()
            for t in tqdm(reversed(range(noise_scheduler.config.num_train_timesteps)),
                         desc="Denoising", leave=False):
                with autocast():
                    noise_pred = model(denoised, torch.tensor([t], device=device)).sample
                denoised = noise_scheduler.step(noise_pred, t, denoised).prev_sample

            # 결과 저장
            comparison = torch.cat([test_noisy, denoised, test_clean.unsqueeze(0).to(device)], dim=0)
            save_image(comparison,
                      os.path.join(output_dir, f"comparison_epoch{epoch+1}_{test_version}.png"),
                      normalize=True, value_range=(-1, 1), nrow=3)

        model.train()

    # 모델 저장
    if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'version_losses': version_losses,
            }, os.path.join(output_dir, f"best_model.pt"))
            print(f"💾 Best model saved! Loss: {avg_loss:.6f}")

        # 일반 체크포인트 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'version_losses': version_losses,
        }, os.path.join(output_dir, f"ddpm_epoch{epoch+1}.pt"))

    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

print("🎉 Training completed!")
print(f"Best loss achieved: {best_loss:.6f}")
print(f"Models saved in: {output_dir}")

# 최종 버전별 통계 출력
print("\n📊 Final version statistics:")
final_counts = {}
for _, _, version in dataset:
    final_counts[version] = final_counts.get(version, 0) + 1
for version, count in final_counts.items():
    radius = VERSION_RADIUS.get(version, [(0, 0)])[0]
    print(f"  {version}: {count} images, donut radius: outer={radius[0]}, inner={radius[1]}")