import os
import torch
from PIL import Image
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

def load_model(model_path, device):
    """모델 로드"""
    model = UNet2DModel(
        sample_size=512,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512, 1024),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="linear",
        prediction_type="epsilon"
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, scheduler

def denoise_image(model, scheduler, image_path, device):
    """이미지 디노이징"""
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('L').resize((512, 512))
    img_array = np.array(img) / 255.0

    # [0,1] -> [-1,1] 정규화, 3채널로 변환
    img_tensor = torch.tensor(img_array * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

    # DDPM 디노이징
    with torch.no_grad():
        denoised = img_tensor.clone()
        for t in tqdm(reversed(range(100)), desc="Denoising", leave=False):
            noise_pred = model(denoised, torch.tensor([t], device=device)).sample
            denoised = scheduler.step(noise_pred, t, denoised).prev_sample

    return img_tensor, denoised

# 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/juneyonglee/Desktop/diffusers/ddmp_checkpoints_gaussian_pure/best_model_gaussian.pt"

    # 모델 로드
    model, scheduler = load_model(model_path, device)
    print("Model loaded successfully!")

    # 이미지 처리할 디렉토리
    input_dir = "data/train_gt"  # V3-V7 이미지가 있는 폴더
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)

    # V3-V7 이미지 찾기
    image_files = [f for f in os.listdir(input_dir)
                   if f.endswith(('.bmp','.png', '.jpg', '.jpeg')) and any(v in f for v in ['V3', 'V4', 'V5', 'V6', 'V7'])]

    print(f"Found {len(image_files)} V3-V7 images")

    # 각 이미지 처리
    for img_file in image_files:
        print(f"Processing: {img_file}")
        img_path = os.path.join(input_dir, img_file)

        # 디노이징
        original, denoised = denoise_image(model, scheduler, img_path, device)

        # 결과 저장
        base_name = os.path.splitext(img_file)[0]

        # 비교 이미지 저장
        comparison = torch.cat([original, denoised], dim=0)
        save_image(comparison,
                  os.path.join(output_dir, f"{base_name}_result.png"),
                  normalize=True, value_range=(-1, 1), nrow=2)

        # 디노이징된 이미지만 저장
        save_image(denoised,
                  os.path.join(output_dir, f"{base_name}_denoised.png"),
                  normalize=True, value_range=(-1, 1))

    print(f"✅ Inference completed! Results saved in {output_dir}")